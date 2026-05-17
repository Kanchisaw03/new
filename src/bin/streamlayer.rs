use std::collections::BTreeSet;
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::{anyhow, bail, Context, Result};
use bytemuck::{pod_read_unaligned, Pod};
use clap::Parser;
use half::{bf16, f16};
use llama_gguf::gguf::{GgmlType, GgufData, GgufFile, TensorInfo};
use llama_gguf::tensor::quant::{
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    dequantize_q8_1, dequantize_q8_k, BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K,
    BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K,
};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "StreamLayer planner for exact-Q4 streamed GGUF inference"
)]
struct Args {
    #[arg(long)]
    gguf: PathBuf,

    #[arg(long, default_value_t = 256)]
    context_tokens: usize,

    #[arg(long, default_value_t = 64)]
    head_chunk_mb: usize,

    #[arg(long, default_value_t = 200)]
    peak_budget_mb: usize,

    #[arg(long, default_value_t = 3.5)]
    nvme_gbps: f64,

    #[arg(long, default_value_t = 16.0)]
    cpu_gflops: f64,

    #[arg(long, default_value_t = 5)]
    activation_scratch_mb: usize,

    #[arg(long)]
    show_tensor_names: bool,

    #[arg(long)]
    trace_runtime: bool,

    #[arg(long, default_value_t = 0)]
    token_id: usize,
}

#[derive(Debug, Clone)]
struct TensorSlot {
    name: String,
    offset: u64,
    size: usize,
    dtype: GgmlType,
    rows: usize,
    cols: usize,
}

#[derive(Debug, Default, Clone)]
struct LayerTensorLayout {
    attn_q: Option<TensorSlot>,
    attn_k: Option<TensorSlot>,
    attn_v: Option<TensorSlot>,
    attn_o: Option<TensorSlot>,
    attn_norm: Option<TensorSlot>,
    ffn_gate: Option<TensorSlot>,
    ffn_up: Option<TensorSlot>,
    ffn_down: Option<TensorSlot>,
    ffn_norm: Option<TensorSlot>,
}

impl LayerTensorLayout {
    fn attention_slots(&self) -> Vec<&TensorSlot> {
        let mut slots = Vec::new();
        if let Some(slot) = &self.attn_q {
            slots.push(slot);
        }
        if let Some(slot) = &self.attn_k {
            slots.push(slot);
        }
        if let Some(slot) = &self.attn_v {
            slots.push(slot);
        }
        if let Some(slot) = &self.attn_o {
            slots.push(slot);
        }
        if let Some(slot) = &self.attn_norm {
            slots.push(slot);
        }
        slots
    }

    fn feed_forward_slots(&self) -> Vec<&TensorSlot> {
        let mut slots = Vec::new();
        if let Some(slot) = &self.ffn_gate {
            slots.push(slot);
        }
        if let Some(slot) = &self.ffn_up {
            slots.push(slot);
        }
        if let Some(slot) = &self.ffn_down {
            slots.push(slot);
        }
        if let Some(slot) = &self.ffn_norm {
            slots.push(slot);
        }
        slots
    }

    fn total_bytes(&self) -> usize {
        self.attention_slots()
            .into_iter()
            .chain(self.feed_forward_slots().into_iter())
            .map(|slot| slot.size)
            .sum()
    }

    fn attention_bytes(&self) -> usize {
        self.attention_slots().into_iter().map(|slot| slot.size).sum()
    }

    fn feed_forward_bytes(&self) -> usize {
        self.feed_forward_slots().into_iter().map(|slot| slot.size).sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorClass {
    Attention,
    FeedForward,
    Normalization,
    Other,
}

#[derive(Debug, Clone)]
struct ModelConfig {
    architecture: String,
    hidden_size: usize,
    feed_forward_size: usize,
    layer_count: usize,
    attention_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
}

#[derive(Debug, Default, Clone)]
struct LayerSummary {
    attention_bytes: usize,
    feed_forward_bytes: usize,
    normalization_bytes: usize,
    other_bytes: usize,
    attention_tensors: usize,
    feed_forward_tensors: usize,
    normalization_tensors: usize,
    other_tensors: usize,
}

#[derive(Debug)]
struct StreamPlan {
    config: ModelConfig,
    context_tokens: usize,
    weight_type: String,
    layer_summaries: Vec<LayerSummary>,
    layer_layouts: Vec<LayerTensorLayout>,
    global_tensors: Vec<(String, usize)>,
    global_slots: Vec<TensorSlot>,
    max_attention_block_bytes: usize,
    max_feed_forward_block_bytes: usize,
    max_layer_total_bytes: usize,
    weight_buffer_bytes: usize,
    kv_cache_bytes: usize,
    activation_scratch_bytes: usize,
    head_name: String,
    head_bytes: usize,
    head_chunk_bytes: usize,
    head_chunk_count: usize,
    peak_ram_bytes: usize,
    estimated_layer_compute_ms: f64,
    estimated_layer_io_ms: f64,
    estimated_head_latency_ms: f64,
    estimated_token_latency_ms: f64,
    estimated_tokens_per_sec: f64,
}

type ModelPlan = StreamPlan;

struct Runtime<'a> {
    file: Arc<GgufFile>,
    plan: &'a ModelPlan,
    buf0: Vec<u8>,
    buf1: Vec<u8>,
    current_is_buf0: bool,
    next_is_buf0: bool,
    hidden: Vec<f32>,
    hidden_norm: Vec<f32>,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    attn_out_buf: Vec<f32>,
    ffn_gate_buf: Vec<f32>,
    ffn_up_buf: Vec<f32>,
    ffn_out_buf: Vec<f32>,
    row_buf: Vec<f32>,
    scores_buf: Vec<f32>,
    kv_keys: Vec<f32>,
    kv_values: Vec<f32>,
    position: usize,
}

impl<'a> Runtime<'a> {
    fn new(file: Arc<GgufFile>, plan: &'a ModelPlan) -> Self {
        let buffer_size = plan.weight_buffer_bytes.max(1);
        let hidden_size = plan.config.hidden_size;
        let kv_dim = plan.config.kv_dim;
        let feed_forward_size = plan.config.feed_forward_size;
        let kv_cache_entries = plan
            .config
            .layer_count
            .saturating_mul(plan.context_tokens)
            .saturating_mul(kv_dim);
        Self {
            file,
            plan,
            buf0: vec![0u8; buffer_size],
            buf1: vec![0u8; buffer_size],
            current_is_buf0: true,
            next_is_buf0: false,
            hidden: vec![0.0; hidden_size],
            hidden_norm: vec![0.0; hidden_size],
            q_buf: vec![0.0; hidden_size],
            k_buf: vec![0.0; kv_dim],
            v_buf: vec![0.0; kv_dim],
            attn_out_buf: vec![0.0; hidden_size],
            ffn_gate_buf: vec![0.0; feed_forward_size],
            ffn_up_buf: vec![0.0; feed_forward_size],
            ffn_out_buf: vec![0.0; hidden_size],
            row_buf: vec![0.0; feed_forward_size.max(hidden_size)],
            scores_buf: vec![0.0; plan.context_tokens.max(1)],
            kv_keys: vec![0.0; kv_cache_entries],
            kv_values: vec![0.0; kv_cache_entries],
            position: 0,
        }
    }

    fn current_buffer_mut(&mut self) -> &mut [u8] {
        if self.current_is_buf0 {
            self.buf0.as_mut_slice()
        } else {
            self.buf1.as_mut_slice()
        }
    }

    fn next_buffer_mut(&mut self) -> &mut [u8] {
        if self.next_is_buf0 {
            self.buf0.as_mut_slice()
        } else {
            self.buf1.as_mut_slice()
        }
    }

    fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.current_is_buf0, &mut self.next_is_buf0);
    }

    fn trace_token(&mut self, token_id: usize) -> Result<()> {
        println!("Runtime trace");
        println!("  seed token id: {}", token_id);
        println!("  double buffer size: {}", format_bytes(self.plan.weight_buffer_bytes));
        println!("  weight type: {}", self.plan.weight_type);
        println!();

        self.load_global_tensors()?;

        let token_embedding = self
            .file
            .data
            .get_tensor("token_embd.weight")
            .ok_or_else(|| anyhow!("missing token_embd.weight tensor"))?;
        load_token_embedding(&self.file, token_embedding, token_id, &mut self.hidden, &mut self.row_buf)?;
        self.position = 0;
        println!("seed hidden rms: {:.4}", vector_rms(&self.hidden));

        for layer_index in 0..self.plan.config.layer_count {
            let layout = &self.plan.layer_layouts[layer_index];
            println!("layer {layer_index}");

            let attn_bytes = self.prefetch_layout(layout, true, true)?;
            let current_buffer = if self.current_is_buf0 { "buf0" } else { "buf1" };
            self.describe_layout("  attention", layout.attention_slots(), attn_bytes, current_buffer);

            let attn_stage = if self.current_is_buf0 {
                &self.buf0[..attn_bytes]
            } else {
                &self.buf1[..attn_bytes]
            };
            let mut scratch = LayerScratch {
                hidden: &mut self.hidden,
                hidden_norm: &mut self.hidden_norm,
                q_buf: &mut self.q_buf,
                k_buf: &mut self.k_buf,
                v_buf: &mut self.v_buf,
                attn_out_buf: &mut self.attn_out_buf,
                ffn_gate_buf: &mut self.ffn_gate_buf,
                ffn_up_buf: &mut self.ffn_up_buf,
                ffn_out_buf: &mut self.ffn_out_buf,
                row_buf: &mut self.row_buf,
                scores_buf: &mut self.scores_buf,
                kv_keys: &mut self.kv_keys,
                kv_values: &mut self.kv_values,
            };
            let attn_stats = compute_attention_layer(
                &self.plan.config,
                layout,
                attn_stage,
                layer_index,
                self.position,
                &mut scratch,
            )?;
            println!("  compute attention: rms {:.4} -> {:.4}", attn_stats.input_rms, attn_stats.output_rms);

            let ffn_bytes = self.prefetch_layout(layout, false, false)?;
            let next_buffer = if self.next_is_buf0 { "buf1" } else { "buf0" };
            self.describe_layout("  ffn", layout.feed_forward_slots(), ffn_bytes, next_buffer);

            let ffn_stage = if self.next_is_buf0 {
                &self.buf0[..ffn_bytes]
            } else {
                &self.buf1[..ffn_bytes]
            };
            let mut scratch = LayerScratch {
                hidden: &mut self.hidden,
                hidden_norm: &mut self.hidden_norm,
                q_buf: &mut self.q_buf,
                k_buf: &mut self.k_buf,
                v_buf: &mut self.v_buf,
                attn_out_buf: &mut self.attn_out_buf,
                ffn_gate_buf: &mut self.ffn_gate_buf,
                ffn_up_buf: &mut self.ffn_up_buf,
                ffn_out_buf: &mut self.ffn_out_buf,
                row_buf: &mut self.row_buf,
                scores_buf: &mut self.scores_buf,
                kv_keys: &mut self.kv_keys,
                kv_values: &mut self.kv_values,
            };
            let ffn_stats = compute_ffn_layer(&self.plan.config, layout, ffn_stage, layer_index, &mut scratch)?;
            println!("  compute ffn: rms {:.4} -> {:.4}", ffn_stats.input_rms, ffn_stats.output_rms);

            self.swap_buffers();
            println!("  swap -> next layer uses the freed buffer");
        }

        if let Some(final_norm) = find_final_norm_tensor(&self.file.data) {
            let input_rms = apply_final_norm(
                &self.file,
                final_norm,
                &self.hidden,
                &mut self.hidden_norm,
                &mut self.row_buf,
            )?;
            println!("final norm: rms {:.4} -> {:.4}", input_rms, vector_rms(&self.hidden_norm));
            self.hidden.copy_from_slice(&self.hidden_norm);
        }

        self.trace_head()?;
        Ok(())
    }

    fn load_global_tensors(&mut self) -> Result<()> {
        for slot in &self.plan.global_slots {
            println!("global load: {} ({})", slot.name, format_bytes(slot.size));
            let data = self
                .file
                .tensor_data(&slot.name)
                .ok_or_else(|| anyhow!("missing tensor data for {}", slot.name))?;
            if data.len() != slot.size {
                bail!("tensor {} size mismatch: {} vs {}", slot.name, data.len(), slot.size);
            }
        }
        Ok(())
    }

    fn prefetch_layout(
        &mut self,
        layout: &LayerTensorLayout,
        is_attention: bool,
        into_current: bool,
    ) -> Result<usize> {
        let slots = if is_attention {
            layout.attention_slots()
        } else {
            layout.feed_forward_slots()
        };
        let file = Arc::clone(&self.file);
        let buffer = if into_current {
            self.current_buffer_mut()
        } else {
            self.next_buffer_mut()
        };
        let bytes_written = copy_slots_into_buffer(file.as_ref(), &slots, buffer)?;
        let stage = if is_attention { "attention" } else { "ffn" };
        println!(
            "  prefetch {stage}: {} bytes into {}",
            bytes_written,
            if into_current {
                if self.current_is_buf0 { "buf0" } else { "buf1" }
            } else if self.next_is_buf0 {
                "buf0"
            } else {
                "buf1"
            }
        );
        Ok(bytes_written)
    }

    fn describe_layout(&self, label: &str, slots: Vec<&TensorSlot>, bytes_written: usize, buffer_name: &str) {
        let total: usize = slots.iter().map(|slot| slot.size).sum();
        let names = slots
            .iter()
            .map(|slot| format!("{}@0x{:x}", slot.name, slot.offset))
            .collect::<Vec<_>>()
            .join(", ");
        println!("{label}: {} ({}) [loaded {} into {buffer_name}]", format_bytes(total), names, bytes_written);
    }

    fn trace_head(&mut self) -> Result<()> {
        let head_info = self
            .file
            .data
            .get_tensor(&self.plan.head_name)
            .ok_or_else(|| anyhow!("missing head tensor {}", self.plan.head_name))?;
        let head_stats = compute_head_argmax(&self.file, head_info, &self.hidden, &mut self.row_buf)?;

        println!();
        println!("LM head: {} chunk(s) of {}", self.plan.head_chunk_count, format_bytes(self.plan.head_chunk_bytes));
        println!("  head tensor: {}", self.plan.head_name);
        println!("  top token id: {} (logit {:.4})", head_stats.token_id, head_stats.logit);
        Ok(())
    }
}

fn trace_runtime(file: &Arc<GgufFile>, plan: &ModelPlan, token_id: usize) -> Result<()> {
    let mut runtime = Runtime::new(Arc::clone(file), plan);
    runtime.trace_token(token_id)
}

fn copy_slots_into_buffer(file: &GgufFile, slots: &[&TensorSlot], dst: &mut [u8]) -> Result<usize> {
    let mut cursor = 0usize;
    for slot in slots {
        let data = file
            .tensor_data(&slot.name)
            .ok_or_else(|| anyhow!("missing tensor data for {}", slot.name))?;
        if data.len() != slot.size {
            bail!("tensor {} size mismatch: expected {}, got {}", slot.name, slot.size, data.len());
        }
        let end = cursor + slot.size;
        if end > dst.len() {
            bail!("buffer too small for {}: need {} bytes, have {}", slot.name, end, dst.len());
        }
        dst[cursor..end].copy_from_slice(data);
        cursor = end;
    }
    Ok(cursor)
}

fn stage_tensor_bytes<'a>(stage_bytes: &'a [u8], slots: &[&'a TensorSlot], name: &str) -> Result<(&'a TensorSlot, &'a [u8])> {
    let mut cursor = 0usize;
    for slot in slots {
        let end = cursor + slot.size;
        if end > stage_bytes.len() {
            bail!(
                "stage buffer too small while locating {}: need {} bytes, have {}",
                name,
                end,
                stage_bytes.len()
            );
        }
        if slot.name == name {
            return Ok((slot, &stage_bytes[cursor..end]));
        }
        cursor = end;
    }

    bail!("tensor {} was not found in the provided stage", name);
}

fn decode_tensor_row_from_file(file: &GgufFile, info: &TensorInfo, row_index: usize, out: &mut [f32]) -> Result<()> {
    let data = file
        .tensor_data(&info.name)
        .ok_or_else(|| anyhow!("missing tensor data for {}", info.name))?;
    let (rows, cols) = tensor_shape(info)?;
    if row_index >= rows {
        bail!(
            "row {} is out of bounds for tensor {} with {} rows",
            row_index,
            info.name,
            rows
        );
    }
    if out.len() != cols {
        bail!(
            "output length mismatch for tensor {} row {}: expected {}, got {}",
            info.name,
            row_index,
            cols,
            out.len()
        );
    }

    let row_bytes = tensor_row_bytes(info.dtype, cols)?;
    let start = row_index * row_bytes;
    let end = start + row_bytes;
    decode_tensor_slice_to_f32(info.dtype, &data[start..end], out)
}

fn decode_stage_tensor_to_f32(stage_bytes: &[u8], slot: &TensorSlot, out: &mut [f32]) -> Result<()> {
    let expected = slot.rows * slot.cols;
    if out.len() != expected {
        bail!(
            "decode output mismatch for tensor {}: expected {}, got {}",
            slot.name,
            expected,
            out.len()
        );
    }
    decode_tensor_slice_to_f32(slot.dtype, stage_bytes, out)
}

fn matmul_tensor_slot(
    stage_bytes: &[u8],
    slots: &[&TensorSlot],
    name: &str,
    input: &[f32],
    output: &mut [f32],
    row_buf: &mut [f32],
) -> Result<()> {
    let (slot, bytes) = stage_tensor_bytes(stage_bytes, slots, name)?;
    if slot.rows == 0 || slot.cols == 0 {
        bail!("tensor {} has empty dimensions", slot.name);
    }
    if input.len() != slot.cols {
        bail!(
            "input length mismatch for tensor {}: expected {}, got {}",
            slot.name,
            slot.cols,
            input.len()
        );
    }
    if output.len() != slot.rows {
        bail!(
            "output length mismatch for tensor {}: expected {}, got {}",
            slot.name,
            slot.rows,
            output.len()
        );
    }
    if row_buf.len() < slot.cols {
        bail!(
            "row buffer too small for tensor {}: need {}, have {}",
            slot.name,
            slot.cols,
            row_buf.len()
        );
    }

    let row_bytes = tensor_row_bytes(slot.dtype, slot.cols)?;
    let expected_bytes = row_bytes * slot.rows;
    if bytes.len() != expected_bytes {
        bail!(
            "tensor {} row bytes mismatch: expected {}, got {}",
            slot.name,
            expected_bytes,
            bytes.len()
        );
    }

    for row_index in 0..slot.rows {
        let row_start = row_index * row_bytes;
        let row_end = row_start + row_bytes;
        let row_output = &mut row_buf[..slot.cols];
        decode_tensor_slice_to_f32(slot.dtype, &bytes[row_start..row_end], row_output)?;
        output[row_index] = dot_product(row_output, input);
    }

    Ok(())
}

fn find_final_norm_tensor(data: &GgufData) -> Option<&TensorInfo> {
    data.tensors.iter().find(|info| {
        let name = info.name.as_str();
        name.ends_with("output_norm.weight")
            || name.ends_with("final_norm.weight")
            || (name.ends_with("norm.weight") && !name.contains("attn_norm") && !name.contains("ffn_norm"))
    })
}

struct LayerScratch<'a> {
    hidden: &'a mut [f32],
    hidden_norm: &'a mut [f32],
    q_buf: &'a mut [f32],
    k_buf: &'a mut [f32],
    v_buf: &'a mut [f32],
    attn_out_buf: &'a mut [f32],
    ffn_gate_buf: &'a mut [f32],
    ffn_up_buf: &'a mut [f32],
    ffn_out_buf: &'a mut [f32],
    row_buf: &'a mut [f32],
    scores_buf: &'a mut [f32],
    kv_keys: &'a mut [f32],
    kv_values: &'a mut [f32],
}

struct LayerComputeStats {
    input_rms: f32,
    output_rms: f32,
}

struct HeadComputeStats {
    token_id: usize,
    logit: f32,
}

fn vector_rms(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    mean_square.sqrt()
}

fn load_token_embedding(file: &GgufFile, token_info: &TensorInfo, token_id: usize, hidden: &mut [f32], row_buf: &mut [f32]) -> Result<()> {
    decode_tensor_row_from_file(file, token_info, token_id, hidden)?;
    if row_buf.len() < hidden.len() {
        bail!("token embedding scratch row buffer too small");
    }
    row_buf[..hidden.len()].copy_from_slice(hidden);
    Ok(())
}

fn apply_final_norm(
    file: &GgufFile,
    norm_info: &TensorInfo,
    hidden: &[f32],
    normed: &mut [f32],
    row_buf: &mut [f32],
) -> Result<f32> {
    if row_buf.len() < hidden.len() {
        bail!("final norm scratch row buffer too small");
    }
    decode_tensor_row_from_file(file, norm_info, 0, &mut row_buf[..hidden.len()])?;
    let input_rms = vector_rms(hidden);
    rms_norm(hidden, &row_buf[..hidden.len()], normed)?;
    Ok(input_rms)
}

fn compute_attention_layer(
    config: &ModelConfig,
    layout: &LayerTensorLayout,
    stage_bytes: &[u8],
    layer_index: usize,
    position: usize,
    scratch: &mut LayerScratch<'_>,
) -> Result<LayerComputeStats> {
    let hidden_size = config.hidden_size;
    let kv_dim = config.kv_dim;
    let head_dim = config.head_dim;
    let kv_group_size = config.attention_heads / config.kv_heads;
    if kv_group_size == 0 {
        bail!("invalid kv group size");
    }

    let input_rms = vector_rms(scratch.hidden);
    let attn_slots = layout.attention_slots();
    let attn_norm_slot = attn_slots
        .iter()
        .find(|slot| slot.name.contains("attn_norm"))
        .ok_or_else(|| anyhow!("missing attention norm tensor for layer {}", layer_index))?;
    decode_stage_tensor_to_f32(stage_bytes, attn_norm_slot, &mut scratch.row_buf[..hidden_size])?;
    rms_norm(scratch.hidden, &scratch.row_buf[..hidden_size], scratch.hidden_norm)?;

    let attn_q_slot = attn_slots
        .iter()
        .find(|slot| slot.name.contains("attn_q"))
        .ok_or_else(|| anyhow!("missing attention Q tensor for layer {}", layer_index))?;
    let attn_k_slot = attn_slots
        .iter()
        .find(|slot| slot.name.contains("attn_k"))
        .ok_or_else(|| anyhow!("missing attention K tensor for layer {}", layer_index))?;
    let attn_v_slot = attn_slots
        .iter()
        .find(|slot| slot.name.contains("attn_v"))
        .ok_or_else(|| anyhow!("missing attention V tensor for layer {}", layer_index))?;
    let attn_o_slot = attn_slots
        .iter()
        .find(|slot| slot.name.contains("attn_output"))
        .ok_or_else(|| anyhow!("missing attention output tensor for layer {}", layer_index))?;

    matmul_tensor_slot(stage_bytes, &attn_slots, &attn_q_slot.name, scratch.hidden_norm, scratch.q_buf, scratch.row_buf)?;
    matmul_tensor_slot(stage_bytes, &attn_slots, &attn_k_slot.name, scratch.hidden_norm, scratch.k_buf, scratch.row_buf)?;
    matmul_tensor_slot(stage_bytes, &attn_slots, &attn_v_slot.name, scratch.hidden_norm, scratch.v_buf, scratch.row_buf)?;

    let cache_layer_base = (layer_index * scratch.scores_buf.len().max(1)).saturating_mul(kv_dim);
    if position >= scratch.scores_buf.len() {
        bail!("position {} exceeds configured context tokens", position);
    }
    let cache_row_base = cache_layer_base + position * kv_dim;
    scratch.kv_keys[cache_row_base..cache_row_base + kv_dim].copy_from_slice(&scratch.k_buf[..kv_dim]);
    scratch.kv_values[cache_row_base..cache_row_base + kv_dim].copy_from_slice(&scratch.v_buf[..kv_dim]);

    scratch.attn_out_buf.fill(0.0);
    let seq_len = position + 1;
    let scale = (head_dim as f32).sqrt().recip();
    for head_index in 0..config.attention_heads {
        let kv_head = head_index / kv_group_size;
        let q_offset = head_index * head_dim;
        let kv_offset = kv_head * head_dim;
        let scores = &mut scratch.scores_buf[..seq_len];

        let mut max_score = f32::NEG_INFINITY;
        for token_index in 0..seq_len {
            let cache_offset = cache_layer_base + token_index * kv_dim + kv_offset;
            let score = dot_product(
                &scratch.q_buf[q_offset..q_offset + head_dim],
                &scratch.kv_keys[cache_offset..cache_offset + head_dim],
            ) * scale;
            scores[token_index] = score;
            max_score = max_score.max(score);
        }

        let mut denom = 0.0f32;
        for score in scores.iter_mut() {
            *score = (*score - max_score).exp();
            denom += *score;
        }
        if denom == 0.0 {
            bail!("attention softmax underflow at layer {}", layer_index);
        }
        for score in scores.iter_mut() {
            *score /= denom;
        }

        let out_slice = &mut scratch.attn_out_buf[q_offset..q_offset + head_dim];
        for token_index in 0..seq_len {
            let weight = scores[token_index];
            let value_offset = cache_layer_base + token_index * kv_dim + kv_offset;
            for (output_value, cache_value) in out_slice.iter_mut().zip(
                scratch.kv_values[value_offset..value_offset + head_dim].iter(),
            ) {
                *output_value += weight * cache_value;
            }
        }
    }

    matmul_tensor_slot(
        stage_bytes,
        &attn_slots,
        &attn_o_slot.name,
        scratch.attn_out_buf,
        scratch.hidden_norm,
        scratch.row_buf,
    )?;
    apply_residual(scratch.hidden, scratch.hidden_norm)?;

    Ok(LayerComputeStats {
        input_rms,
        output_rms: vector_rms(scratch.hidden),
    })
}

fn compute_ffn_layer(
    config: &ModelConfig,
    layout: &LayerTensorLayout,
    stage_bytes: &[u8],
    layer_index: usize,
    scratch: &mut LayerScratch<'_>,
) -> Result<LayerComputeStats> {
    let hidden_size = config.hidden_size;
    let ffn_size = config.feed_forward_size;

    let input_rms = vector_rms(scratch.hidden);
    let ffn_slots = layout.feed_forward_slots();
    let ffn_norm_slot = ffn_slots
        .iter()
        .find(|slot| slot.name.contains("ffn_norm"))
        .ok_or_else(|| anyhow!("missing FFN norm tensor for layer {}", layer_index))?;
    decode_stage_tensor_to_f32(stage_bytes, ffn_norm_slot, &mut scratch.row_buf[..hidden_size])?;
    rms_norm(scratch.hidden, &scratch.row_buf[..hidden_size], scratch.hidden_norm)?;

    let ffn_gate_slot = ffn_slots
        .iter()
        .find(|slot| slot.name.contains("ffn_gate"))
        .ok_or_else(|| anyhow!("missing FFN gate tensor for layer {}", layer_index))?;
    let ffn_up_slot = ffn_slots
        .iter()
        .find(|slot| slot.name.contains("ffn_up"))
        .ok_or_else(|| anyhow!("missing FFN up tensor for layer {}", layer_index))?;
    let ffn_down_slot = ffn_slots
        .iter()
        .find(|slot| slot.name.contains("ffn_down"))
        .ok_or_else(|| anyhow!("missing FFN down tensor for layer {}", layer_index))?;

    matmul_tensor_slot(stage_bytes, &ffn_slots, &ffn_gate_slot.name, scratch.hidden_norm, scratch.ffn_gate_buf, scratch.row_buf)?;
    matmul_tensor_slot(stage_bytes, &ffn_slots, &ffn_up_slot.name, scratch.hidden_norm, scratch.ffn_up_buf, scratch.row_buf)?;

    if scratch.ffn_gate_buf.len() < ffn_size || scratch.ffn_up_buf.len() < ffn_size {
        bail!("FFN scratch buffers are too small for layer {}", layer_index);
    }
    silu_in_place(&mut scratch.ffn_gate_buf[..ffn_size]);
    for index in 0..ffn_size {
        scratch.ffn_gate_buf[index] *= scratch.ffn_up_buf[index];
    }

    matmul_tensor_slot(
        stage_bytes,
        &ffn_slots,
        &ffn_down_slot.name,
        &scratch.ffn_gate_buf[..ffn_size],
        &mut scratch.ffn_out_buf[..hidden_size],
        scratch.row_buf,
    )?;
    apply_residual(scratch.hidden, &scratch.ffn_out_buf[..hidden_size])?;

    Ok(LayerComputeStats {
        input_rms,
        output_rms: vector_rms(scratch.hidden),
    })
}

fn compute_head_argmax(
    file: &GgufFile,
    head_info: &TensorInfo,
    input: &[f32],
    row_buf: &mut [f32],
) -> Result<HeadComputeStats> {
    let data = file
        .tensor_data(&head_info.name)
        .ok_or_else(|| anyhow!("missing tensor data for {}", head_info.name))?;
    let (rows, cols) = tensor_shape(head_info)?;
    if input.len() != cols {
        bail!(
            "head input length mismatch: expected {}, got {}",
            cols,
            input.len()
        );
    }
    if row_buf.len() < cols {
        bail!(
            "head row buffer too small: need {}, have {}",
            cols,
            row_buf.len()
        );
    }

    let row_bytes = tensor_row_bytes(head_info.dtype, cols)?;
    let expected_bytes = row_bytes * rows;
    if data.len() != expected_bytes {
        bail!(
            "head tensor byte count mismatch: expected {}, got {}",
            expected_bytes,
            data.len()
        );
    }

    let mut best_token_id = 0usize;
    let mut best_logit = f32::NEG_INFINITY;
    for row_index in 0..rows {
        let row_start = row_index * row_bytes;
        let row_end = row_start + row_bytes;
        decode_tensor_slice_to_f32(head_info.dtype, &data[row_start..row_end], &mut row_buf[..cols])?;
        let logit = dot_product(&row_buf[..cols], input);
        if logit > best_logit {
            best_logit = logit;
            best_token_id = row_index;
        }
    }

    Ok(HeadComputeStats {
        token_id: best_token_id,
        logit: best_logit,
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    if args.context_tokens == 0 {
        bail!("context_tokens must be greater than zero");
    }
    if args.head_chunk_mb == 0 {
        bail!("head_chunk_mb must be greater than zero");
    }
    if args.nvme_gbps <= 0.0 {
        bail!("nvme_gbps must be greater than zero");
    }
    if args.cpu_gflops <= 0.0 {
        bail!("cpu_gflops must be greater than zero");
    }

    let file = Arc::new(
        GgufFile::open(&args.gguf)
            .with_context(|| format!("failed to open GGUF file {}", args.gguf.display()))?,
    );
    let plan = build_plan(&file.data, args.context_tokens, args.head_chunk_mb, args.activation_scratch_mb, args.nvme_gbps, args.cpu_gflops)?;

    if args.trace_runtime {
        trace_runtime(&file, &plan, args.token_id)?;
        return Ok(());
    }

    print_plan(&plan, args.peak_budget_mb, args.show_tensor_names);
    Ok(())
}

fn build_plan(
    data: &GgufData,
    context_tokens: usize,
    head_chunk_mb: usize,
    activation_scratch_mb: usize,
    nvme_gbps: f64,
    cpu_gflops: f64,
) -> Result<StreamPlan> {
    let config = ModelConfig::from_data(data)?;
    let mut layer_summaries = vec![LayerSummary::default(); config.layer_count];
    let mut layer_layouts = vec![LayerTensorLayout::default(); config.layer_count];
    let mut global_tensors = Vec::new();
    let mut global_slots = Vec::new();
    let mut weight_types = BTreeSet::new();

    for tensor in &data.tensors {
        if let Some(layer_index) = parse_layer_index(&tensor.name) {
            if layer_index < layer_summaries.len() {
                let bytes = tensor.data_size();
                let summary = &mut layer_summaries[layer_index];
                let slot = tensor_slot_from_info(tensor)?;

                match classify_tensor(&tensor.name) {
                    TensorClass::Attention => {
                        weight_types.insert(format!("{:?}", tensor.dtype));
                        summary.attention_bytes += bytes;
                        summary.attention_tensors += 1;
                        attach_layer_slot(&mut layer_layouts[layer_index], slot, TensorClass::Attention);
                    }
                    TensorClass::FeedForward => {
                        weight_types.insert(format!("{:?}", tensor.dtype));
                        summary.feed_forward_bytes += bytes;
                        summary.feed_forward_tensors += 1;
                        attach_layer_slot(&mut layer_layouts[layer_index], slot, TensorClass::FeedForward);
                    }
                    TensorClass::Normalization => {
                        summary.normalization_bytes += bytes;
                        summary.normalization_tensors += 1;
                        attach_layer_slot(&mut layer_layouts[layer_index], slot, TensorClass::Normalization);
                    }
                    TensorClass::Other => {
                        summary.other_bytes += bytes;
                        summary.other_tensors += 1;
                    }
                }
            }
        } else if is_global_tensor(&tensor.name) {
            global_tensors.push((tensor.name.clone(), tensor.data_size()));
            global_slots.push(tensor_slot_from_info(tensor)?);
        }
    }

    let weight_type = match weight_types.len() {
        0 => "unknown".to_string(),
        1 => weight_types.into_iter().next().unwrap(),
        _ => format!("mixed ({})", weight_types.into_iter().collect::<Vec<_>>().join(", ")),
    };

    let max_attention_block_bytes = layer_layouts
        .iter()
        .map(LayerTensorLayout::attention_bytes)
        .max()
        .unwrap_or(0);
    let max_feed_forward_block_bytes = layer_layouts
        .iter()
        .map(LayerTensorLayout::feed_forward_bytes)
        .max()
        .unwrap_or(0);
    let max_layer_total_bytes = layer_summaries
        .iter()
        .map(|summary| summary.attention_bytes + summary.feed_forward_bytes + summary.normalization_bytes + summary.other_bytes)
        .max()
        .unwrap_or(0);

    let weight_buffer_bytes = max_attention_block_bytes.max(max_feed_forward_block_bytes);
    let kv_cache_bytes = estimate_kv_cache_bytes(&config, context_tokens);
    let activation_scratch_bytes = activation_scratch_mb * mb();

    let head_info = data
        .get_tensor("output.weight")
        .or_else(|| data.get_tensor("token_embd.weight"))
        .ok_or_else(|| anyhow!("could not find output.weight or token_embd.weight in GGUF"))?;
    let head_bytes = head_info.data_size();
    let head_chunk_bytes = head_chunk_mb * mb();
    let head_chunk_bytes = head_chunk_bytes.min(head_bytes.max(1));
    let head_chunk_count = ceil_div(head_bytes.max(1), head_chunk_bytes);

    let estimated_layer_compute_ms = estimate_layer_compute_ms(&config, context_tokens, cpu_gflops);
    let estimated_layer_io_ms = bytes_to_ms(weight_buffer_bytes, nvme_gbps);
    let estimated_head_latency_ms = max_duration_ms(
        bytes_to_ms(head_bytes, nvme_gbps),
        estimate_head_compute_ms(head_info, cpu_gflops),
    );
    let estimated_token_latency_ms = config.layer_count as f64 * max_duration_ms(estimated_layer_compute_ms, estimated_layer_io_ms)
        + estimated_head_latency_ms;
    let estimated_tokens_per_sec = if estimated_token_latency_ms > 0.0 {
        1000.0 / estimated_token_latency_ms
    } else {
        0.0
    };

    let peak_ram_bytes = 2 * weight_buffer_bytes
        + kv_cache_bytes
        + activation_scratch_bytes
        + head_chunk_bytes;

    Ok(StreamPlan {
        config,
        context_tokens,
        weight_type,
        layer_summaries,
        layer_layouts,
        global_tensors,
        global_slots,
        max_attention_block_bytes,
        max_feed_forward_block_bytes,
        max_layer_total_bytes,
        weight_buffer_bytes,
        kv_cache_bytes,
        activation_scratch_bytes,
        head_name: head_info.name.clone(),
        head_bytes,
        head_chunk_bytes,
        head_chunk_count,
        peak_ram_bytes,
        estimated_layer_compute_ms,
        estimated_layer_io_ms,
        estimated_head_latency_ms,
        estimated_token_latency_ms,
        estimated_tokens_per_sec,
    })
}

impl ModelConfig {
    fn from_data(data: &GgufData) -> Result<Self> {
        let architecture = data.get_string("general.architecture").unwrap_or("llama").to_string();
        let hidden_size = get_arch_u32(data, &architecture, "embedding_length")?;
        let feed_forward_size = get_arch_u32(data, &architecture, "feed_forward_length")?;
        let layer_count = get_arch_u32(data, &architecture, "block_count")?;
        let attention_heads = get_arch_u32(data, &architecture, "attention.head_count")?;
        let kv_heads = get_arch_u32(data, &architecture, "attention.head_count_kv").unwrap_or(attention_heads);

        if attention_heads == 0 || hidden_size == 0 || feed_forward_size == 0 || layer_count == 0 {
            bail!("GGUF metadata contained zero-valued model dimensions");
        }
        if hidden_size % attention_heads != 0 {
            bail!(
                "hidden size {} is not divisible by attention heads {}",
                hidden_size,
                attention_heads
            );
        }
        if attention_heads % kv_heads != 0 {
            bail!(
                "attention heads {} is not divisible by kv heads {}",
                attention_heads,
                kv_heads
            );
        }

        let head_dim = hidden_size / attention_heads;

        Ok(Self {
            architecture,
            hidden_size,
            feed_forward_size,
            layer_count,
            attention_heads,
            kv_heads,
            head_dim,
            kv_dim: kv_heads * head_dim,
        })
    }
}

fn get_arch_u32(data: &GgufData, arch: &str, key: &str) -> Result<usize> {
    let full_key = format!("{}.{}", arch, key);
    data.get_u32(&full_key)
        .map(|value| value as usize)
        .ok_or_else(|| anyhow!("missing GGUF metadata key {full_key}"))
}

fn parse_layer_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (layer_text, _) = rest.split_once('.')?;
    layer_text.parse().ok()
}

fn classify_tensor(name: &str) -> TensorClass {
    if name.contains("attn_norm") || name.contains("ffn_norm") {
        TensorClass::Normalization
    } else if name.contains("attn_") {
        TensorClass::Attention
    } else if name.contains(".ffn_") {
        TensorClass::FeedForward
    } else {
        TensorClass::Other
    }
}

fn attach_layer_slot(layout: &mut LayerTensorLayout, slot: TensorSlot, class: TensorClass) {
    match class {
        TensorClass::Attention => {
            if slot.name.contains("attn_q") {
                layout.attn_q = Some(slot);
            } else if slot.name.contains("attn_k") {
                layout.attn_k = Some(slot);
            } else if slot.name.contains("attn_v") {
                layout.attn_v = Some(slot);
            } else if slot.name.contains("attn_output") {
                layout.attn_o = Some(slot);
            } else {
                layout.attn_q.get_or_insert(slot);
            }
        }
        TensorClass::FeedForward => {
            if slot.name.contains("ffn_gate") {
                layout.ffn_gate = Some(slot);
            } else if slot.name.contains("ffn_up") {
                layout.ffn_up = Some(slot);
            } else if slot.name.contains("ffn_down") {
                layout.ffn_down = Some(slot);
            } else {
                layout.ffn_gate.get_or_insert(slot);
            }
        }
        TensorClass::Normalization => {
            if slot.name.contains("attn_norm") {
                layout.attn_norm = Some(slot);
            } else if slot.name.contains("ffn_norm") {
                layout.ffn_norm = Some(slot);
            }
        }
        TensorClass::Other => {}
    }
}

fn is_global_tensor(name: &str) -> bool {
    matches!(name, "token_embd.weight" | "output.weight")
}

fn estimate_kv_cache_bytes(config: &ModelConfig, context_tokens: usize) -> usize {
    let bytes_per_value = 1usize;
    config.layer_count
        .saturating_mul(2)
    .saturating_mul(config.kv_dim)
        .saturating_mul(context_tokens)
        .saturating_mul(bytes_per_value)
}

fn estimate_layer_compute_ms(config: &ModelConfig, context_tokens: usize, cpu_gflops: f64) -> f64 {
    let hidden = config.hidden_size as f64;
    let ffn = config.feed_forward_size as f64;
    let heads = config.attention_heads as f64;
    let head_dim = config.head_dim as f64;

    let attention_proj_flops = 8.0 * hidden * hidden;
    let attention_context_flops = 4.0 * heads * context_tokens as f64 * head_dim;
    let feed_forward_flops = 6.0 * hidden * ffn;
    let total_flops = attention_proj_flops + attention_context_flops + feed_forward_flops;

    total_flops / (cpu_gflops * 1.0e9) * 1.0e3
}

fn estimate_head_compute_ms(info: &TensorInfo, cpu_gflops: f64) -> f64 {
    let total_flops = 2.0 * info.n_elements() as f64;
    total_flops / (cpu_gflops * 1.0e9) * 1.0e3
}

fn bytes_to_ms(bytes: usize, bandwidth_gbps: f64) -> f64 {
    bytes as f64 / (bandwidth_gbps * 1.0e9) * 1.0e3
}

fn max_duration_ms(a: f64, b: f64) -> f64 {
    a.max(b)
}

fn mb() -> usize {
    1024 * 1024
}

fn ceil_div(numer: usize, denom: usize) -> usize {
    (numer + denom - 1) / denom
}

fn format_bytes(bytes: usize) -> String {
    let mb_value = bytes as f64 / mb() as f64;
    if mb_value >= 100.0 {
        format!("{mb_value:.1} MB")
    } else {
        format!("{mb_value:.2} MB")
    }
}

fn tensor_shape(info: &TensorInfo) -> Result<(usize, usize)> {
    match info.dims.as_slice() {
        [cols] => Ok((1, *cols as usize)),
        [cols, rows] => Ok((*rows as usize, *cols as usize)),
        [] => bail!("tensor {} has no dimensions", info.name),
        dims => bail!("tensor {} has unsupported rank {}", info.name, dims.len()),
    }
}

fn tensor_slot_from_info(info: &TensorInfo) -> Result<TensorSlot> {
    let (rows, cols) = tensor_shape(info)?;
    Ok(TensorSlot {
        name: info.name.clone(),
        offset: info.offset,
        size: info.data_size(),
        dtype: info.dtype,
        rows,
        cols,
    })
}

fn tensor_row_bytes(dtype: GgmlType, cols: usize) -> Result<usize> {
    let block_size = dtype.block_size();
    if cols % block_size != 0 {
        bail!(
            "tensor column count {} is not divisible by block size {} for {:?}",
            cols,
            block_size,
            dtype
        );
    }
    Ok(cols / block_size * dtype.type_size())
}

fn decode_quantized_row<T, const BLOCK_SIZE: usize, F>(
    bytes: &[u8],
    out: &mut [f32],
    mut dequantize: F,
) -> Result<()>
where
    T: Pod,
    F: FnMut(&T, &mut [f32; BLOCK_SIZE]),
{
    if out.len() % BLOCK_SIZE != 0 {
        bail!(
            "output length {} is not divisible by block size {}",
            out.len(),
            BLOCK_SIZE
        );
    }

    let block_count = out.len() / BLOCK_SIZE;
    let expected_bytes = block_count * size_of::<T>();
    if bytes.len() != expected_bytes {
        bail!(
            "quantized row length mismatch: expected {} bytes, got {}",
            expected_bytes,
            bytes.len()
        );
    }

    let mut block_out = [0.0f32; BLOCK_SIZE];
    for block_index in 0..block_count {
        let start = block_index * size_of::<T>();
        let end = start + size_of::<T>();
        let block: T = pod_read_unaligned(&bytes[start..end]);
        dequantize(&block, &mut block_out);
        let out_start = block_index * BLOCK_SIZE;
        out[out_start..out_start + BLOCK_SIZE].copy_from_slice(&block_out);
    }

    Ok(())
}

fn decode_tensor_slice_to_f32(dtype: GgmlType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlType::F32 => {
            let expected = out.len() * size_of::<f32>();
            if bytes.len() != expected {
                bail!("F32 tensor length mismatch: expected {} bytes, got {}", expected, bytes.len());
            }
            for (index, chunk) in bytes.chunks_exact(size_of::<f32>()).enumerate() {
                out[index] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            Ok(())
        }
        GgmlType::F16 => {
            let expected = out.len() * size_of::<u16>();
            if bytes.len() != expected {
                bail!("F16 tensor length mismatch: expected {} bytes, got {}", expected, bytes.len());
            }
            for (index, chunk) in bytes.chunks_exact(size_of::<u16>()).enumerate() {
                out[index] = f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32();
            }
            Ok(())
        }
        GgmlType::BF16 => {
            let expected = out.len() * size_of::<u16>();
            if bytes.len() != expected {
                bail!("BF16 tensor length mismatch: expected {} bytes, got {}", expected, bytes.len());
            }
            for (index, chunk) in bytes.chunks_exact(size_of::<u16>()).enumerate() {
                out[index] = bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32();
            }
            Ok(())
        }
        GgmlType::I8 => {
            if bytes.len() != out.len() {
                bail!("I8 tensor length mismatch: expected {} bytes, got {}", out.len(), bytes.len());
            }
            for (value, out_value) in bytes.iter().zip(out.iter_mut()) {
                *out_value = *value as i8 as f32;
            }
            Ok(())
        }
        GgmlType::I16 => {
            let expected = out.len() * size_of::<i16>();
            if bytes.len() != expected {
                bail!("I16 tensor length mismatch: expected {} bytes, got {}", expected, bytes.len());
            }
            for (index, chunk) in bytes.chunks_exact(size_of::<i16>()).enumerate() {
                out[index] = i16::from_le_bytes(chunk.try_into().unwrap()) as f32;
            }
            Ok(())
        }
        GgmlType::I32 => {
            let expected = out.len() * size_of::<i32>();
            if bytes.len() != expected {
                bail!("I32 tensor length mismatch: expected {} bytes, got {}", expected, bytes.len());
            }
            for (index, chunk) in bytes.chunks_exact(size_of::<i32>()).enumerate() {
                out[index] = i32::from_le_bytes(chunk.try_into().unwrap()) as f32;
            }
            Ok(())
        }
        GgmlType::I64 => {
            let expected = out.len() * size_of::<i64>();
            if bytes.len() != expected {
                bail!("I64 tensor length mismatch: expected {} bytes, got {}", expected, bytes.len());
            }
            for (index, chunk) in bytes.chunks_exact(size_of::<i64>()).enumerate() {
                out[index] = i64::from_le_bytes(chunk.try_into().unwrap()) as f32;
            }
            Ok(())
        }
        GgmlType::F64 => {
            let expected = out.len() * size_of::<f64>();
            if bytes.len() != expected {
                bail!("F64 tensor length mismatch: expected {} bytes, got {}", expected, bytes.len());
            }
            for (index, chunk) in bytes.chunks_exact(size_of::<f64>()).enumerate() {
                out[index] = f64::from_le_bytes(chunk.try_into().unwrap()) as f32;
            }
            Ok(())
        }
        GgmlType::Q4_0 => decode_quantized_row::<BlockQ4_0, 32, _>(bytes, out, dequantize_q4_0),
        GgmlType::Q4_1 => decode_quantized_row::<BlockQ4_1, 32, _>(bytes, out, dequantize_q4_1),
        GgmlType::Q5_0 => decode_quantized_row::<BlockQ5_0, 32, _>(bytes, out, dequantize_q5_0),
        GgmlType::Q5_1 => decode_quantized_row::<BlockQ5_1, 32, _>(bytes, out, dequantize_q5_1),
        GgmlType::Q8_0 => decode_quantized_row::<BlockQ8_0, 32, _>(bytes, out, dequantize_q8_0),
        GgmlType::Q8_1 => decode_quantized_row::<BlockQ8_1, 32, _>(bytes, out, dequantize_q8_1),
        GgmlType::Q2K => decode_quantized_row::<BlockQ2K, 256, _>(bytes, out, dequantize_q2_k),
        GgmlType::Q3K => decode_quantized_row::<BlockQ3K, 256, _>(bytes, out, dequantize_q3_k),
        GgmlType::Q4K => decode_quantized_row::<BlockQ4K, 256, _>(bytes, out, dequantize_q4_k),
        GgmlType::Q5K => decode_quantized_row::<BlockQ5K, 256, _>(bytes, out, dequantize_q5_k),
        GgmlType::Q6K => decode_quantized_row::<BlockQ6K, 256, _>(bytes, out, dequantize_q6_k),
        GgmlType::Q8K => decode_quantized_row::<BlockQ8K, 256, _>(bytes, out, dequantize_q8_k),
        other => bail!("unsupported tensor dtype {:?} for decode", other),
    }
}

fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(left, right)| left * right).sum()
}

fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32]) -> Result<f32> {
    if input.len() != weight.len() || output.len() != input.len() {
        bail!(
            "rms_norm length mismatch: input {}, weight {}, output {}",
            input.len(),
            weight.len(),
            output.len()
        );
    }

    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = 1.0 / (mean_square + 1e-5).sqrt();
    for ((input_value, weight_value), output_value) in input.iter().zip(weight.iter()).zip(output.iter_mut()) {
        *output_value = input_value * scale * weight_value;
    }
    Ok(scale)
}

fn silu_in_place(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value *= 1.0 / (1.0 + (-*value).exp());
    }
}

fn apply_residual(dst: &mut [f32], src: &[f32]) -> Result<()> {
    if dst.len() != src.len() {
        bail!("residual length mismatch: {} vs {}", dst.len(), src.len());
    }
    for (dst_value, src_value) in dst.iter_mut().zip(src.iter()) {
        *dst_value += src_value;
    }
    Ok(())
}

fn describe_slots(slots: Vec<&TensorSlot>) -> String {
    if slots.is_empty() {
        return "(none)".to_string();
    }

    slots
        .iter()
        .map(|slot| format!("{}@0x{:x} ({})", slot.name, slot.offset, format_bytes(slot.size)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_plan(plan: &StreamPlan, peak_budget_mb: usize, show_tensor_names: bool) {
    println!("StreamLayer plan");
    println!("architecture: {}", plan.config.architecture);
    println!("weight type: {}", plan.weight_type);
    println!("layers: {}", plan.config.layer_count);
    println!("hidden size: {}", plan.config.hidden_size);
    println!("feed-forward size: {}", plan.config.feed_forward_size);
    println!("attention heads: {}", plan.config.attention_heads);
    println!("kv heads: {}", plan.config.kv_heads);
    println!("head dim: {}", plan.config.head_dim);
    println!();

    println!("Global tensors (mmap'd, not counted in peak working buffers):");
    if plan.global_tensors.is_empty() {
        println!("  (none detected)");
    } else {
        for (name, bytes) in &plan.global_tensors {
            println!("  {name:<24} {}", format_bytes(*bytes));
        }
    }
    println!();

    println!("Layer blocks:");
    for (index, layer) in plan.layer_summaries.iter().enumerate() {
        let layout = &plan.layer_layouts[index];
        let layer_total = layout.total_bytes() + layer.other_bytes;
        println!(
            "  blk.{index:<2} attn {:>10} ({:>2} tensors) | ffn {:>10} ({:>2} tensors) | norm {:>10} ({:>2}) | other {:>10} ({:>2}) | total {:>10}",
            format_bytes(layout.attention_bytes()),
            layer.attention_tensors,
            format_bytes(layout.feed_forward_bytes()),
            layer.feed_forward_tensors,
            format_bytes(layer.normalization_bytes),
            layer.normalization_tensors,
            format_bytes(layer.other_bytes),
            layer.other_tensors,
            format_bytes(layer_total),
        );

        if show_tensor_names {
            println!("    attention: {}", describe_slots(layout.attention_slots()));
            println!("    ffn: {}", describe_slots(layout.feed_forward_slots()));
        }
    }
    println!();

    println!("Peak-memory model:");
    println!("  max attention block: {}", format_bytes(plan.max_attention_block_bytes));
    println!("  max FFN block:       {}", format_bytes(plan.max_feed_forward_block_bytes));
    println!("  max whole layer:     {}", format_bytes(plan.max_layer_total_bytes));
    println!("  double-buffered weights: {}", format_bytes(2 * plan.weight_buffer_bytes));
    println!("  KV cache ({} tok):   {}", plan.context_tokens, format_bytes(plan.kv_cache_bytes));
    println!("  activations/scratch:  {}", format_bytes(plan.activation_scratch_bytes));
    println!("  head chunk buffer:    {}", format_bytes(plan.head_chunk_bytes));
    println!("  peak RAM estimate:    {}", format_bytes(plan.peak_ram_bytes));
    println!("  budget limit:         {}", format_bytes(peak_budget_mb * mb()));
    println!(
        "  verdict: {}",
        if plan.peak_ram_bytes <= peak_budget_mb * mb() {
            "within budget"
        } else {
            "over budget"
        }
    );
    println!();

    println!("I/O and compute model:");
    println!("  estimated layer compute: {:.2} ms", plan.estimated_layer_compute_ms);
    println!("  estimated layer I/O:     {:.2} ms", plan.estimated_layer_io_ms);
    println!("  estimated head latency:  {:.2} ms", plan.estimated_head_latency_ms);
    println!("  estimated token latency: {:.2} ms", plan.estimated_token_latency_ms);
    println!("  estimated throughput:    {:.2} tok/s", plan.estimated_tokens_per_sec);
    println!("  head tensor: {} ({})", plan.head_name, format_bytes(plan.head_bytes));
    println!("  head chunks: {} x {}", plan.head_chunk_count, format_bytes(plan.head_chunk_bytes));
    println!();

    println!("Scheduler outline:");
    println!("  1. mmap the GGUF file and index tensor offsets by layer.");
    println!("  2. Allocate two buffers of size {}.", format_bytes(plan.weight_buffer_bytes));
    println!("  3. Prefetch layer 0 attention into buffer A.");
    println!("  4. For each layer, overlap next-layer attention prefetch with current-layer compute.");
    println!("  5. Load the FFN block after attention, reusing the same buffer pair.");
    println!("  6. Stream the LM head in {} chunk(s) of at most {}.", plan.head_chunk_count, format_bytes(plan.head_chunk_bytes));
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_layer_index_recognizes_block_names() {
        assert_eq!(parse_layer_index("blk.14.attn_q.weight"), Some(14));
        assert_eq!(parse_layer_index("token_embd.weight"), None);
    }

    #[test]
    fn classify_tensor_separates_attention_and_ffn() {
        assert_eq!(classify_tensor("blk.0.attn_output.weight"), TensorClass::Attention);
        assert_eq!(classify_tensor("blk.0.ffn_down.weight"), TensorClass::FeedForward);
        assert_eq!(classify_tensor("token_embd.weight"), TensorClass::Other);
    }

    #[test]
    fn kv_cache_estimate_matches_expected_shape() {
        let config = ModelConfig {
            architecture: "llama".to_string(),
            hidden_size: 3072,
            feed_forward_size: 8192,
            layer_count: 28,
            attention_heads: 32,
            kv_heads: 8,
            head_dim: 96,
            kv_dim: 768,
        };

        assert_eq!(estimate_kv_cache_bytes(&config, 256), 11_010_048);
    }
}