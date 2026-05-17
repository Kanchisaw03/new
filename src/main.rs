use std::convert::TryInto;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use half::{bf16, f16};
use nalgebra::{linalg::SVD, DMatrix};
use ndarray::ArrayView2;
use llama_gguf::gguf::GgufFile;
use llama_gguf::tensor::quant::{
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    dequantize_q8_1, dequantize_q8_k, BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K,
    BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K,
};
use llama_gguf::tensor::DType;

macro_rules! decode_quantized_blocks {
    ($raw:expr, $elements:expr, $dtype:expr, $block_ty:ty, $block_size:expr, $block_bytes:expr, $dequant_fn:ident) => {{
        if $elements % $block_size != 0 {
            return Err(anyhow::anyhow!(
                "quantized tensor length {} is not divisible by block size {}",
                $elements,
                $block_size
            ));
        }
        if $dtype.block_size() != $block_size {
            return Err(anyhow::anyhow!(
                "dtype block size mismatch: {:?} reports {}, expected {}",
                $dtype,
                $dtype.block_size(),
                $block_size
            ));
        }
        if $dtype.block_bytes() != $block_bytes {
            return Err(anyhow::anyhow!(
                "dtype block byte size mismatch: {:?} reports {}, expected {}",
                $dtype,
                $dtype.block_bytes(),
                $block_bytes
            ));
        }
        let expected_bytes = ($elements / $block_size) * $block_bytes;
        if $raw.len() != expected_bytes {
            return Err(anyhow::anyhow!(
                "quantized tensor byte length mismatch: got {}, expected {}",
                $raw.len(),
                expected_bytes
            ));
        }

        let mut out = Vec::with_capacity($elements);
        for chunk in $raw.chunks_exact($block_bytes) {
            let block: $block_ty = unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const $block_ty) };
            let mut decoded = [0.0f32; $block_size];
            $dequant_fn(&block, &mut decoded);
            out.extend_from_slice(&decoded);
        }
        Ok(out)
    }};
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Single-layer MPO compute-in-compression test for a GGUF Q projection"
)]
struct Args {
    #[arg(long)]
    gguf: PathBuf,

    #[arg(long, default_value = "blk.0.attn_q.weight")]
    tensor: String,

    #[arg(long, value_delimiter = ',')]
    chi: Vec<usize>,

    #[arg(long, default_value_t = 128)]
    samples: usize,

    #[arg(long, default_value_t = 1)]
    seed: u64,

    #[arg(long)]
    input_binary: Option<PathBuf>,

    #[arg(long, default_value_t = 4096)]
    frob_samples: usize,
}

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
    spare: Option<f32>,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / ((1u32 << 24) as f32)
    }

    fn standard_normal(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }

        let u1 = self.next_unit_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_unit_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.spare = Some(z1);
        z0
    }

    fn sample_standard_normal_vec(&mut self, len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.standard_normal());
        }
        out
    }
}

#[derive(Debug, Clone)]
struct TensorMatrix {
    name: String,
    rows: usize,
    cols: usize,
    dtype: DType,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct MpoCore {
    chi_right: usize,
    data: Vec<f32>,
}

impl MpoCore {
    fn new(chi_left: usize, m: usize, n: usize, chi_right: usize, data: Vec<f32>) -> Result<Self> {
        let expected = chi_left
            .checked_mul(m)
            .and_then(|value| value.checked_mul(n))
            .and_then(|value| value.checked_mul(chi_right))
            .ok_or_else(|| anyhow!("core size overflow"))?;
        if data.len() != expected {
            bail!(
                "core data length mismatch: got {}, expected {}",
                data.len(),
                expected
            );
        }
        Ok(Self {
            chi_right,
            data,
        })
    }

    fn parameter_count(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Clone, Copy)]
struct TtStats {
    original_frobenius_norm: f32,
    first_tail_energy: f32,
    second_tail_energy: f32,
    approx_relative_frobenius_error_bound: f32,
}

#[derive(Debug, Clone)]
struct MpoDecomposition {
    rows: usize,
    cols: usize,
    shape_m: [usize; 3],
    shape_n: [usize; 3],
    chi_max: usize,
    cores: [MpoCore; 3],
    stats: TtStats,
}

impl MpoDecomposition {
    fn parameter_count(&self) -> usize {
        self.cores.iter().map(MpoCore::parameter_count).sum()
    }

    fn compression_ratio(&self) -> f64 {
        (self.rows * self.cols) as f64 / self.parameter_count() as f64
    }

    fn apply(&self, vector: &[f32]) -> Result<Vec<f32>> {
        let [m1, m2, m3] = self.shape_m;
        let [n1, n2, n3] = self.shape_n;
        let r1 = self.cores[0].chi_right;
        let r2 = self.cores[1].chi_right;

        if vector.len() != n1 * n2 * n3 {
            bail!(
                "input vector length mismatch: got {}, expected {}",
                vector.len(),
                n1 * n2 * n3
            );
        }

        let core1 = DMatrix::from_row_slice(m1, n1 * r1, &self.cores[0].data);
        let core2 = DMatrix::from_row_slice(r1 * m2, n2 * r2, &self.cores[1].data);
        let core3 = DMatrix::from_row_slice(r2 * m3, n3, &self.cores[2].data);
        let input = DMatrix::from_row_slice(n1 * n2, n3, vector);

        // First contract the input over the n3 axis against the rightmost core.
        let step1 = input * core3.transpose(); // (n1*n2, r2*m3)

        // Then contract the middle core for each (i3, j1) pair.
        let mut middle_cache = vec![0.0f32; n1 * r1 * m2 * m3];
        for i3 in 0..m3 {
            for j1 in 0..n1 {
                let mut w = vec![0.0f32; n2 * r2];
                for j2 in 0..n2 {
                    let row = j1 * n2 + j2;
                    for bond2 in 0..r2 {
                        w[j2 * r2 + bond2] = step1[(row, bond2 * m3 + i3)];
                    }
                }

                let w_matrix = DMatrix::from_column_slice(n2 * r2, 1, &w);
                let z = &core2 * w_matrix; // (r1*m2, 1)
                for bond1 in 0..r1 {
                    for i2 in 0..m2 {
                        let cache_index = (((i3 * n1 + j1) * r1 + bond1) * m2) + i2;
                        middle_cache[cache_index] = z[(bond1 * m2 + i2, 0)];
                    }
                }
            }
        }

        // Final contraction with the leftmost core.
        let mut output = vec![0.0f32; m1 * m2 * m3];
        for i3 in 0..m3 {
            for i2 in 0..m2 {
                let mut v = vec![0.0f32; n1 * r1];
                for j1 in 0..n1 {
                    for bond1 in 0..r1 {
                        let cache_index = (((i3 * n1 + j1) * r1 + bond1) * m2) + i2;
                        v[j1 * r1 + bond1] = middle_cache[cache_index];
                    }
                }

                let v_matrix = DMatrix::from_column_slice(n1 * r1, 1, &v);
                let y = &core1 * v_matrix; // (m1, 1)
                for i1 in 0..m1 {
                    let row = ((i1 * m2) + i2) * m3 + i3;
                    output[row] = y[(i1, 0)];
                }
            }
        }

        Ok(output)
    }

    fn entry(&self, row: usize, col: usize) -> f32 {
        let [_m1, m2, m3] = self.shape_m;
        let [n1, n2, n3] = self.shape_n;
        let r1 = self.cores[0].chi_right;
        let r2 = self.cores[1].chi_right;

        let row_i1 = row / (m2 * m3);
        let row_rem = row % (m2 * m3);
        let row_i2 = row_rem / m3;
        let row_i3 = row_rem % m3;

        let col_j1 = col / (n2 * n3);
        let col_rem = col % (n2 * n3);
        let col_j2 = col_rem / n3;
        let col_j3 = col_rem % n3;

        let mut sum = 0.0f32;
        for bond1 in 0..r1 {
            let core1_index = ((row_i1 * n1 + col_j1) * r1) + bond1;
            let core1_value = self.cores[0].data[core1_index];
            for bond2 in 0..r2 {
                let core2_index = (((bond1 * m2 + row_i2) * n2 + col_j2) * r2) + bond2;
                let core2_value = self.cores[1].data[core2_index];
                let core3_index = ((bond2 * m3 + row_i3) * n3) + col_j3;
                let core3_value = self.cores[2].data[core3_index];
                sum += core1_value * core2_value * core3_value;
            }
        }

        sum
    }

    fn reconstruct_dense(&self) -> Result<Vec<f32>> {
        let mut dense = vec![0.0f32; self.rows * self.cols];
        for row in 0..self.rows {
            for col in 0..self.cols {
                dense[row * self.cols + col] = self.entry(row, col);
            }
        }

        Ok(dense)
    }
}

fn load_tensor_matrix(path: &Path, tensor_name: &str) -> Result<TensorMatrix> {
    let file = GgufFile::open(path)
        .with_context(|| format!("failed to open GGUF file {}", path.display()))?;
    let tensor_info = file
        .data
        .get_tensor(tensor_name)
        .ok_or_else(|| anyhow!("tensor {tensor_name:?} not found in GGUF metadata"))?;

    if tensor_info.dims.len() != 2 {
        bail!(
            "tensor {tensor_name:?} has {} dimensions; expected a matrix",
            tensor_info.dims.len()
        );
    }

    let rows: usize = tensor_info.dims[0]
        .try_into()
        .context("row dimension does not fit into usize")?;
    let cols: usize = tensor_info.dims[1]
        .try_into()
        .context("column dimension does not fit into usize")?;

    let dtype: DType = tensor_info.dtype.into();
    let raw = file
        .tensor_data(tensor_name)
        .ok_or_else(|| anyhow!("tensor data for {tensor_name:?} not available"))?;
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("tensor element count overflow"))?;
    let values = decode_tensor_bytes(raw, dtype, elements)?;

    Ok(TensorMatrix {
        name: tensor_name.to_string(),
        rows,
        cols,
        dtype,
        values,
    })
}

fn decode_tensor_bytes(raw: &[u8], dtype: DType, elements: usize) -> Result<Vec<f32>> {
    match dtype {
        DType::F32 => {
            if raw.len() != elements * 4 {
                bail!(
                    "F32 tensor byte length mismatch: got {}, expected {}",
                    raw.len(),
                    elements * 4
                );
            }
            let mut out = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(out)
        }
        DType::F16 => {
            if raw.len() != elements * 2 {
                bail!(
                    "F16 tensor byte length mismatch: got {}, expected {}",
                    raw.len(),
                    elements * 2
                );
            }
            let mut out = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(2) {
                out.push(f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32());
            }
            Ok(out)
        }
        DType::BF16 => {
            if raw.len() != elements * 2 {
                bail!(
                    "BF16 tensor byte length mismatch: got {}, expected {}",
                    raw.len(),
                    elements * 2
                );
            }
            let mut out = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(2) {
                out.push(bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32());
            }
            Ok(out)
        }
        DType::F64 => {
            if raw.len() != elements * 8 {
                bail!(
                    "F64 tensor byte length mismatch: got {}, expected {}",
                    raw.len(),
                    elements * 8
                );
            }
            let mut out = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(8) {
                out.push(f64::from_le_bytes(chunk.try_into().unwrap()) as f32);
            }
            Ok(out)
        }
        DType::I8 => {
            if raw.len() != elements {
                bail!(
                    "I8 tensor byte length mismatch: got {}, expected {}",
                    raw.len(),
                    elements
                );
            }
            Ok(raw.iter().map(|&byte| byte as i8 as f32).collect())
        }
        DType::U8 => {
            if raw.len() != elements {
                bail!(
                    "U8 tensor byte length mismatch: got {}, expected {}",
                    raw.len(),
                    elements
                );
            }
            Ok(raw.iter().map(|&byte| byte as f32).collect())
        }
        DType::I16 => {
            if raw.len() != elements * 2 {
                bail!(
                    "I16 tensor byte length mismatch: got {}, expected {}",
                    raw.len(),
                    elements * 2
                );
            }
            Ok(raw
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()) as f32)
                .collect())
        }
        DType::I32 => {
            if raw.len() != elements * 4 {
                bail!(
                    "I32 tensor byte length mismatch: got {}, expected {}",
                    raw.len(),
                    elements * 4
                );
            }
            Ok(raw
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as f32)
                .collect())
        }
        DType::I64 => {
            if raw.len() != elements * 8 {
                bail!(
                    "I64 tensor byte length mismatch: got {}, expected {}",
                    raw.len(),
                    elements * 8
                );
            }
            Ok(raw
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()) as f32)
                .collect())
        }
        DType::Q2K => decode_quantized_blocks!(raw, elements, dtype, BlockQ2K, 256, 84, dequantize_q2_k),
        DType::Q3K => decode_quantized_blocks!(raw, elements, dtype, BlockQ3K, 256, 110, dequantize_q3_k),
        DType::Q4_0 => decode_quantized_blocks!(raw, elements, dtype, BlockQ4_0, 32, 18, dequantize_q4_0),
        DType::Q4_1 => decode_quantized_blocks!(raw, elements, dtype, BlockQ4_1, 32, 20, dequantize_q4_1),
        DType::Q4K => decode_quantized_blocks!(raw, elements, dtype, BlockQ4K, 256, 144, dequantize_q4_k),
        DType::Q5_0 => decode_quantized_blocks!(raw, elements, dtype, BlockQ5_0, 32, 22, dequantize_q5_0),
        DType::Q5_1 => decode_quantized_blocks!(raw, elements, dtype, BlockQ5_1, 32, 24, dequantize_q5_1),
        DType::Q5K => decode_quantized_blocks!(raw, elements, dtype, BlockQ5K, 256, 176, dequantize_q5_k),
        DType::Q6K => decode_quantized_blocks!(raw, elements, dtype, BlockQ6K, 256, 210, dequantize_q6_k),
        DType::Q8_0 => decode_quantized_blocks!(raw, elements, dtype, BlockQ8_0, 32, 34, dequantize_q8_0),
        DType::Q8_1 => decode_quantized_blocks!(raw, elements, dtype, BlockQ8_1, 32, 36, dequantize_q8_1),
        DType::Q8K => decode_quantized_blocks!(raw, elements, dtype, BlockQ8K, 256, 292, dequantize_q8_k),
        _ => bail!("unsupported tensor dtype {dtype:?}"),
    }
}

fn balanced_three_factors(n: usize) -> [usize; 3] {
    if n == 0 {
        return [0, 0, 0];
    }

    let mut best = [1, 1, n];
    let mut best_score = f64::INFINITY;

    for a in divisors(n) {
        let rest = n / a;
        for b in divisors(rest) {
            let c = rest / b;
            let mut triplet = [a, b, c];
            triplet.sort_unstable();
            let logs = [
                (triplet[0] as f64).ln(),
                (triplet[1] as f64).ln(),
                (triplet[2] as f64).ln(),
            ];
            let mean = (logs[0] + logs[1] + logs[2]) / 3.0;
            let score = (logs[0] - mean).powi(2) + (logs[1] - mean).powi(2) + (logs[2] - mean).powi(2);
            if score < best_score {
                best_score = score;
                best = triplet;
            }
        }
    }

    best.sort_unstable();
    best
}

fn factor_tensor_shapes(rows: usize, cols: usize) -> ([usize; 3], [usize; 3]) {
    (balanced_three_factors(rows), balanced_three_factors(cols))
}

fn divisors(n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = 1usize;
    while i * i <= n {
        if n % i == 0 {
            out.push(i);
            let other = n / i;
            if other != i {
                out.push(other);
            }
        }
        i += 1;
    }
    out.sort_unstable();
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
}

fn estimate_relative_frobenius_error(
    reference: &[f32],
    mpo: &MpoDecomposition,
    samples: usize,
    seed: u64,
) -> f32 {
    let sample_count = samples.max(1);
    let mut rng = SimpleRng::new(seed);
    let mut error_sum = 0.0f64;
    let mut reference_sum = 0.0f64;

    for _ in 0..sample_count {
        let row = (rng.next_u64() as usize) % mpo.rows;
        let col = (rng.next_u64() as usize) % mpo.cols;
        let index = row * mpo.cols + col;
        let reference_value = reference[index] as f64;
        let approx_value = mpo.entry(row, col) as f64;
        let diff = reference_value - approx_value;
        error_sum += diff * diff;
        reference_sum += reference_value * reference_value;
    }

    if reference_sum == 0.0 {
        0.0
    } else {
        (error_sum / reference_sum).sqrt() as f32
    }
}

fn dense_matvec(matrix: &[f32], rows: usize, cols: usize, vector: &[f32]) -> Result<Vec<f32>> {
    if vector.len() != cols {
        bail!(
            "dense matvec input length mismatch: got {}, expected {}",
            vector.len(),
            cols
        );
    }
    let matrix = ArrayView2::from_shape((rows, cols), matrix)
        .context("failed to view dense matrix")?;
    let vector = ArrayView2::from_shape((cols, 1), vector)
        .context("failed to view dense input vector")?;
    let output = matrix.dot(&vector);
    Ok(output.column(0).iter().copied().collect())
}

fn precompute_dense_outputs(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    inputs: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        outputs.push(dense_matvec(matrix, rows, cols, input)?);
    }
    Ok(outputs)
}

fn matrix_to_row_major_vec(matrix: &DMatrix<f32>) -> Vec<f32> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[row * cols + col] = matrix[(row, col)];
        }
    }
    out
}

fn scale_rows_and_copy(matrix: &DMatrix<f32>, scales: &[f32]) -> Vec<f32> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    assert_eq!(rows, scales.len());
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let scale = scales[row];
        for col in 0..cols {
            out[row * cols + col] = matrix[(row, col)] * scale;
        }
    }
    out
}

fn build_first_unfolding(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    shape_m: [usize; 3],
    shape_n: [usize; 3],
) -> Result<Vec<f32>> {
    let expected_rows = shape_m.iter().product::<usize>();
    let expected_cols = shape_n.iter().product::<usize>();
    if rows != expected_rows || cols != expected_cols {
        bail!(
            "shape factors do not match matrix dimensions: rows={}, cols={}, factors {:?} x {:?}",
            rows,
            cols,
            shape_m,
            shape_n
        );
    }

    let [m1, m2, m3] = shape_m;
    let [n1, n2, n3] = shape_n;
    let p2 = m2 * n2;
    let p3 = m3 * n3;
    let mut unfolding = vec![0.0f32; (m1 * n1) * (p2 * p3)];

    for i1 in 0..m1 {
        for j1 in 0..n1 {
            let row_group = i1 * n1 + j1;
            for i2 in 0..m2 {
                for j2 in 0..n2 {
                    let col_group_2 = i2 * n2 + j2;
                    for i3 in 0..m3 {
                        let row = ((i1 * m2 + i2) * m3) + i3;
                        for j3 in 0..n3 {
                            let col = ((j1 * n2 + j2) * n3) + j3;
                            let col_group_3 = i3 * n3 + j3;
                            let out_col = col_group_2 * p3 + col_group_3;
                            unfolding[row_group * (p2 * p3) + out_col] = matrix[row * cols + col];
                        }
                    }
                }
            }
        }
    }

    Ok(unfolding)
}

fn tt_svd_mpo(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    shape_m: [usize; 3],
    shape_n: [usize; 3],
    chi_max: usize,
) -> Result<MpoDecomposition> {
    if chi_max == 0 {
        bail!("chi_max must be greater than zero");
    }

    let expected_rows = shape_m.iter().product::<usize>();
    let expected_cols = shape_n.iter().product::<usize>();
    if rows != expected_rows || cols != expected_cols {
        bail!(
            "matrix dimensions do not match factorization: rows={}, cols={}, factors {:?} x {:?}",
            rows,
            cols,
            shape_m,
            shape_n
        );
    }

    if matrix.len() != rows * cols {
        bail!(
            "matrix length mismatch: got {}, expected {}",
            matrix.len(),
            rows * cols
        );
    }

    let p1 = shape_m[0] * shape_n[0];
    let p2 = shape_m[1] * shape_n[1];
    let p3 = shape_m[2] * shape_n[2];

    let original_frobenius_norm = matrix
        .iter()
        .map(|&value| (value as f64) * (value as f64))
        .sum::<f64>()
        .sqrt() as f32;

    let unfolding = build_first_unfolding(matrix, rows, cols, shape_m, shape_n)?;
    let unfolding = DMatrix::from_row_slice(p1, p2 * p3, &unfolding);
    let svd1 = SVD::new(unfolding, true, true);
    let u1 = svd1.u.ok_or_else(|| anyhow!("stage 1 SVD did not compute U"))?;
    let v_t1 = svd1
        .v_t
        .ok_or_else(|| anyhow!("stage 1 SVD did not compute V^T"))?;
    let s1 = svd1.singular_values;
    let rank1 = chi_max.min(s1.len());
    let first_tail_energy = s1
        .iter()
        .skip(rank1)
        .map(|&value| (value as f64) * (value as f64))
        .sum::<f64>() as f32;

    let u1 = u1.columns(0, rank1).into_owned();
    let v_t1 = v_t1.rows(0, rank1).into_owned();
    let core1 = MpoCore::new(1, shape_m[0], shape_n[0], rank1, matrix_to_row_major_vec(&u1))?;
    let b1 = scale_rows_and_copy(&v_t1, &s1.as_slice()[..rank1]);
    let b1 = DMatrix::from_row_slice(rank1 * p2, p3, &b1);

    let svd2 = SVD::new(b1, true, true);
    let u2 = svd2.u.ok_or_else(|| anyhow!("stage 2 SVD did not compute U"))?;
    let v_t2 = svd2
        .v_t
        .ok_or_else(|| anyhow!("stage 2 SVD did not compute V^T"))?;
    let s2 = svd2.singular_values;
    let rank2 = chi_max.min(s2.len());
    let second_tail_energy = s2
        .iter()
        .skip(rank2)
        .map(|&value| (value as f64) * (value as f64))
        .sum::<f64>() as f32;

    let u2 = u2.columns(0, rank2).into_owned();
    let v_t2 = v_t2.rows(0, rank2).into_owned();
    let core2 = MpoCore::new(rank1, shape_m[1], shape_n[1], rank2, matrix_to_row_major_vec(&u2))?;
    let core3 = MpoCore::new(rank2, shape_m[2], shape_n[2], 1, scale_rows_and_copy(&v_t2, &s2.as_slice()[..rank2]))?;

    let approx_relative_frobenius_error_bound = if original_frobenius_norm == 0.0 {
        0.0
    } else {
        ((first_tail_energy + second_tail_energy).sqrt() as f32) / original_frobenius_norm
    };

    Ok(MpoDecomposition {
        rows,
        cols,
        shape_m,
        shape_n,
        chi_max,
        cores: [core1, core2, core3],
        stats: TtStats {
            original_frobenius_norm,
            first_tail_energy,
            second_tail_energy,
            approx_relative_frobenius_error_bound,
        },
    })
}

#[derive(Debug, Clone, Copy)]
struct EvaluationSummary {
    mean_cosine: f32,
    min_cosine: f32,
    max_cosine: f32,
}

fn evaluate_inputs(
    mpo: &MpoDecomposition,
    inputs: &[Vec<f32>],
    dense_outputs: &[Vec<f32>],
) -> Result<EvaluationSummary> {
    if inputs.is_empty() {
        bail!("at least one input vector is required");
    }
    if inputs.len() != dense_outputs.len() {
        bail!(
            "input/output count mismatch: got {} inputs and {} dense outputs",
            inputs.len(),
            dense_outputs.len()
        );
    }

    let mut sum = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for (input, dense_output) in inputs.iter().zip(dense_outputs.iter()) {
        if input.len() != mpo.cols {
            bail!(
                "input vector length mismatch: got {}, expected {}",
                input.len(),
                mpo.cols
            );
        }
        if dense_output.len() != mpo.rows {
            bail!(
                "dense output length mismatch: got {}, expected {}",
                dense_output.len(),
                mpo.rows
            );
        }
        let mpo_output = mpo.apply(input)?;
        let cosine = cosine_similarity(dense_output, &mpo_output);
        sum += cosine as f64;
        min = min.min(cosine);
        max = max.max(cosine);
    }

    Ok(EvaluationSummary {
        mean_cosine: (sum / inputs.len() as f64) as f32,
        min_cosine: min,
        max_cosine: max,
    })
}

fn generate_random_inputs(samples: usize, hidden_dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SimpleRng::new(seed);
    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        out.push(rng.sample_standard_normal_vec(hidden_dim));
    }
    out
}

fn load_input_vectors(path: &Path, hidden_dim: usize) -> Result<Vec<Vec<f32>>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let row_bytes = hidden_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow!("input vector size overflow"))?;
    if row_bytes == 0 {
        bail!("hidden dimension must be greater than zero");
    }
    if bytes.len() % row_bytes != 0 {
        bail!(
            "input binary length mismatch: got {} bytes, expected a multiple of {} bytes",
            bytes.len(),
            row_bytes
        );
    }

    let mut out = Vec::with_capacity(bytes.len() / row_bytes);
    for row in bytes.chunks_exact(row_bytes) {
        let mut vector = Vec::with_capacity(hidden_dim);
        for chunk in row.chunks_exact(4) {
            vector.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        out.push(vector);
    }
    Ok(out)
}

fn print_tensor_summary(tensor: &TensorMatrix, shape_m: [usize; 3], shape_n: [usize; 3]) {
    println!("tensor: {}", tensor.name);
    println!("shape: {} x {}", tensor.rows, tensor.cols);
    println!("dtype: {:?}", tensor.dtype);
    println!("factors (rows): {:?}", shape_m);
    println!("factors (cols): {:?}", shape_n);
    println!();
}

fn run() -> Result<()> {
    let args = Args::parse();
    let mut chi_values = if args.chi.is_empty() {
        vec![32, 48, 64, 96]
    } else {
        args.chi
    };
    chi_values.retain(|&value| value > 0);
    chi_values.sort_unstable();
    chi_values.dedup();
    if chi_values.is_empty() {
        bail!("at least one chi value must be provided");
    }

    let tensor = load_tensor_matrix(&args.gguf, &args.tensor)?;
    let (shape_m, shape_n) = factor_tensor_shapes(tensor.rows, tensor.cols);
    print_tensor_summary(&tensor, shape_m, shape_n);

    let inputs = if let Some(path) = &args.input_binary {
        let vectors = load_input_vectors(path, tensor.cols)?;
        println!(
            "inputs: {} hidden states loaded from {}\n",
            vectors.len(),
            path.display()
        );
        vectors
    } else {
        let vectors = generate_random_inputs(args.samples, tensor.cols, args.seed);
        println!(
            "inputs: {} random standard-normal hidden states (seed {})\n",
            vectors.len(),
            args.seed
        );
        vectors
    };
    let dense_outputs = precompute_dense_outputs(&tensor.values, tensor.rows, tensor.cols, &inputs)?;

    let dense_params = tensor.rows * tensor.cols;
    let mut best_passing_chi = None;

    println!(
        "{:>6} {:>14} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "chi",
        "params",
        "compress",
        "frob_est",
        "bound",
        "mean_cos",
        "min_cos",
        "max_cos",
        "verdict"
    );

    for &chi in &chi_values {
        let mpo = tt_svd_mpo(
            &tensor.values,
            tensor.rows,
            tensor.cols,
            shape_m,
            shape_n,
            chi,
        )?;
        let frob_error = estimate_relative_frobenius_error(
            &tensor.values,
            &mpo,
            args.frob_samples,
            args.seed ^ (chi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        let eval = evaluate_inputs(&mpo, &inputs, &dense_outputs)?;
        let compression = mpo.compression_ratio();
        let verdict = if eval.mean_cosine >= 0.95 && eval.min_cosine >= 0.90 {
            if best_passing_chi.is_none() {
                best_passing_chi = Some(chi);
            }
            "PASS"
        } else {
            "FAIL"
        };

        println!(
            "{:>6} {:>14} {:>9.2}x {:>12.5} {:>12.5} {:>12.5} {:>12.5} {:>12.5} {:>8}",
            chi,
            mpo.parameter_count(),
            compression,
            frob_error,
            mpo.stats.approx_relative_frobenius_error_bound,
            eval.mean_cosine,
            eval.min_cosine,
            eval.max_cosine,
            verdict
        );

        let _ = dense_params;
        let _ = mpo.stats.original_frobenius_norm;
        let _ = mpo.stats.first_tail_energy;
        let _ = mpo.stats.second_tail_energy;
        let _ = mpo.chi_max;
    }

    if let Some(chi) = best_passing_chi {
        println!(
            "\nthreshold met at chi {} (mean cosine >= 0.95 and min cosine >= 0.90)",
            chi
        );
    } else {
        println!("\nthreshold not met for any requested chi value");
    }

    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factors_for_3072_are_balanced() {
        let factors = balanced_three_factors(3072);
        assert_eq!(factors.iter().product::<usize>(), 3072);
        assert_eq!(factors, [12, 16, 16]);
    }

    #[test]
    fn mpo_round_trip_small_matrix() -> Result<()> {
        let rows = 8;
        let cols = 8;
        let factors = [2, 2, 2];
        let matrix = (0..rows * cols)
            .map(|index| ((index as f32) * 0.17).sin())
            .collect::<Vec<_>>();

        let mpo = tt_svd_mpo(&matrix, rows, cols, factors, factors, 8)?;
        let mut rng = SimpleRng::new(42);
        let vector = rng.sample_standard_normal_vec(cols);
        let dense_output = dense_matvec(&matrix, rows, cols, &vector)?;
        let mpo_output = mpo.apply(&vector)?;
        let cosine = cosine_similarity(&dense_output, &mpo_output);
        assert!(cosine > 0.999, "cosine similarity was {cosine}");

        Ok(())
    }

    #[test]
    fn mpo_round_trip_one_hot_matrix() -> Result<()> {
        let rows = 8;
        let cols = 8;
        let factors = [2, 2, 2];
        let mut matrix = vec![0.0f32; rows * cols];
        matrix[3 * cols + 5] = 1.0;

        let mpo = tt_svd_mpo(&matrix, rows, cols, factors, factors, 8)?;
        let reconstructed = mpo.reconstruct_dense()?;

        let mut max_value = f32::NEG_INFINITY;
        let mut max_index = 0usize;
        let mut max_abs_diff = 0.0f32;
        for (index, (&reference, &approx)) in matrix.iter().zip(reconstructed.iter()).enumerate() {
            let diff = (reference - approx).abs();
            if approx > max_value {
                max_value = approx;
                max_index = index;
            }
            max_abs_diff = max_abs_diff.max(diff);
        }

        assert!(
            max_abs_diff < 1e-5,
            "one-hot reconstruction failed: max_value={max_value} at index {max_index}, max_abs_diff={max_abs_diff}"
        );

        Ok(())
    }
}
