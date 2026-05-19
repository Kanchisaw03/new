use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use llama_gguf::{Engine, EngineConfig};

#[derive(Parser, Debug)]
#[command(version, about = "Baseline GGUF prompt runner using llama-gguf Engine")]
struct Args {
    #[arg(long)]
    gguf: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let model_bytes = std::fs::metadata(&args.gguf)
        .with_context(|| format!("failed to stat GGUF file {}", args.gguf.display()))?
        .len();

    let config = EngineConfig {
        model_path: args.gguf.to_string_lossy().to_string(),
        tokenizer_path: None,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repeat_penalty: 1.0,
        max_tokens: args.max_new_tokens,
        seed: Some(0),
        use_gpu: false,
        max_context_len: None,
        kv_cache_type: Default::default(),
    };

    let started = Instant::now();
    let engine = Engine::load(config).context("failed to load engine")?;
    let output = engine
        .generate(&args.prompt, args.max_new_tokens)
        .context("failed to generate completion")?;
    let elapsed = started.elapsed();

    println!("Baseline engine run");
    println!("  prompt: {}", args.prompt);
    println!("  model file size: {} bytes", model_bytes);
    println!("  generated tokens: {}", args.max_new_tokens);
    println!("  elapsed: {:.3?}", elapsed);
    println!("completion: {}", output);

    Ok(())
}