// OpenAI privacy-filter (and OpenMed/privacy-filter-nemotron) PII detector.
//
//   cargo run --example openai-privacy-filter --release -- \
//       --model-id OpenMed/privacy-filter-nemotron \
//       --text "Patient Sarah Johnson, MRN 4872910, phone 415-555-0123."

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::openai_privacy_filter::{
    viterbi, Config, OpenAIPrivacyFilterForTokenClassification,
};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Run on CPU rather than CUDA/Metal.
    #[arg(long)]
    cpu: bool,

    /// HuggingFace model id (e.g. "OpenMed/privacy-filter-nemotron",
    /// "openai/privacy-filter").
    #[arg(long, default_value = "OpenMed/privacy-filter-nemotron")]
    model_id: String,

    /// Model revision (branch / tag / commit hash).
    #[arg(long, default_value = "main")]
    revision: String,

    /// Local directory of model files. Overrides --model-id when set.
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// Text to scan.
    #[arg(long)]
    text: String,

    /// Use plain argmax decoding instead of constrained Viterbi.
    #[arg(long)]
    argmax: bool,
}

fn pick_device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    if candle::utils::cuda_is_available() {
        return Ok(Device::new_cuda(0)?);
    }
    if candle::utils::metal_is_available() {
        return Ok(Device::new_metal(0)?);
    }
    Ok(Device::Cpu)
}

fn locate_files(args: &Args) -> Result<(PathBuf, PathBuf, Vec<PathBuf>)> {
    if let Some(local) = &args.model_path {
        let cfg = local.join("config.json");
        let tok = local.join("tokenizer.json");
        let safetensors_index = local.join("model.safetensors.index.json");
        let weights = if safetensors_index.exists() {
            let idx: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&safetensors_index)?)?;
            let map = idx
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| anyhow!("weight_map missing in safetensors index"))?;
            let mut shards: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shards.sort();
            shards.dedup();
            shards.into_iter().map(|s| local.join(s)).collect()
        } else {
            vec![local.join("model.safetensors")]
        };
        return Ok((cfg, tok, weights));
    }
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));
    let cfg = repo.get("config.json")?;
    let tok = repo.get("tokenizer.json")?;
    let weights = match repo.get("model.safetensors.index.json") {
        Ok(idx_path) => {
            let idx: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(idx_path)?)?;
            let map = idx
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| anyhow!("weight_map missing in safetensors index"))?;
            let mut shards: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shards.sort();
            shards.dedup();
            shards
                .into_iter()
                .map(|s| repo.get(&s))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
        Err(_) => vec![repo.get("model.safetensors")?],
    };
    Ok((cfg, tok, weights))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = pick_device(args.cpu)?;

    println!(
        "loading config + tokenizer + weights from {} (rev {})",
        args.model_id, args.revision
    );
    let (cfg_path, tok_path, weight_paths) = locate_files(&args)?;
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("{e}"))?;
    let num_classes = (cfg.num_labels - 1) / 4;
    println!(
        "config: {} layers, {} experts top-{}, {} BIOES labels over {} classes",
        cfg.num_hidden_layers,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
        cfg.num_labels,
        num_classes
    );

    let dtype = DType::BF16;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, &device)? };
    let mut model = OpenAIPrivacyFilterForTokenClassification::new(&cfg, vb)?;

    let enc = tokenizer
        .encode(args.text.as_str(), true)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let ids = enc.get_ids();
    let offsets = enc.get_offsets().to_vec();
    println!("tokenized to {} tokens", ids.len());

    let input_ids = Tensor::new(ids, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input_ids)?;
    let logits = logits.squeeze(0)?;
    let spans = if args.argmax {
        viterbi::decode_argmax(&logits, num_classes)?
    } else {
        viterbi::decode(&logits, num_classes)?
    };

    println!("\nfound {} PII spans:", spans.len());
    for (start_tok, end_tok, class_idx) in spans {
        let char_start = offsets.get(start_tok).map(|o| o.0).unwrap_or(0);
        let char_end = offsets
            .get(end_tok.saturating_sub(1))
            .map(|o| o.1)
            .unwrap_or(args.text.len());
        let slice = args
            .text
            .get(char_start..char_end)
            .unwrap_or("<offset error>");
        println!(
            "  [{:>3}..{:<3}] class={} text={:?}",
            char_start, char_end, class_idx, slice
        );
    }

    Ok(())
}
