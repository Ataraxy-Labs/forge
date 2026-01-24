//! Batcher module - handles tensor preparation and sampling

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::request::SamplingParams;

/// Token sampler with configurable parameters
pub struct Sampler {
    rng: StdRng,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample a token from logits
    pub fn sample(&mut self, logits: &Tensor, params: &SamplingParams) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let logits = logits.squeeze(0)?; // Remove batch dim if present

        // Get last token logits if sequence
        let logits = if logits.dims().len() > 1 {
            let seq_len = logits.dim(0)?;
            logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?
        } else {
            logits
        };

        // Apply temperature
        let logits = if params.temperature > 0.0 && params.temperature != 1.0 {
            (logits / params.temperature)?
        } else {
            logits
        };

        // Convert to probabilities
        let probs = candle_nn::ops::softmax(&logits, 0)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Apply top-p (nucleus) sampling
        let token = if let Some(top_p) = params.top_p {
            self.sample_top_p(&probs_vec, top_p)
        } else if let Some(top_k) = params.top_k {
            self.sample_top_k(&probs_vec, top_k)
        } else if params.temperature == 0.0 {
            // Greedy
            probs_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        } else {
            self.sample_multinomial(&probs_vec)
        };

        Ok(token)
    }

    fn sample_top_p(&mut self, probs: &[f32], top_p: f64) -> u32 {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += *p as f64;
            if cumsum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        let filtered: Vec<(usize, f32)> = indexed.into_iter().take(cutoff_idx).collect();
        let sum: f32 = filtered.iter().map(|(_, p)| p).sum();

        let r: f32 = self.rng.gen::<f32>() * sum;
        let mut cumsum = 0.0;
        for (idx, p) in filtered {
            cumsum += p;
            if cumsum >= r {
                return idx as u32;
            }
        }
        0
    }

    fn sample_top_k(&mut self, probs: &[f32], top_k: usize) -> u32 {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(top_k);

        let sum: f32 = indexed.iter().map(|(_, p)| p).sum();
        let r: f32 = self.rng.gen::<f32>() * sum;

        let mut cumsum = 0.0;
        for (idx, p) in indexed {
            cumsum += p;
            if cumsum >= r {
                return idx as u32;
            }
        }
        0
    }

    fn sample_multinomial(&mut self, probs: &[f32]) -> u32 {
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }
}

/// Prepare input tensors for batched inference
pub fn prepare_input(tokens: &[u32], device: &Device) -> Result<Tensor> {
    let tokens: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
    let tensor = Tensor::new(tokens.as_slice(), device)?;
    Ok(tensor.unsqueeze(0)?) // Add batch dimension
}

/// Prepare position IDs for input
pub fn prepare_positions(seq_len: usize, offset: usize, device: &Device) -> Result<Tensor> {
    let positions: Vec<i64> = (offset..offset + seq_len).map(|i| i as i64).collect();
    let tensor = Tensor::new(positions.as_slice(), device)?;
    Ok(tensor.unsqueeze(0)?)
}
