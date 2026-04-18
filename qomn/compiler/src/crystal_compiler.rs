// ═══════════════════════════════════════════════════════════════════════
// QOMN v1.2 — Oracle → .qomntal Compiler
//
// Compila declaraciones `oracle` de QOMN a formato binario .qomntal
// usando Physics-as-Oracle (PaO): muestrea el oráculo en una grilla,
// cuantiza las activaciones a ternario {-1,0,+1} (BitNet absmean),
// y empaqueta en el formato CRYS binary.
//
// Flujo:
//   oracle f(x) -> y  →  sample grid  →  quantize  →  .qomntal
//
// El .qomntal resultante puede cargarse en Qomni como cualquier crystal
// entrenado con SFT — misma API, mismo formato.
// ═══════════════════════════════════════════════════════════════════════

use std::io::Write;
use crate::ast::{OracleDecl, Program, Decl, Expr, BinaryOp, UnaryOp};

// ── Crystal binary format constants ─────────────────────────────────
const MAGIC:   &[u8; 4] = b"CRYS";
const VERSION: u8       = 1u8;
const ROWS:    usize    = 896;   // hidden dim Qwen-0.5B FFN
const COLS:    usize    = 4864;  // input dim

// ── Oracle sampler ────────────────────────────────────────────────────

/// Sample an oracle function over a parameter grid and produce a
/// ROWS×COLS float matrix of activations.
fn sample_oracle(oracle: &OracleDecl, n_samples: usize) -> Vec<f32> {
    let n_params = oracle.params.len();
    let mut matrix = vec![0f32; ROWS * COLS];

    // Generate stratified parameter samples in [0.001, 10.0]
    let samples: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| {
            (0..n_params)
                .map(|p| {
                    let t = (i * n_params + p) as f64 / n_samples as f64;
                    // log-uniform sampling for physical parameters
                    (0.001f64).powf(1.0 - t) * (100.0f64).powf(t)
                })
                .collect()
        })
        .collect();

    for (sample_idx, params) in samples.iter().enumerate() {
        let y = eval_oracle_f64(oracle, params);
        let y_f32 = y as f32;

        // Encode output into a row of the matrix via sinusoidal projection
        let row = sample_idx % ROWS;
        for col in 0..COLS {
            let freq = (col as f64 * 0.001 + 1.0) as f32;
            let phase = (sample_idx as f32 * 0.1) % std::f32::consts::TAU;
            matrix[row * COLS + col] += (y_f32 * freq + phase).sin();
        }
    }

    // Normalize each row to unit variance
    for row in 0..ROWS {
        let start = row * COLS;
        let end   = start + COLS;
        let mean: f32 = matrix[start..end].iter().sum::<f32>() / COLS as f32;
        let var:  f32 = matrix[start..end].iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / COLS as f32;
        let std   = var.sqrt().max(1e-8);
        for x in &mut matrix[start..end] { *x = (*x - mean) / std; }
    }

    matrix
}

/// Evaluate an oracle AST with given parameter values (f64 for precision).
fn eval_oracle_f64(oracle: &OracleDecl, params: &[f64]) -> f64 {
    // Build simple env: param_name → value
    let env: Vec<(&str, f64)> = oracle.params.iter()
        .zip(params.iter())
        .map(|(p, &v)| (p.name.as_str(), v))
        .collect();

    for stmt in &oracle.body {
        if let crate::ast::Stmt::Return(expr) = stmt {
            return eval_expr_f64(expr, &env);
        }
    }
    0.0
}

fn eval_expr_f64(expr: &Expr, env: &[(&str, f64)]) -> f64 {
    match expr {
        Expr::Int(n)   => *n as f64,
        Expr::Float(f) => *f,
        Expr::Ident(name) => {
            env.iter().find(|(n, _)| *n == name.as_str())
               .map(|(_, v)| *v)
               .unwrap_or(0.0)
        }
        Expr::Binary(op, lhs, rhs) => {
            let l = eval_expr_f64(lhs, env);
            let r = eval_expr_f64(rhs, env);
            match op {
                BinaryOp::Add => l + r,
                BinaryOp::Sub => l - r,
                BinaryOp::Mul => l * r,
                BinaryOp::Div => if r.abs() < 1e-12 { 0.0 } else { l / r },
                BinaryOp::Pow => l.powf(r),
                BinaryOp::Mod => l % r,
                _ => 0.0,
            }
        }
        Expr::Unary(UnaryOp::Neg, e) => -eval_expr_f64(e, env),
        Expr::Call(func, args) => {
            // Built-in math functions
            if let Expr::Ident(name) = func.as_ref() {
                let a: Vec<f64> = args.iter().map(|a| eval_expr_f64(a, env)).collect();
                match name.as_str() {
                    "sin"  => a.first().copied().unwrap_or(0.0).sin(),
                    "cos"  => a.first().copied().unwrap_or(0.0).cos(),
                    "sqrt" => a.first().copied().unwrap_or(0.0).sqrt(),
                    "abs"  => a.first().copied().unwrap_or(0.0).abs(),
                    "log"  => a.first().copied().unwrap_or(1.0).ln(),
                    "exp"  => a.first().copied().unwrap_or(0.0).exp(),
                    _      => 0.0,
                }
            } else { 0.0 }
        }
        _ => 0.0,
    }
}

// ── Ternary quantization (BitNet absmean) ─────────────────────────────

/// Quantize float matrix to ternary {-1, 0, +1} using absmean threshold.
/// Returns: (weights: Vec<i8>, scale: Vec<f32>)
fn quantize_absmean(matrix: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut weights = vec![0i8; rows * cols];
    let mut scales  = vec![0f32; rows];

    for row in 0..rows {
        let start = row * cols;
        let end   = start + cols;
        let slice = &matrix[start..end];

        let mean: f32 = slice.iter().map(|x| x.abs()).sum::<f32>() / cols as f32;
        let threshold = mean;
        scales[row] = mean.max(1e-8);

        for (i, &x) in slice.iter().enumerate() {
            weights[start + i] = if x > threshold { 1 }
                                  else if x < -threshold { -1 }
                                  else { 0 };
        }
    }
    (weights, scales)
}

// ── Crystal packing (2-bit per trit, 4 trits/byte) ───────────────────

/// Pack i8 weights {-1,0,+1} to 2-bit per trit format (4 trits/byte).
/// Encoding: 0→00, +1→01, -1→10
fn pack_2bit(weights: &[i8]) -> Vec<u8> {
    let n_bytes = (weights.len() + 3) / 4;
    let mut packed = vec![0u8; n_bytes];
    for (i, &w) in weights.iter().enumerate() {
        let encoded: u8 = match w { 1 => 1, -1 => 2, _ => 0 };
        let byte_idx = i / 4;
        let bit_off  = (i % 4) * 2;
        packed[byte_idx] |= encoded << bit_off;
    }
    packed
}

// ── Crystal binary writer ─────────────────────────────────────────────

/// Write a single-layer crystal file for an oracle.
/// Format: header(64B) + layer_index(32B) + payload(2-bit packed)
pub fn write_crystal(
    oracle_name: &str,
    weights: &[i8],
    rows: usize,
    cols: usize,
    out_path: &str,
) -> Result<usize, String> {
    let packed = pack_2bit(weights);
    let n_layers = 1usize;  // oracle compiles to a single-layer crystal

    let mut f = std::fs::File::create(out_path)
        .map_err(|e| format!("Cannot create '{}': {}", out_path, e))?;

    // ── Header (64 bytes) ──────────────────────────────────────────
    // Header: MAGIC(4) + VERSION(1) + PAD(3) + N_LAYERS(4) + ARCH(48) + PAD(4) = 64 bytes
    f.write_all(MAGIC).map_err(|e| e.to_string())?;          // 4B
    f.write_all(&[VERSION, 0, 0, 0]).map_err(|e| e.to_string())?; // 4B (version + 3 pad)
    f.write_all(&(n_layers as u32).to_le_bytes()).map_err(|e| e.to_string())?; // 4B

    // arch string: 48 bytes, null-padded
    let arch = format!("oracle-{}", oracle_name);
    let mut arch_bytes = [0u8; 48];
    let arch_src = arch.as_bytes();
    let copy_len = arch_src.len().min(48);
    arch_bytes[..copy_len].copy_from_slice(&arch_src[..copy_len]);
    f.write_all(&arch_bytes).map_err(|e| e.to_string())?;    // 48B

    f.write_all(&[0u8; 8]).map_err(|e| e.to_string())?;      // 8B padding → total 64B

    // ── Layer index (32 bytes per layer) ──────────────────────────
    let offset_payload: u64 = 64 + 32 * n_layers as u64;
    f.write_all(&0u32.to_le_bytes()).map_err(|e| e.to_string())?;           // layer n
    f.write_all(&(offset_payload as u32).to_le_bytes()).map_err(|e| e.to_string())?; // offset
    f.write_all(&(rows as u32).to_le_bytes()).map_err(|e| e.to_string())?;  // rows
    f.write_all(&(cols as u32).to_le_bytes()).map_err(|e| e.to_string())?;  // cols
    f.write_all(&[0u8; 16]).map_err(|e| e.to_string())?;                    // reserved

    // ── Payload (2-bit packed weights) ────────────────────────────
    f.write_all(&packed).map_err(|e| e.to_string())?;

    let file_size = 64 + 32 + packed.len();
    Ok(file_size)
}

// ── Public API ────────────────────────────────────────────────────────

pub struct CompileResult {
    pub oracle_name: String,
    pub out_path:    String,
    pub file_size:   usize,
    pub rows:        usize,
    pub cols:        usize,
    pub n_samples:   usize,
    pub sparsity:    f32,  // fraction of zeros in ternary weights
}

/// Compile all oracle declarations in a program to .qomntal files.
pub fn compile_oracles(prog: &Program, out_dir: &str) -> Vec<Result<CompileResult, String>> {
    let mut results = vec![];

    for decl in &prog.decls {
        if let Decl::Oracle(oracle) = decl {
            let result = compile_oracle(oracle, out_dir);
            results.push(result);
        }
    }
    results
}

/// Compile a single oracle to a .qomntal file.
pub fn compile_oracle(oracle: &OracleDecl, out_dir: &str) -> Result<CompileResult, String> {
    let n_samples = ROWS * 4;  // 4 samples per output row for good coverage

    // 1. Sample oracle over parameter grid → float matrix
    let matrix = sample_oracle(oracle, n_samples);

    // 2. Quantize to ternary
    let (weights, _scales) = quantize_absmean(&matrix, ROWS, COLS);

    // 3. Stats
    let zeros    = weights.iter().filter(|&&w| w == 0).count();
    let sparsity = zeros as f32 / weights.len() as f32;

    // 4. Write .qomntal
    let out_path = format!("{}/{}.qomntal", out_dir, oracle.name);
    let file_size = write_crystal(&oracle.name, &weights, ROWS, COLS, &out_path)?;

    Ok(CompileResult {
        oracle_name: oracle.name.clone(),
        out_path,
        file_size,
        rows: ROWS,
        cols: COLS,
        n_samples,
        sparsity,
    })
}
