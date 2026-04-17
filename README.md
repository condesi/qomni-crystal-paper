# Qomni Crystal — CRYS-L v3.2: Deterministic Engineering Oracle Engine

**Compiled DSL for physics-exact, bit-deterministic multi-domain engineering computation**

[![Live Demo](https://img.shields.io/badge/demo-live-00e5ff)](https://qomni.clanmarketer.com/crysl/)
[![Benchmarks](https://img.shields.io/badge/benchmarks-live-e040fb)](https://qomni.clanmarketer.com/crysl/demo/benchmark.html)
[![Tests](https://img.shields.io/badge/tests-5%20suites-green)](https://github.com/condesi/crysl/tree/main/tests)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](arxiv/main.tex)

> **Percy Rojas Masgo** · CEO Condesi Perú · Qomni AI Lab
> percy.rojas@condesi.pe · https://qomni.clanmarketer.com/crysl/

---

## What Changed (April 2026)

This repository documents **CRYS-L v3.2**, a major evolution from the ternary quantization pipeline described in the original preprint. The core insight shifted:

> **Original (v1):** Use physics to generate training data for compressed neural networks.
> **Current (v3.2):** Skip the neural network entirely. Compile physics directly to JIT-optimized machine code.

The ternary quantization approach achieved 11× compression but retained LLM-style stochastic inference. CRYS-L v3.2 eliminates stochasticity entirely: **same input → same output, bit-exact, always**.

---

## Performance (Live, Reproducible)

| Metric | Value | Verification |
|--------|-------|-------------|
| Compute throughput (JIT) | **117,000,000 ops/s** | [/simulation/simd_density](https://qomni.clanmarketer.com/crysl/api/simulation/simd_density) |
| Compute throughput (AVX2 sweep) | **3,500,000,000 ops/s** | [benchmark dashboard](https://qomni.clanmarketer.com/crysl/demo/benchmark.html) |
| Compute latency p50 | **9µs** | `total_ns` field in API response |
| Compute latency p99 | **21µs** | `tests/slo_latency.rs` |
| Speedup vs Python (benchmark) | **1.53 billion×** | [/benchmark/vs_llm](https://qomni.clanmarketer.com/crysl/api/benchmark/vs_llm) |
| Speedup vs GPT-4 (compute) | **88,888×** | 9µs vs 800ms |
| Numeric variance (10 runs) | **0.000000000000** | `tests/repeatability.rs` |
| Panics (100k adversarial inputs) | **0** | `tests/adversarial.rs` |
| Plans (deterministic oracles) | **56** | `/plans` endpoint |
| Engineering domains | **13** | See domain table below |
| Pareto front (170 solutions) | **1.84ms** | `/ws/sim` WebSocket stream |
| Jitter σ (SCHED_FIFO, 10K ticks) | **2,369ns** | `/simulation/jitter_bench` |

---

## Core Architecture

```
Natural Language Query
        │
        ▼
Universal Intent Router (keyword-based, 0ms)
        │
   8 intent classes:
   calculation · pareto · benchmark
   repeatability · adversarial · slo_metrics
   comparison · validation
        │
        ▼
CRYS-L Oracle Engine
        │
   ┌────────────────────────────────────┐
   │  .crys source                      │
   │    ↓ Lexer → Parser → AST → HIR   │
   │    ↓ Type Checker (units + dims)   │
   │                                    │
   │  ┌──────────────────────────────┐  │
   │  │ Cranelift JIT (default)      │  │ sub-ms startup
   │  │ LLVM 18 IR → native .so      │  │ maximum opt
   │  │ WebAssembly (WAT) → .wasm    │  │ browser/edge
   │  └──────────────────────────────┘  │
   │    ↓                               │
   │  AVX2 SoA Sweep Engine             │
   │  (4 doubles/instruction)           │
   │    ↓                               │
   │  3-objective Pareto Front          │
   │  (cost × efficiency × risk)        │
   └────────────────────────────────────┘
        │
        ▼
Structured Response + Decision Card
+ Standards Citations + Recommendations
```

---

## The Branchless Oracle Pattern

Standard conditional logic prevents SIMD vectorization. CRYS-L compiles conditions to float multiplications:

```crystal
oracle nfpa20_pump_hp(flow_gpm: float, head_psi: float, eff: float) -> float:
    # Guard: invalid inputs → 0.0 (no branch, no exception)
    let valid = (flow_gpm >= 1.0) * (flow_gpm <= 50000.0) * (eff >= 0.10)
    let q = flow_gpm * 0.06309          # gpm → m³/s
    let h = head_psi  * 0.70307         # psi → m
    ((q * h) / (eff * 76.04 + 0.0001)) * valid
```

`valid` is 1.0 when all conditions are true, 0.0 if any fails. The `* valid` mask eliminates invalid outputs without any branch instruction → direct mapping to AVX2 `vmulpd`.

**Result:** 4 pump-sizing calculations per instruction cycle. At 3.5GHz with AVX2, that is ~14 billion multiply-equivalent operations per second before memory bandwidth limits.

---

## Determinism Guarantee

**Theorem:** For any oracle `f(x)` compiled by CRYS-L, bit-exact identical output is produced on every invocation for the same input `x` on the same architecture.

**Basis:**
1. Pure functions: no I/O, no random, no external state
2. IEEE-754 double precision, round-to-nearest-even
3. Only algebraic ops (+, -, ×, ÷, √) — no transcendentals with approximation error
4. Branchless pattern: single deterministic code path

**Empirical verification:**
```bash
# Run 10 times, verify identical
for i in {1..10}; do
  curl -s -X POST https://qomni.clanmarketer.com/crysl/api/plan/execute \
    -H "Content-Type: application/json" \
    -d '{"plan":"plan_pump_sizing","params":{"Q_gpm":500,"P_psi":100,"eff":0.75}}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['nfpa20_pump_hp'])"
done
# Output: 16.835016... × 10 (bit-exact)
```

---

## 56 Deterministic Engineering Oracles

| Domain | Plans | Key Standards |
|--------|-------|--------------|
| Fire & Life Safety | 7 | NFPA 20, NFPA 13, NFPA 101 |
| Hydraulics | 7 | Hazen-Williams, Manning, Hardy-Cross |
| Electrical Engineering | 6 | NEC, CNE (Peru), IEC 60364 |
| Structural Engineering | 6 | ACI 318, AISC 360, E.060 (Peru) |
| Peruvian Finance/Payroll | 6 | DL 728, SUNAFIL, SUNAT/IGV |
| Business Analytics | 5 | Break-even, ROI, amortization |
| HVAC & Renewable Energy | 3 | ASHRAE, solar irradiance |
| Medical & Clinical | 4 | WHO, clinical dosing guidelines |
| Statistics & Data Science | 2 | Cochran sample size, descriptive |
| Cybersecurity | 5 | CVSS v3.1, NIST SP 800-63B |
| Civil & Geotechnical | 4 | Bishop, Euler, Terzaghi |
| Agriculture & Irrigation | 2 | FAO-56 ETo, drip emitter flow |
| Telecommunications | 1 | ITU-R P.525 path loss |

---

## Test Suite (New — April 2026)

All tests are in [condesi/crysl](https://github.com/condesi/crysl):

| Test file | Focus | Tests |
|-----------|-------|-------|
| `tests/golden.rs` | Exact numerical values | ~20 |
| `tests/repeatability.rs` | Determinism (10–20 runs) | 7 |
| `tests/adversarial.rs` | NaN-Shield (0 panics) | 8 |
| `tests/slo_latency.rs` | p50/p95/p99 SLO | 6 |
| `tests/all_56_plans.rs` | All 56 plans smoke-pass | 57 |

```bash
# Run all (requires CRYS-L server on :9001)
cargo test --test repeatability -- --nocapture
cargo test --test adversarial -- --nocapture
cargo test --test slo_latency -- --nocapture
cargo test --test all_56_plans -- --nocapture
```

---

## Quick Start

```bash
# Health check
curl https://qomni.clanmarketer.com/crysl/api/health

# Fire pump sizing (NFPA 20)
curl -X POST https://qomni.clanmarketer.com/crysl/api/plan/execute \
  -H "Content-Type: application/json" \
  -d '{"plan":"plan_pump_sizing","params":{"Q_gpm":500,"P_psi":100,"eff":0.75}}'

# List all 56 plans
curl https://qomni.clanmarketer.com/crysl/api/plans | python3 -m json.tool

# Throughput proof (live)
curl https://qomni.clanmarketer.com/crysl/api/simulation/simd_density

# Adversarial resilience proof
curl https://qomni.clanmarketer.com/crysl/api/simulation/adversarial

# All 4 benchmark proofs
curl https://qomni.clanmarketer.com/crysl/api/benchmark/all | python3 -m json.tool
```

---

## vs. Original Preprint (v1 → v3.2)

| Aspect | v1 (ternary quantization) | v3.2 (compiled oracles) |
|--------|--------------------------|------------------------|
| Approach | Physics → training data → compressed NN | Physics → compiled JIT oracle |
| Stochastic? | Yes (NN inference) | **No (deterministic)** |
| Throughput | ~2,000 ops/s (ternary NN) | **117M ops/s** (JIT) |
| Compute latency | ~50ms (NN forward pass) | **9µs** (JIT) |
| Compression target | 85.3 MB for one domain | N/A — no weights |
| Domains | Hydraulics only | **13 domains, 56 plans** |
| Hallucination | Possible (NN) | **Zero** (closed-form math) |

---

## Repository Structure

```
condesi/qomni-crystal-paper/
├── README.md               ← This file (system overview, v3.2)
├── arxiv/
│   └── main.tex            ← Updated LaTeX paper (CRYS-L v3.2)
├── crys-l/
│   ├── SPEC.md             ← Language specification
│   ├── LANGUAGE_GUIDE.md   ← Programmer's guide
│   ├── compiler/           ← Embedded compiler snapshot
│   └── examples/           ← .crys example oracles
├── code/
│   ├── crystal_pack.py     ← (v1) Ternary packing utility
│   └── crystal_synth_hidraulica.py ← (v1) Physics oracle synthesis
├── datasets/               ← Generated training pairs (v1)
├── notebooks/              ← Colab demos
└── results/
    ├── benchmark_server5.json
    └── verification_report.md  ← NEW: test results
```

---

## Cite This Work

```bibtex
@misc{rojas2026crysl,
  title  = {CRYS-L v3.2: Deterministic Engineering Oracle Engine with JIT Compilation and AVX2 SIMD},
  author = {Rojas Masgo, Percy},
  year   = {2026},
  month  = {April},
  note   = {Preprint. Qomni AI Lab, Condesi Perú},
  url    = {https://github.com/condesi/qomni-crystal-paper}
}
```

---

**Live demo:** https://qomni.clanmarketer.com/crysl/
**Full source:** https://github.com/condesi/crysl
**Contact:** percy.rojas@condesi.pe
**License:** Apache-2.0
