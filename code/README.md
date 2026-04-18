# Code — Qomni Crystal Pipeline

## Files

### `qomn_synth_hidraulica.py`
Physics-as-Oracle data generator for hydraulic engineering domain.
- 6 generators: Manning, Darcy-Weisbach (Colebrook-White), Bernoulli, Joukowsky, Hazen-Williams, Continuidad
- 9 Peruvian cities with altitude data
- IS.010 / IS.020 compliance checks
- Output: Alpaca JSONL format

**Usage:**
```bash
python qomn_synth_hidraulica.py --n 2000 --output hidraulica_data.jsonl
```

### `qomn_pack.py`
Converts BF16 safetensors → `.qomntal` binary format (2-bit ternary).
- AbsMean quantization (BitNet b1.58)
- 4 weights/byte MSB-first packing
- CRYS binary header + layer index

**Usage:**
```bash
python qomn_pack.py --input ./hidraulica_crystal --output hidraulica.qomntal --domain hidraulica
```

### `colab_qomn_hidraulica.ipynb`
End-to-end Colab notebook (T4 GPU, free tier):
1. Verify GPU
2. Install dependencies (TRL 1.0+ compatible)
3. Generate 2,000 physics pairs inline
4. Fine-tune Qwen 2.5 0.5B
5. Pack to `.qomntal`
6. Download result

**Run on:** [Google Colab](https://colab.research.google.com/) with T4 GPU runtime.

## Requirements

```
torch >= 2.0
transformers >= 5.0
trl >= 1.0
safetensors
datasets
```

## Deploy to Server

```bash
# Upload crystal
scp -P 2291 hidraulica.qomntal root@SERVER:/opt/nexus/crystals/

# Register
curl -X POST http://SERVER:8090/qomni/qomn/register \
  -H 'Content-Type: application/json' \
  -d '{"domain":"hidraulica","path":"/opt/nexus/crystals/hidraulica.qomntal"}'

# Activate
curl -X POST http://SERVER:8090/qomni/qomn/activate \
  -d '{"domain":"hidraulica"}'
```
