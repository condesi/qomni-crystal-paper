# Code — Qomni Crystal Pipeline

## Files

### `crystal_synth_hidraulica.py`
Physics-as-Oracle data generator for hydraulic engineering domain.
- 6 generators: Manning, Darcy-Weisbach (Colebrook-White), Bernoulli, Joukowsky, Hazen-Williams, Continuidad
- 9 Peruvian cities with altitude data
- IS.010 / IS.020 compliance checks
- Output: Alpaca JSONL format

**Usage:**
```bash
python crystal_synth_hidraulica.py --n 2000 --output hidraulica_data.jsonl
```

### `crystal_pack.py`
Converts BF16 safetensors → `.crystal` binary format (2-bit ternary).
- AbsMean quantization (BitNet b1.58)
- 4 weights/byte MSB-first packing
- CRYS binary header + layer index

**Usage:**
```bash
python crystal_pack.py --input ./hidraulica_crystal --output hidraulica.crystal --domain hidraulica
```

### `colab_crystal_hidraulica.ipynb`
End-to-end Colab notebook (T4 GPU, free tier):
1. Verify GPU
2. Install dependencies (TRL 1.0+ compatible)
3. Generate 2,000 physics pairs inline
4. Fine-tune Qwen 2.5 0.5B
5. Pack to `.crystal`
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
scp -P 2291 hidraulica.crystal root@SERVER:/opt/nexus/crystals/

# Register
curl -X POST http://SERVER:8090/qomni/crystal/register \
  -H 'Content-Type: application/json' \
  -d '{"domain":"hidraulica","path":"/opt/nexus/crystals/hidraulica.crystal"}'

# Activate
curl -X POST http://SERVER:8090/qomni/crystal/activate \
  -d '{"domain":"hidraulica"}'
```
