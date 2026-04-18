#!/usr/bin/env python3
"""
qomn_pack.py — Qomni BitNet Crystal Packer
==============================================
Convierte pesos entrenados (BF16/FP16 .safetensors) al formato binario .qomntal
usando quantización absmean BitNet b1.58: {-1, 0, +1} empaquetados a 2 bits/peso.

Uso:
    python3 qomn_pack.py \
        --input model.safetensors \
        --output hidraulica.qomntal \
        --arch bitnet-500m \
        --domain hidraulica

    python3 qomn_pack.py --input ./lora_output/ --output hidraulica.qomntal

Formato .qomntal (binario little-endian):
    [0:4]   magic       b"CRYS"
    [4:6]   version     u16 = 1
    [6:8]   n_layers    u16
    [8:40]  arch_name   [u8; 32] null-padded
    [40:44] hidden_dim  u32
    [44:48] ffn_dim     u32
    [48:50] n_heads     u16
    [50:52] n_kv_heads  u16
    [52:64] reserved    [u8; 12]
    --- header = 64 bytes ---
    Por cada capa (variable orden):
        LayerHeader (32 bytes):
            [0:4]   layer_idx   u32
            [4:8]   offset      u32  (byte offset desde inicio del archivo)
            [8:12]  n_weights   u32  (total de pesos ternarios)
            [12:16] rows        u32
            [16:20] cols        u32
            [20:32] reserved    [u8; 12]
        Payload ternario:
            2 bits/peso, MSB-first: 00=+1, 01=0, 10=-1
"""

import argparse
import struct
import sys
import os
import json
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    print("ERROR: pip install torch")
    sys.exit(1)

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("AVISO: safetensors no instalado — pip install safetensors")

# ── Constantes formato .qomntal ───────────────────────────────────────────────
CRYSTAL_MAGIC   = b"CRYS"
CRYSTAL_VERSION = 1
HEADER_SIZE     = 64
LAYER_HEADER_SIZE = 32

# Valores de empaquetado 2-bit (MSB-first)
PACK_POS = 0b00  # +1
PACK_ZER = 0b01  # 0
PACK_NEG = 0b10  # -1

# Capas relevantes para BitNet (filtros de nombre)
LAYER_PATTERNS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "lm_head",
]

# ── Quantización BitNet b1.58 ─────────────────────────────────────────────────

def quantize_absmean(W: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    BitNet b1.58 absmean quantization.
    scale = mean(|W|)
    W_q = round(W / scale).clamp(-1, 1)
    Retorna: (ternary i8 tensor, scale float)
    """
    W_f = W.to(torch.float32)
    scale = W_f.abs().mean().item()
    if scale < 1e-8:
        return torch.zeros_like(W_f, dtype=torch.int8), 1.0
    W_q = (W_f / scale).round().clamp(-1.0, 1.0).to(torch.int8)
    return W_q, scale

def pack_ternary(weights_i8: torch.Tensor) -> bytes:
    """
    Empaqueta tensor i8 {-1, 0, +1} a 2 bits/peso, MSB-first.
    4 pesos por byte: bits [7:6] [5:4] [3:2] [1:0]
    """
    flat = weights_i8.flatten().tolist()
    n = len(flat)
    n_bytes = (n + 3) // 4
    packed = bytearray(n_bytes)
    for i, w in enumerate(flat):
        byte_idx = i // 4
        bit_pos  = (3 - (i % 4)) * 2  # MSB-first: peso 0 → bits 7:6
        if w == 1:
            bits = PACK_POS
        elif w == -1:
            bits = PACK_NEG
        else:
            bits = PACK_ZER
        packed[byte_idx] |= (bits << bit_pos)
    return bytes(packed)

def unpack_ternary_verify(packed: bytes, n: int) -> list[int]:
    """Verifica round-trip: desempaqueta y retorna lista de enteros."""
    out = []
    for i in range(n):
        byte_idx = i // 4
        shift    = (3 - (i % 4)) * 2
        bits = (packed[byte_idx] >> shift) & 0b11
        if bits == PACK_POS:
            out.append(1)
        elif bits == PACK_NEG:
            out.append(-1)
        else:
            out.append(0)
    return out

# ── Detección de arquitectura ─────────────────────────────────────────────────

def detect_arch(tensors: dict, config: Optional[dict] = None) -> dict:
    """Inferir dimensiones de arquitectura desde los tensores."""
    arch = {
        "name": "bitnet-unknown",
        "hidden_dim": 0,
        "ffn_dim": 0,
        "n_heads": 0,
        "n_kv_heads": 0,
    }

    if config:
        arch["hidden_dim"] = config.get("hidden_size", 0)
        arch["ffn_dim"]    = config.get("intermediate_size", 0)
        arch["n_heads"]    = config.get("num_attention_heads", 0)
        arch["n_kv_heads"] = config.get("num_key_value_heads", arch["n_heads"])
        return arch

    # Inferir desde tensores
    for name, t in tensors.items():
        if "q_proj.weight" in name and arch["hidden_dim"] == 0:
            arch["hidden_dim"] = t.shape[-1]
        if "gate_proj.weight" in name and arch["ffn_dim"] == 0:
            arch["ffn_dim"] = t.shape[0]

    # Estimar cabezas (asunción: head_dim = 128 para la mayoría de modelos)
    if arch["hidden_dim"] > 0:
        arch["n_heads"]    = arch["hidden_dim"] // 128
        arch["n_kv_heads"] = arch["n_heads"]

    return arch

def filter_layers(tensors: dict, patterns: list[str]) -> dict:
    """Filtrar solo las capas de proyección relevantes."""
    filtered = {}
    for name, t in tensors.items():
        if any(p in name for p in patterns) and t.ndim == 2:
            filtered[name] = t
    return filtered

# ── Escritura del archivo .qomntal ────────────────────────────────────────────

def write_crystal(
    output_path: str,
    layers: list[tuple[str, torch.Tensor]],  # [(name, weight_tensor)]
    arch: dict,
    verbose: bool = True,
):
    """
    Escribe el archivo binario .qomntal.
    layers: lista ordenada de (nombre, tensor BF16/FP16)
    """
    n_layers = len(layers)
    if n_layers > 65535:
        raise ValueError(f"Demasiadas capas: {n_layers}")

    # ── Paso 1: quantizar y empaquetar todos los pesos ────────────────────────
    packed_layers = []
    total_original_bytes = 0
    total_packed_bytes   = 0
    total_weights        = 0
    sparsity_counts      = {"pos": 0, "zero": 0, "neg": 0}

    print(f"\nQuantizando {n_layers} capas...")
    for i, (name, W) in enumerate(layers):
        W_q, scale = quantize_absmean(W)
        packed     = pack_ternary(W_q)

        # Estadísticas sparsidad
        flat = W_q.flatten().tolist()
        sparsity_counts["pos"]  += sum(1 for w in flat if w == 1)
        sparsity_counts["zero"] += sum(1 for w in flat if w == 0)
        sparsity_counts["neg"]  += sum(1 for w in flat if w == -1)

        # Verificar round-trip (solo primeros 100 pesos)
        if i == 0:
            check = unpack_ternary_verify(packed, min(100, len(flat)))
            assert check == flat[:len(check)], "ERROR: round-trip pack/unpack fallido"

        n_weights   = W.numel()
        orig_bytes  = n_weights * 2  # BF16 = 2 bytes/peso
        pack_bytes  = len(packed)

        total_original_bytes += orig_bytes
        total_packed_bytes   += pack_bytes
        total_weights        += n_weights

        packed_layers.append({
            "name":      name,
            "rows":      W.shape[0],
            "cols":      W.shape[1],
            "n_weights": n_weights,
            "packed":    packed,
            "scale":     scale,
        })

        if verbose:
            ratio = orig_bytes / pack_bytes
            print(f"  [{i:3d}/{n_layers}] {name:<50} "
                  f"{W.shape[0]}×{W.shape[1]:>5}  "
                  f"{orig_bytes/1024:.0f}KB → {pack_bytes/1024:.1f}KB  "
                  f"({ratio:.1f}x)")

    # ── Paso 2: calcular offsets ──────────────────────────────────────────────
    # Estructura: HEADER(64) + LAYER_INDEX(32×n) + PAYLOADS
    index_size   = LAYER_HEADER_SIZE * n_layers
    payload_start = HEADER_SIZE + index_size

    offsets = []
    cursor  = payload_start
    for pl in packed_layers:
        offsets.append(cursor)
        cursor += len(pl["packed"])

    # ── Paso 3: escribir archivo ──────────────────────────────────────────────
    arch_bytes = arch["name"].encode("utf-8")[:31].ljust(32, b"\x00")

    with open(output_path, "wb") as f:
        # Header (64 bytes)
        f.write(CRYSTAL_MAGIC)                                   # [0:4]
        f.write(struct.pack("<H", CRYSTAL_VERSION))              # [4:6]
        f.write(struct.pack("<H", n_layers))                     # [6:8]
        f.write(arch_bytes)                                      # [8:40]
        f.write(struct.pack("<I", arch["hidden_dim"]))           # [40:44]
        f.write(struct.pack("<I", arch["ffn_dim"]))              # [44:48]
        f.write(struct.pack("<H", arch["n_heads"]))              # [48:50]
        f.write(struct.pack("<H", arch["n_kv_heads"]))           # [50:52]
        f.write(b"\x00" * 12)                                    # [52:64] reserved

        # Layer index (32 bytes × n_layers)
        for i, (pl, offset) in enumerate(zip(packed_layers, offsets)):
            f.write(struct.pack("<I", i))                        # layer_idx
            f.write(struct.pack("<I", offset))                   # offset
            f.write(struct.pack("<I", pl["n_weights"]))          # n_weights
            f.write(struct.pack("<I", pl["rows"]))               # rows
            f.write(struct.pack("<I", pl["cols"]))               # cols
            f.write(b"\x00" * 12)                                # reserved

        # Payloads ternarios
        for pl in packed_layers:
            f.write(pl["packed"])

    file_size = os.path.getsize(output_path)

    # ── Reporte final ─────────────────────────────────────────────────────────
    total = sum(sparsity_counts.values())
    sparsity_pct = sparsity_counts["zero"] / total * 100 if total > 0 else 0
    compression  = total_original_bytes / total_packed_bytes if total_packed_bytes > 0 else 0

    print(f"\n{'═'*60}")
    print(f"  Crystal generado: {output_path}")
    print(f"  Arquitectura:     {arch['name']}")
    print(f"  Capas:            {n_layers}")
    print(f"  Total pesos:      {total_weights:,}")
    print(f"  BF16 original:    {total_original_bytes/1024**2:.1f} MB")
    print(f"  Crystal packed:   {file_size/1024**2:.1f} MB")
    print(f"  Compresión:       {compression:.1f}x")
    print(f"  Sparsidad:        {sparsity_pct:.1f}% zeros")
    print(f"  Distribución:     +1={sparsity_counts['pos']/total*100:.1f}%  "
          f"0={sparsity_counts['zero']/total*100:.1f}%  "
          f"-1={sparsity_counts['neg']/total*100:.1f}%")
    print(f"{'═'*60}\n")

    return file_size

# ── Cargadores de modelos ─────────────────────────────────────────────────────

def load_model(path: str) -> tuple[dict, Optional[dict]]:
    """
    Carga pesos desde varias fuentes:
    - archivo .safetensors
    - directorio con múltiples .safetensors (HuggingFace sharded)
    - directorio con pytorch_model.bin
    Retorna (tensors_dict, config_dict_or_None)
    """
    p = Path(path)
    config = None

    # Intentar cargar config.json
    config_path = p if p.is_dir() else p.parent
    config_file = config_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"Config: {config_file}")

    if p.is_file() and p.suffix == ".safetensors":
        if not HAS_SAFETENSORS:
            raise ImportError("pip install safetensors")
        print(f"Cargando safetensors: {p}")
        tensors = load_safetensors(str(p))
        return tensors, config

    if p.is_dir():
        # Buscar safetensors sharded
        sf_files = sorted(p.glob("*.safetensors"))
        if sf_files and HAS_SAFETENSORS:
            print(f"Cargando {len(sf_files)} shards safetensors...")
            tensors = {}
            for sf in sf_files:
                print(f"  {sf.name}")
                tensors.update(load_safetensors(str(sf)))
            return tensors, config

        # Fallback: pytorch_model.bin
        bin_files = sorted(p.glob("pytorch_model*.bin"))
        if bin_files:
            print(f"Cargando {len(bin_files)} pytorch bin files...")
            tensors = {}
            for bf in bin_files:
                print(f"  {bf.name}")
                tensors.update(torch.load(str(bf), map_location="cpu"))
            return tensors, config

    raise FileNotFoundError(f"No se encontraron pesos en: {path}")

# ── Diagnóstico (sin empaquetar) ──────────────────────────────────────────────

def diagnose_model(path: str):
    """Muestra estadísticas de quantización sin escribir el archivo."""
    tensors, config = load_model(path)
    layers = filter_layers(tensors, LAYER_PATTERNS)

    print(f"\nModelo: {path}")
    print(f"Total tensores: {len(tensors)}")
    print(f"Capas de proyección: {len(layers)}\n")

    total_weights = 0
    for name, W in sorted(layers.items())[:20]:
        W_q, scale = quantize_absmean(W)
        flat = W_q.flatten().tolist()
        n = len(flat)
        zeros = sum(1 for w in flat if w == 0)
        pos   = sum(1 for w in flat if w == 1)
        neg   = sum(1 for w in flat if w == -1)
        error = (W.float() - W_q.float() * scale).norm().item() / W.float().norm().item()
        print(f"  {name:<50} {W.shape[0]}×{W.shape[1]:>5}  "
              f"scale={scale:.4f}  err={error:.3f}  "
              f"+={pos/n*100:.0f}% 0={zeros/n*100:.0f}% -={neg/n*100:.0f}%")
        total_weights += n

    orig_mb   = total_weights * 2 / 1024**2  # BF16
    packed_mb = total_weights * 2 / 8 / 1024**2  # 2 bits
    print(f"\nTotal pesos:    {total_weights/1e6:.1f}M")
    print(f"Tamaño BF16:    {orig_mb:.1f} MB")
    print(f"Tamaño Crystal: {packed_mb:.1f} MB")
    print(f"Compresión:     {orig_mb/packed_mb:.1f}x")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Empaqueta pesos BitNet b1.58 al formato .qomntal")
    parser.add_argument("--input",    required=True,
                        help="Ruta al modelo: .safetensors o directorio HF")
    parser.add_argument("--output",   default=None,
                        help="Ruta de salida .qomntal (default: <input>.qomntal)")
    parser.add_argument("--arch",     default="bitnet-500m",
                        help="Nombre de arquitectura (metadata)")
    parser.add_argument("--domain",   default="",
                        help="Dominio del cristal (metadata)")
    parser.add_argument("--layers",   default="all",
                        help="Filtro de capas: all, attn, ffn")
    parser.add_argument("--diagnose", action="store_true",
                        help="Solo mostrar estadísticas, no escribir")
    parser.add_argument("--no-verify", action="store_true",
                        help="Saltar verificación round-trip")
    args = parser.parse_args()

    if args.diagnose:
        diagnose_model(args.input)
        return

    # Determinar output path
    if args.output is None:
        base = Path(args.input).stem if Path(args.input).is_file() else Path(args.input).name
        args.output = f"{base}.qomntal"

    # Cargar modelo
    tensors, config = load_model(args.input)

    # Filtrar capas
    patterns = LAYER_PATTERNS
    if args.layers == "attn":
        patterns = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif args.layers == "ffn":
        patterns = ["gate_proj", "up_proj", "down_proj"]

    layers_dict = filter_layers(tensors, patterns)
    if not layers_dict:
        print(f"ERROR: ninguna capa encontrada con patrones {patterns}")
        print(f"Tensores disponibles (primeros 20):")
        for k in list(tensors.keys())[:20]:
            print(f"  {k}: {tensors[k].shape}")
        sys.exit(1)

    # Ordenar por nombre para reproducibilidad
    layers_sorted = sorted(layers_dict.items())

    # Detectar arquitectura
    arch = detect_arch(tensors, config)
    if args.arch != "bitnet-unknown":
        arch["name"] = args.arch
    if args.domain:
        arch["name"] = f"bitnet-{args.domain}"

    print(f"\nArquitectura detectada:")
    print(f"  hidden_dim:  {arch['hidden_dim']}")
    print(f"  ffn_dim:     {arch['ffn_dim']}")
    print(f"  n_heads:     {arch['n_heads']}")
    print(f"  n_kv_heads:  {arch['n_kv_heads']}")
    print(f"  n_layers:    {len(layers_sorted)}")

    # Empaquetar y escribir
    write_crystal(args.output, layers_sorted, arch, verbose=True)


if __name__ == "__main__":
    main()
