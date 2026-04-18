# QOMN Language Guide
**QOMN Language — Guía Completa**
Qomni AI Lab · Condesi Perú · 2026

---

## ¿Qué es QOMN?

QOMN es el primer lenguaje de programación con **tipos ternarios nativos**.
Diseñado para orquestar modelos de IA comprimidos en formato `.qomntal`
(BitNet b1.58, pesos `{-1, 0, +1}`), calcular con oráculos físicos deterministas,
y enrutar consultas a través de pipelines multi-dominio.

**Funciona en cualquier sistema con CPU x86-64.** No requiere GPU.

---

## Instalación

```bash
# Clonar el repo
git clone https://github.com/condesi/qomni-crystal-paper
cd qomni-crystal-paper/qomn/compiler

# Compilar (requiere Rust 1.70+)
cargo build --release

# El binario queda en:
./target/release/qomn
```

---

## Comandos del CLI

```bash
qomn                          # Inicia el REPL interactivo
qomn run   programa.qomn      # Ejecuta un programa
qomn check programa.qomn      # Solo verifica tipos
qomn lex   programa.qomn      # Muestra tokens (debug)
qomn eval  "let x = 1+2"      # Evalúa una expresión
```

---

## El REPL

```
$ qomn

  ╔═══════════════════════════════════════════════════╗
  ║   QOMN v1.0  — QOMN Language REPL           ║
  ╚═══════════════════════════════════════════════════╝

crys> let x = 1.5 + 2.0
  → 3.5

crys> oracle igv(base: f32) -> f32:
  ...     return base * 0.18
  ...
  → oracle 'igv' registered

crys> igv(1000.0)
  → 180.0000

crys> :quit
```

Comandos especiales del REPL:

| Comando | Descripción |
|---------|-------------|
| `:help` | Muestra la ayuda |
| `:quit` / `:q` | Sale |
| `:crystals` | Lista crystals cargados |
| `:routes` | Muestra la tabla de rutas |
| `:load archivo.qomn` | Carga y ejecuta un archivo |
| `:query texto` | Enruta un texto por el router |

---

## Tipos

### Tipos escalares
```crys
let a: f32  = 3.14
let b: f64  = 2.718281828
let c: i32  = 42
let d: i64  = 1000000
let e: bool = true
let s: str  = "hola"
```

### Tipos ternarios (únicos en QOMN)
```crys
# Un solo trit
let t: trit = +1     # valores posibles: +1, 0t, -1

# Vector ternario de dimensión n
let v: tvec[8] = tvec[+1, 0t, -1, +1, -1, 0t, 0t, +1]

# Matriz ternaria r × c (= una capa de un crystal)
let W: tmat[896][4864] = quantize(mi_crystal.layer(0))
```

---

## Oráculos

Los oráculos son **funciones puras deterministas** — como ecuaciones
físicas o matemáticas. No tienen efectos secundarios.

```crys
# Manning (hidráulica)
oracle manning(n: f32, R: f32, S: f32) -> f32:
    return (1.0 / n) * R ^ (2.0 / 3.0) * S ^ 0.5

# IGV (contabilidad peruana)
oracle igv(base: f32) -> f32:
    return base * 0.18

# Cuota hipoteca (finanzas)
oracle hipoteca(P: f32, i: f32, n: i32) -> f32:
    return P * i / (1.0 - (1.0 + i) ^ (-1 * n))

# Darcy-Weisbach (pérdidas de carga)
oracle darcy(f: f32, L: f32, D: f32, V: f32) -> f32:
    return f * (L / D) * (V ^ 2.0 / 19.62)

# Usar un oráculo:
let Q = manning(0.013, 0.5, 0.001)   # Q = 0.487 m³/s
let igv_monto = igv(50000.0)          # = 9000.0
```

---

## Crystals

Un crystal es un modelo de IA comprimido (formato `.qomntal`, BitNet b1.58).

```crys
# Cargar con demand-paging (0ms swap)
crystal hidraulica   = load @mmap "hidraulica.qomntal"
crystal contabilidad = load @mmap "contabilidad.qomntal"
crystal legal_peru   = load @mmap "legal_peru.qomntal"
crystal nfpa         = load @mmap "nfpa_electrico.qomntal"
crystal general      = load @mmap "general.qomntal"

# Inferencia: extraer activaciones de una capa
let x = encode(42.5, 4864)                       # codifica valor → vector 4864-dim
let y = hidraulica.infer(layer=0, x=x)           # inferencia capa 0
let y2 = hidraulica.infer(x=x)                   # capa por defecto

# Extraer pesos de una capa como matriz ternaria
let W: tmat[896][4864] = quantize(hidraulica.layer(0))

# Norma del crystal (medida de magnitud)
let n = hidraulica.norm()
```

### Hardware hints

| Hint | Descripción |
|------|-------------|
| `@mmap` | Carga lazy via mmap, 0ms hot-swap (recomendado) |
| `@avx2` | Fuerza kernel AVX2 ternario (1.3× vs float32) |
| `@cpu` | Scalar fallback (sin instrucciones especiales) |
| `@auto` | El compilador decide en runtime |

---

## Pipelines

Un pipeline define un flujo de procesamiento con pasos nombrados.

```crys
# Pipeline hidráulico
pipe caudal(n: f32, R: f32, S: f32):
    Q    = manning(n, R, S)
    v    = encode(Q, 4864)
    result = hidraulica.infer(layer=0, x=v)
    -> respond(result)

# Pipeline contable
pipe factura(monto: f32):
    base = monto / 1.18
    imp  = monto - base
    v    = encode(imp, 4864)
    result = contabilidad.infer(x=v)
    -> respond(result)

# Pipeline con oráculo + crystal
pipe analisis_ci(Q_gpm: f32, P_psi: f32):
    potencia = Q_gpm * P_psi / 2772.0    # NFPA 20
    v        = encode(potencia, 4864)
    result   = nfpa.infer(x=v)
    -> respond(result)
```

---

## Routing

El sistema de routing dirige consultas al qomn/oráculo/pipe correcto.

```crys
# Patrones exactos (con prefijo)
route "IGV*"          -> crystal:contabilidad
route "Manning*"      -> pipe:caudal
route "NFPA*"         -> crystal:nfpa
route "rociador*"     -> crystal:nfpa
route "contrato*"     -> crystal:legal_peru

# Wildcard total (fallback)
route *               -> crystal:general

# Ruta a oráculo directamente
route "calcular IGV*" -> oracle:igv

# Composición pipe
route "análisis*"     -> pipe:analisis_ci
```

---

## Schedule (Hardware)

Declara la estrategia de ejecución según hardware disponible.

```crys
schedule hidraulica.infer(x=v):
    if avx2_available:  @avx2      # Intel/AMD moderno
    if ternary_chip:    @auto      # NÚCLEO Q-1 futuro
    else:               @cpu       # fallback universal
```

---

## Ejemplo Completo — Sistema Multi-Dominio

```crys
# archivo: sistema.qomn

# Oráculos físicos
oracle igv(base: f32) -> f32:
    return base * 0.18

oracle manning(n: f32, R: f32, S: f32) -> f32:
    return (1.0 / n) * R ^ (2.0 / 3.0) * S ^ 0.5

oracle bomba_ci(Q_gpm: f32, P_psi: f32) -> f32:
    return Q_gpm * P_psi / 2772.0

# Crystals
crystal hidraulica   = load @mmap "/opt/nexus/crystals/hidraulica.qomntal"
crystal contabilidad = load @mmap "/opt/nexus/crystals/contabilidad.qomntal"
crystal legal_peru   = load @mmap "/opt/nexus/crystals/legal_peru.qomntal"
crystal nfpa         = load @mmap "/opt/nexus/crystals/nfpa_electrico.qomntal"
crystal general      = load @mmap "/opt/nexus/crystals/general.qomntal"

# Pipelines
pipe consulta_hidro(n: f32, R: f32, S: f32):
    Q = manning(n, R, S)
    v = encode(Q, 4864)
    -> respond(hidraulica.infer(x=v))

pipe consulta_contable(monto: f32):
    impuesto = igv(monto)
    v = encode(impuesto, 4864)
    -> respond(contabilidad.infer(x=v))

# Router global
route "Manning*"      -> pipe:consulta_hidro
route "caudal*"       -> crystal:hidraulica
route "IGV*"          -> pipe:consulta_contable
route "factura*"      -> crystal:contabilidad
route "NFPA*"         -> crystal:nfpa
route "rociador*"     -> crystal:nfpa
route "detector*"     -> crystal:nfpa
route "contrato*"     -> crystal:legal_peru
route "despido*"      -> crystal:legal_peru
route *               -> crystal:general

# Hardware
schedule *.infer(x=v):
    if avx2_available:  @avx2
    else:               @cpu
```

Ejecutar:
```bash
qomn run sistema.qomn
# crystal 'hidraulica' registered from '/opt/nexus/crystals/hidraulica.qomntal'
# crystal 'contabilidad' registered ...
# ...
```

---

## Gramática EBNF Resumida

```ebnf
program     ::= { decl } EOF
decl        ::= oracle_decl | crystal_decl | pipe_decl
              | route_decl | schedule_decl | let_stmt

oracle_decl ::= "oracle" IDENT "(" params ")" "->" type ":" block
crystal_decl::= "crystal" IDENT "=" "load" hw_hint STRING
pipe_decl   ::= "pipe" IDENT "(" params ")" ":" INDENT steps "->" "respond" "(" expr ")" DEDENT
route_decl  ::= "route" pattern "->" target
schedule_decl::= "schedule" expr ":" INDENT { "if" hw_cond ":" hw_hint } DEDENT

type        ::= f32 | f64 | i32 | i64 | bool | str
              | trit | tvec[n] | tmat[r][c]
hw_hint     ::= @mmap | @avx2 | @cpu | @auto
hw_cond     ::= avx2_available | ternary_chip | gpu_available | else
```

Ver spec completa: [`SPEC.md`](SPEC.md)

---

## Integración con Qomni

QOMN se conecta al servidor Qomni via HTTP:

```bash
# Variables de entorno
export QOMNI_URL="http://qomni.clanmarketer.com:8090"
export QOMNI_KEY="tu-api-key"

qomn run programa.qomn
```

Endpoints usados:
- `POST /qomni/qomn/register` — registrar crystal
- `POST /qomni/qomn/activate` — activar (0ms swap)
- `POST /qomni/qomn/infer` — inferencia por capa

---

## Roadmap

| Versión | Feature | Estado |
|---------|---------|--------|
| v0.1 | EBNF spec + ejemplos | ✓ |
| v0.2 | Lexer (tokenizer) | ✓ |
| v0.3 | Parser → AST | ✓ |
| v0.4 | Type checker ternario | ✓ |
| v0.5 | VM tree-walking | ✓ |
| v1.0 | REPL + CLI completo | ✓ |
| v1.1 | Compilador oracle → .qomntal | Planeado |
| v1.2 | Bytecode IR + optimizer | Planeado |
| v2.0 | Backend NÚCLEO Q-1 | Research |

---

*QOMN — Apache 2.0 — Qomni AI Lab, Condesi Perú*
*https://github.com/condesi/qomni-crystal-paper*
