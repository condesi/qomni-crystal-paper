# QOMN Language Specification v0.1
**QOMN Language — Formal EBNF Grammar**
Percy Rojas Masgo · Condesi Perú · Qomni AI Lab · 2026

---

## Abstract

QOMN (QOMN Language) is a domain-specific language designed for
ternary-native AI inference pipelines. It is the first language to treat
ternary types `{-1, 0, +1}` as first-class primitives, enabling direct
expression of BitNet b1.58 weight matrices without emulation overhead.
QOMN compiles to Qomni bytecode and targets the Crystal kernel AVX2
executor on commodity hardware.

---

## 1. Design Principles

1. **Ternary-native**: `trit`, `tvec[n]`, `tmat[r][c]` are primitive types
2. **Oracle-first**: physical equations are language constructs, not libraries
3. **Hardware-aware**: scheduling hints are declarative, not imperative
4. **Crystal-centric**: `.qomntal` files are first-class values
5. **No magic**: every construct maps to a concrete Qomni operation

---

## 2. Formal EBNF Grammar

```ebnf
(* ═══════════════════════════════════════════════════════════════
   QOMN v0.1 — Complete EBNF
   Notation: ::= definition | alternative, [] optional, {} repeat
   ═══════════════════════════════════════════════════════════════ *)

(* ─── TOP LEVEL ──────────────────────────────────────────────── *)
program         ::= { statement } EOF

statement       ::= oracle_decl
                  | crystal_decl
                  | pipe_decl
                  | route_decl
                  | schedule_decl
                  | let_stmt
                  | expr_stmt

(* ─── ORACLE DECLARATIONS ────────────────────────────────────── *)
(* Oracles are pure deterministic functions — Physics-as-Oracle   *)
oracle_decl     ::= "oracle" IDENT "(" [ param_list ] ")" "->" type_expr ":"
                    NEWLINE INDENT body DEDENT

param_list      ::= param { "," param }
param           ::= IDENT ":" type_expr

(* ─── CRYSTAL DECLARATIONS ───────────────────────────────────── *)
crystal_decl    ::= "crystal" IDENT "=" "load" hw_hint STRING

hw_hint         ::= "@mmap"    (* lazy demand-paging, 0ms swap  *)
                  | "@avx2"    (* force AVX2 ternary kernel      *)
                  | "@cpu"     (* scalar fallback                *)
                  | "@auto"    (* compiler chooses at runtime    *)

(* ─── PIPELINE DECLARATIONS ──────────────────────────────────── *)
pipe_decl       ::= "pipe" IDENT "(" [ param_list ] ")" ":"
                    NEWLINE INDENT pipe_body DEDENT

pipe_body       ::= { pipe_step } pipe_sink
pipe_step       ::= IDENT "=" pipe_expr NEWLINE
pipe_sink       ::= "->" "respond" "(" expr ")" NEWLINE

pipe_expr       ::= expr { "|" expr }

(* ─── ROUTE DECLARATIONS ─────────────────────────────────────── *)
route_decl      ::= "route" route_pattern "->" route_target NEWLINE

route_pattern   ::= STRING | GLOB | "*"
route_target    ::= "crystal" ":" IDENT
                  | "oracle"  ":" IDENT
                  | "pipe"    ":" IDENT
                  | pipe_expr

(* ─── SCHEDULE DECLARATIONS ──────────────────────────────────── *)
(* Compiler selects optimal hardware path at runtime              *)
schedule_decl   ::= "schedule" expr ":" NEWLINE
                    INDENT { schedule_branch } DEDENT

schedule_branch ::= "if" hw_condition ":" hw_hint NEWLINE

hw_condition    ::= "avx2_available"
                  | "ternary_chip"
                  | "gpu_available"
                  | "else"

(* ─── TYPE SYSTEM ────────────────────────────────────────────── *)
type_expr       ::= scalar_type
                  | ternary_type
                  | tensor_type
                  | "(" type_expr ")"

scalar_type     ::= "f32" | "f64" | "i32" | "i64" | "bool" | "str"

ternary_type    ::= "trit"                      (* single trit {-1,0,+1}   *)
                  | "tvec" "[" INT "]"           (* ternary vector dim n    *)
                  | "tmat" "[" INT "]" "[" INT "]" (* ternary matrix r×c   *)

tensor_type     ::= "tensor" "[" type_expr "," INT { "," INT } "]"

(* ─── EXPRESSIONS ────────────────────────────────────────────── *)
expr            ::= assign_expr

assign_expr     ::= IDENT "=" expr
                  | logical_expr

logical_expr    ::= cmp_expr { ( "and" | "or" ) cmp_expr }

cmp_expr        ::= arith_expr { ( "==" | "!=" | "<" | ">" | "<=" | ">=" ) arith_expr }

arith_expr      ::= term { ( "+" | "-" ) term }

term            ::= power { ( "*" | "/" | "%" ) power }

power           ::= unary { "^" unary }

unary           ::= "-" unary
                  | "not" unary
                  | call_expr

call_expr       ::= primary { call_suffix }
call_suffix     ::= "(" [ arg_list ] ")"
                  | "." IDENT
                  | "[" expr "]"
                  | ".infer" "(" infer_args ")"

primary         ::= IDENT
                  | literal
                  | crystal_ref
                  | encode_expr
                  | quantize_expr
                  | "(" expr ")"

(* Crystal operations *)
crystal_ref     ::= IDENT ".infer" "(" infer_args ")"
                  | IDENT ".layer" "(" INT ")"
                  | IDENT ".norm" "(" ")"

infer_args      ::= "layer" "=" INT "," "x" "=" expr
                  | "x" "=" expr

(* Encoding: f32 vector -> suitable input for crystal inference *)
encode_expr     ::= "encode" "(" expr [ "," INT ] ")"

(* Quantize: f32 -> trit using absmean quantization (BitNet) *)
quantize_expr   ::= "quantize" "(" expr ")"

arg_list        ::= expr { "," expr }

(* ─── LITERALS ───────────────────────────────────────────────── *)
literal         ::= INT
                  | FLOAT
                  | STRING
                  | BOOL
                  | trit_literal
                  | tvec_literal
                  | tmat_literal

trit_literal    ::= "+1" | "0t" | "-1"
tvec_literal    ::= "tvec" "[" trit_literal { "," trit_literal } "]"
tmat_literal    ::= "tmat" "[" { tvec_literal } "]"

(* ─── STATEMENTS ─────────────────────────────────────────────── *)
let_stmt        ::= "let" IDENT [ ":" type_expr ] "=" expr NEWLINE
expr_stmt       ::= expr NEWLINE

body            ::= { let_stmt | expr_stmt | if_stmt | for_stmt | return_stmt }

if_stmt         ::= "if" expr ":" NEWLINE INDENT body DEDENT
                    [ "else" ":" NEWLINE INDENT body DEDENT ]

for_stmt        ::= "for" IDENT "in" expr ":" NEWLINE INDENT body DEDENT

return_stmt     ::= "return" expr NEWLINE

(* ─── LEXICAL RULES ──────────────────────────────────────────── *)
IDENT           ::= [a-zA-Z_] [a-zA-Z0-9_]*
INT             ::= [0-9]+
FLOAT           ::= [0-9]+ "." [0-9]+
STRING          ::= '"' { char } '"'
GLOB            ::= '"' { char } "*" { char } '"'
BOOL            ::= "true" | "false"
COMMENT         ::= "#" { any_char } NEWLINE
WHITESPACE      ::= { " " | "\t" }    (* ignored between tokens *)
NEWLINE         ::= "\n" | "\r\n"
INDENT          ::= increase in indentation level (4 spaces or 1 tab)
DEDENT          ::= decrease in indentation level

KEYWORDS        ::= "oracle" | "crystal" | "pipe" | "route" | "schedule"
                  | "load" | "encode" | "quantize" | "respond"
                  | "let" | "if" | "else" | "for" | "in" | "return"
                  | "and" | "or" | "not" | "true" | "false"
                  | "f32" | "f64" | "i32" | "i64" | "bool" | "str"
                  | "trit" | "tvec" | "tmat" | "tensor"
                  | "@mmap" | "@avx2" | "@cpu" | "@auto"
                  | "avx2_available" | "ternary_chip" | "gpu_available"
```

---

## 3. Example Programs

### 3.1 Oracle físico — Manning
```crys
oracle manning(n: f32, R: f32, S: f32) -> f32:
    return (1.0 / n) * R ^ (2.0 / 3.0) * S ^ 0.5

oracle darcy_weisbach(f: f32, L: f32, D: f32, V: f32) -> f32:
    return f * (L / D) * (V ^ 2.0 / 19.62)
```

### 3.2 Crystal loading con hardware hint
```crys
crystal hidraulica   = load @mmap "hidraulica.qomntal"
crystal contabilidad = load @mmap "contabilidad.qomntal"
crystal legal_peru   = load @mmap "legal_peru.qomntal"
crystal nfpa         = load @mmap "nfpa_electrico.qomntal"
```

### 3.3 Pipeline multi-crystal
```crys
pipe consulta_hidraulica(n: f32, R: f32, S: f32):
    q = manning(n, R, S)
    v = encode(q, 4864)
    resultado = hidraulica.infer(layer=0, x=v)
    -> respond(resultado)

pipe consulta_contable(monto: f32):
    igv = monto * 0.18
    base = monto - igv
    v = encode(base, 4864)
    resultado = contabilidad.infer(layer=0, x=v)
    -> respond(resultado)
```

### 3.4 Routing automático
```crys
route "Manning*"     -> pipe:consulta_hidraulica
route "IGV*"         -> pipe:consulta_contable
route "NFPA*"        -> crystal:nfpa
route "contrato*"    -> crystal:legal_peru
route *              -> crystal:general
```

### 3.5 Schedule hardware-aware
```crys
schedule hidraulica.infer(layer=0, x=v):
    if avx2_available:  @avx2
    if ternary_chip:    @auto
    else:               @cpu
```

### 3.6 Tipos ternarios nativos
```crys
# Cuantizar pesos float a ternario (BitNet absmean)
let W: tmat[896][4864] = quantize(hidraulica.layer(0))

# Vector ternario literal
let v: tvec[8] = tvec[+1, 0, -1, +1, -1, 0, 0, +1]

# Producto matriz-vector ternario
let y: tvec[896] = W * v
```

---

## 4. Compilation Targets

| Target | Description | Status |
|--------|-------------|--------|
| `qomni-bytecode` | Qomni internal IR, executes on Server5 | Planned v1 |
| `rust-ffi` | Generates Rust bindings for crystal_kernel.rs | Planned v1 |
| `crystal-pack` | Compiles oracle → `.qomntal` binary format | Planned v2 |
| `ternary-asm` | Native ternary assembly (future NÚCLEO Q-1) | Research |

---

## 5. Novelty Statement

QOMN introduces three contributions not present in existing languages:

1. **Ternary primitive types** (`trit`, `tvec`, `tmat`) — no existing general-purpose
   language includes ternary as a native type. Python, Rust, C++ emulate it in binary.

2. **Oracle declarations** — physics equations as pure, auto-vectorizable
   language constructs separate from functions. Enables Physics-as-Oracle (PaO)
   pattern at the language level.

3. **Crystal references** — `.qomntal` binary files (BitNet b1.58 compressed
   weights) are first-class values with type-safe load, infer, and layer operations.

---

## 6. Implementation Roadmap

```
v0.1  EBNF spec + examples (this document)
v0.2  Lexer in Rust (tokenizer)
v0.3  Parser → AST
v0.4  Type checker (ternary compatibility rules)
v0.5  Bytecode emitter → Qomni executor
v1.0  Full compiler + REPL
v1.1  crystal-pack: oracle → .qomntal compilation
```

---

## 7. References

- Qomni Crystal Format Specification (this repo, `arxiv/main.tex`)
- BitNet b1.58: Ma et al., 2024 — "The Era of 1-bit LLMs"
- Physics-as-Oracle: Rojas Masgo, 2026 — Qomni AI Lab
- MLIR: Lattner et al., 2020 — Multi-Level Intermediate Representation
- Futhark: Henriksen et al., 2017 — Purely Functional GPU Language

---

*QOMN is open source — Apache 2.0 — Qomni AI Lab, Condesi Perú*
