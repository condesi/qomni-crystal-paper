// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v0.3 — Abstract Syntax Tree
// ═══════════════════════════════════════════════════════════════════════

// ── Types ─────────────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    F32, F64, I32, I64, Bool, Str,
    Trit,
    Tvec(usize),              // tvec[n]
    Tmat(usize, usize),       // tmat[r][c]
    Tensor(Box<Type>, Vec<usize>),
    Inferred,                 // type to be resolved by typeck
}

// ── Hardware hints ─────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq)]
pub enum HwHint { Mmap, Avx2, Cpu, Auto }

#[derive(Debug, Clone, PartialEq)]
pub enum HwCond { Avx2Available, TernaryChip, GpuAvailable, Else }

// ── Expressions ───────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Trit(i8),
    Tvec(Vec<i8>),            // tvec[+1, 0t, -1, ...]

    // Variable / field access
    Ident(String),
    Field(Box<Expr>, String), // expr.field

    // Operations
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Index(Box<Expr>, Box<Expr>),             // expr[idx]
    Call(Box<Expr>, Vec<Expr>),              // expr(args)

    // Crystal-specific
    CrystalInfer {                           // crystal.infer(layer=N, x=expr)
        crystal: Box<Expr>,
        layer:   Option<usize>,
        x:       Box<Expr>,
    },
    CrystalLayer(Box<Expr>, usize),          // crystal.layer(N)
    CrystalNorm(Box<Expr>),                  // crystal.norm()

    // Built-ins
    Encode(Box<Expr>, Option<usize>),        // encode(expr, dim)
    Quantize(Box<Expr>),                     // quantize(expr)

    // Pipeline composition
    PipeComp(Vec<Expr>),                     // a | b | c
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp { Neg, Not }

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Mod, Pow,
    Eq, Ne, Lt, Gt, Le, Ge,
    And, Or,
    Assign,
}

// ── Statements ────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        ty:   Option<Type>,
        val:  Expr,
    },
    Expr(Expr),
    Return(Expr),
    If {
        cond:      Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
    },
    For {
        var:  String,
        iter: Expr,
        body: Vec<Stmt>,
    },
    Respond(Expr),
}

// ── Top-level declarations ─────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty:   Type,
}

#[derive(Debug, Clone)]
pub struct OracleDecl {
    pub name:    String,
    pub params:  Vec<Param>,
    pub ret_ty:  Type,
    pub body:    Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub struct CrystalDecl {
    pub name: String,
    pub hint: HwHint,
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct PipeDecl {
    pub name:   String,
    pub params: Vec<Param>,
    pub steps:  Vec<(String, Expr)>,  // name = expr
    pub sink:   Expr,                 // respond(...)
}

#[derive(Debug, Clone)]
pub enum RoutePattern {
    Exact(String),
    Glob(String),   // contains *
    Any,            // *
}

#[derive(Debug, Clone)]
pub enum RouteTarget {
    Crystal(String),
    Oracle(String),
    Pipe(String),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct RouteDecl {
    pub pattern: RoutePattern,
    pub target:  RouteTarget,
}

#[derive(Debug, Clone)]
pub struct ScheduleBranch {
    pub cond: HwCond,
    pub hint: HwHint,
}

#[derive(Debug, Clone)]
pub struct ScheduleDecl {
    pub expr:     Expr,
    pub branches: Vec<ScheduleBranch>,
}

#[derive(Debug, Clone)]
pub enum Decl {
    Oracle(OracleDecl),
    Crystal(CrystalDecl),
    Pipe(PipeDecl),
    Route(RouteDecl),
    Schedule(ScheduleDecl),
    Let(String, Option<Type>, Expr),
    Stmt(Stmt),
}

// ── Program ────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct Program {
    pub decls: Vec<Decl>,
}
