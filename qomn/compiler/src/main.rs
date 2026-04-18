// ═══════════════════════════════════════════════════════════════════════
// QOMN v1.0 — Main entry point
// Usage:
//   qomn                       start REPL
//   qomn run <file.qomn>       execute a program
//   qomn check <file.qomn>     type-check only
//   qomn lex <file.qomn>       dump tokens (debug)
// ═══════════════════════════════════════════════════════════════════════

mod lexer;
mod ast;
mod parser;
mod typeck;
mod vm;
mod repl;
mod server;
mod qomn_compiler;

use lexer::Lexer;
use parser::Parser;
use typeck::TypeEnv;
use vm::{Vm, QomniConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        None | Some("repl") => {
            repl::run_repl(None, None);
        }

        Some("run") => {
            let path = args.get(2).expect("Usage: qomn run <file.qomn>");
            let src  = std::fs::read_to_string(path)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1) });

            let prog = compile(&src);
            let config = QomniConfig::default();
            let mut vm = Vm::new(config);
            match vm.run(&prog) {
                Ok(out) => { for line in out { println!("{}", line); } }
                Err(e)  => { eprintln!("Runtime error: {}", e); std::process::exit(1); }
            }
        }

        Some("check") => {
            let path = args.get(2).expect("Usage: qomn check <file.qomn>");
            let src  = std::fs::read_to_string(path)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1) });

            let prog   = compile(&src);
            let mut tc = TypeEnv::new();
            let errors = tc.check_program(&prog);
            if errors.is_empty() {
                println!("OK — no type errors");
            } else {
                for e in &errors { eprintln!("Type error: {}", e); }
                std::process::exit(1);
            }
        }

        Some("lex") => {
            let path = args.get(2).expect("Usage: qomn lex <file.qomn>");
            let src  = std::fs::read_to_string(path)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1) });
            let mut lexer = Lexer::new(&src);
            for tok in lexer.tokenize() {
                println!("{:3}:{:2}  {:?}", tok.span.line, tok.span.col, tok.token);
            }
        }

        Some("eval") => {
            // eval inline expression: qomn eval "let x = 1.0 + 2.0"
            let src = args.get(2).expect("Usage: qomn eval <expr>");
            let prog = compile(src);
            let mut vm = Vm::new(QomniConfig::default());
            match vm.run(&prog) {
                Ok(out) => { for line in out { println!("{}", line); } }
                Err(e)  => eprintln!("Error: {}", e),
            }
        }

        Some("compile") => {
            // qomn compile <file.qomn> [out_dir]
            // Compiles all oracle declarations to .qomntal files
            let path    = args.get(2).expect("Usage: qomn compile <file.qomn> [out_dir]");
            let out_dir = args.get(3).map(|s| s.as_str()).unwrap_or(".");
            let src     = std::fs::read_to_string(path)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1) });
            let prog = compile(&src);

            println!("QOMN Compiler — oracle → .qomntal");
            println!("Input:   {}", path);
            println!("Out dir: {}", out_dir);
            println!();

            let results = qomn_compiler::compile_oracles(&prog, out_dir);
            if results.is_empty() {
                println!("No oracle declarations found in '{}'.", path);
                std::process::exit(0);
            }

            let mut ok = 0;
            for r in results {
                match r {
                    Ok(c) => {
                        let kb = c.file_size / 1024;
                        println!("  oracle {} → {} ({}KB, sparsity={:.1}%)",
                            c.oracle_name, c.out_path, kb, c.sparsity * 100.0);
                        ok += 1;
                    }
                    Err(e) => eprintln!("  ERROR: {}", e),
                }
            }
            println!();
            println!("{} oracle(s) compiled to .qomntal", ok);
        }

        Some("serve") => {
            // qomn serve <file.qomn> [port]
            let path = args.get(2).expect("Usage: qomn serve <file.qomn> [port]");
            let port: u16 = args.get(3).and_then(|p| p.parse().ok()).unwrap_or(9000);
            let src  = std::fs::read_to_string(path)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1) });
            let prog   = compile(&src);
            let config = vm::QomniConfig::default();
            let mut vm_inst = vm::Vm::new(config);
            let _ = vm_inst.run(&prog);
            println!("  QOMN serve: loading '{}'", path);
            let srv = server::CrysServer::new(vm_inst, prog, port);
            srv.run();
        }

        Some(cmd) => {
            eprintln!("Unknown command: '{}'. Use: repl | run | check | lex | eval | compile | serve", cmd);
            std::process::exit(1);
        }
    }
}

fn compile(src: &str) -> ast::Program {
    let mut lexer  = Lexer::new(src);
    let tokens     = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    parser.parse().unwrap_or_else(|e| {
        eprintln!("Parse error: {}", e);
        std::process::exit(1)
    })
}
