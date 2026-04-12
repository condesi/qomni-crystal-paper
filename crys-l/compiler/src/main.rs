// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v1.0 — Main entry point
// Usage:
//   crysl                       start REPL
//   crysl run <file.crys>       execute a program
//   crysl check <file.crys>     type-check only
//   crysl lex <file.crys>       dump tokens (debug)
// ═══════════════════════════════════════════════════════════════════════

mod lexer;
mod ast;
mod parser;
mod typeck;
mod vm;
mod repl;
mod server;

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
            let path = args.get(2).expect("Usage: crysl run <file.crys>");
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
            let path = args.get(2).expect("Usage: crysl check <file.crys>");
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
            let path = args.get(2).expect("Usage: crysl lex <file.crys>");
            let src  = std::fs::read_to_string(path)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1) });
            let mut lexer = Lexer::new(&src);
            for tok in lexer.tokenize() {
                println!("{:3}:{:2}  {:?}", tok.span.line, tok.span.col, tok.token);
            }
        }

        Some("eval") => {
            // eval inline expression: crysl eval "let x = 1.0 + 2.0"
            let src = args.get(2).expect("Usage: crysl eval <expr>");
            let prog = compile(src);
            let mut vm = Vm::new(QomniConfig::default());
            match vm.run(&prog) {
                Ok(out) => { for line in out { println!("{}", line); } }
                Err(e)  => eprintln!("Error: {}", e),
            }
        }

        Some("serve") => {
            // crysl serve <file.crys> [port]
            let path = args.get(2).expect("Usage: crysl serve <file.crys> [port]");
            let port: u16 = args.get(3).and_then(|p| p.parse().ok()).unwrap_or(9000);
            let src  = std::fs::read_to_string(path)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1) });
            let prog   = compile(&src);
            let config = vm::QomniConfig::default();
            let mut vm_inst = vm::Vm::new(config);
            let _ = vm_inst.run(&prog);
            println!("  CRYS-L serve: loading '{}'", path);
            let srv = server::CrysServer::new(vm_inst, prog, port);
            srv.run();
        }

        Some(cmd) => {
            eprintln!("Unknown command: '{}'. Use: repl | run | check | lex | eval | serve", cmd);
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
