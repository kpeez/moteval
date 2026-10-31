//! The `moteval` command-line interface, ported from `moteval/cli.py`.
//!
//! One subcommand (`run`) and eight flags, parsed by hand from `std::env::args` (no
//! `clap`). Flags, messages, exit codes, and table, JSON, and CSV output must match the
//! Python CLI byte for byte. The Python console script calls this entry point, so one CLI
//! implementation exists.
//!
//! The results-cli-bindings package fills the body.

/// Run the CLI on `args` (without the program name). Returns the process exit code.
pub fn main(_args: &[String]) -> i32 {
    todo!("results-cli-bindings: port moteval.cli.main")
}
