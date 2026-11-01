#![forbid(unsafe_code)]

fn main() {
    // A non-UTF-8 argument is read lossily rather than panicking.
    let args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    std::process::exit(moteval::cli::main(&args));
}
