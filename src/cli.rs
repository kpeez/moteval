//! The `moteval` command-line interface, ported from `moteval/cli.py`.
//!
//! One subcommand (`run`) and eight flags, parsed by hand (no `clap`) with the rules of
//! CPython 3.14's `argparse` for this parser: unique-prefix abbreviations, `--opt=value`,
//! `--` handling, unknown options collected and reported as unrecognized, and the same
//! error texts, usage lines, and exit codes. The Python console script calls this entry
//! point, so one CLI implementation exists.
//!
//! Table, JSON, and CSV output equal the Python CLI's byte for byte
//! (`tests/integration/exports.rs`, `tests/integration/cli.rs`). Known differences, all
//! outside those outputs:
//! - Usage and help text are argparse's layout for an 80-column terminal without color.
//!   Python wraps them to the terminal width and colors them on a color terminal.
//! - Where Python dies with a traceback (an `OSError` while reading or writing a file, an
//!   infinite count in the table), this prints `moteval: error: <message>` and exits 1, as
//!   Python's uncaught exception does.

use std::fmt;
use std::io::Write;
use std::path::Path;

use crate::benchmarks::{self, BenchmarkError};
use crate::eval::{EvalError, evaluate, is_value_error};
use crate::metrics::{Clear, Count, Hota, Identity, JAndF, Metric, TrackMap};
use crate::model::MOTDataset;
use crate::pyfmt::{self, repr_str};
use crate::results::{self, EvaluationResult};

const TOP_PROG: &str = "moteval";
const RUN_PROG: &str = "moteval run";

// argparse's usage and help text at its default width (an 80-column terminal).
const TOP_USAGE: &str = "usage: moteval [-h] {run} ...\n";
const TOP_HELP: &str = concat!(
    "usage: moteval [-h] {run} ...\n",
    "\n",
    "positional arguments:\n",
    "  {run}\n",
    "    run       evaluate prediction files\n",
    "\n",
    "options:\n",
    "  -h, --help  show this help message and exit\n",
);
const RUN_USAGE: &str = concat!(
    "usage: moteval run [-h] [--dataset DATASET] [--split SPLIT] [--gt GT]\n",
    "                   [--format {mot,mots}] --pred PRED [--metrics METRICS]\n",
    "                   [--out-csv OUT_CSV] [--out-json OUT_JSON]\n",
);
const RUN_HELP: &str = concat!(
    "usage: moteval run [-h] [--dataset DATASET] [--split SPLIT] [--gt GT]\n",
    "                   [--format {mot,mots}] --pred PRED [--metrics METRICS]\n",
    "                   [--out-csv OUT_CSV] [--out-json OUT_JSON]\n",
    "\n",
    "options:\n",
    "  -h, --help           show this help message and exit\n",
    "  --dataset DATASET    built-in benchmark name (omit to load --gt directly)\n",
    "  --split SPLIT        dataset split (uses the loader default when omitted)\n",
    "  --gt GT              ground-truth dataset root\n",
    "  --format {mot,mots}  ground-truth layout when --dataset is omitted (default:\n",
    "                       mot)\n",
    "  --pred PRED          prediction directory\n",
    "  --metrics METRICS    comma-separated metrics\n",
    "  --out-csv OUT_CSV    write long-form CSV results\n",
    "  --out-json OUT_JSON  write JSON results\n",
);

const DEFAULT_METRICS: &str = "hota,clear,identity,count";
/// `_METRICS`, in Python's dict order (the order of the "available" list).
const METRIC_NAMES: &[&str] = &["hota", "clear", "identity", "count", "track_map", "jf"];
const FORMAT_CHOICES: &[&str] = &["mot", "mots"];

/// `_HEADLINES`: (metric, field, column label), in table order.
const HEADLINES: &[(&str, &str, &str)] = &[
    ("HOTA", "HOTA", "HOTA"),
    ("HOTA", "DetA", "DetA"),
    ("HOTA", "AssA", "AssA"),
    ("CLEAR", "MOTA", "MOTA"),
    ("CLEAR", "MOTP", "MOTP"),
    ("CLEAR", "IDSW", "IDSW"),
    ("Identity", "IDF1", "IDF1"),
    ("Count", "Dets", "Dets"),
    ("Count", "GT_Dets", "GT_Dets"),
];
const INTEGER_HEADLINES: &[&str] = &["IDSW", "Dets", "GT_Dets"];

/// Run the CLI on `args` (without the program name), writing to the process's stdout and
/// stderr. Returns the process exit code.
pub fn main(args: &[String]) -> i32 {
    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();
    run(args, &mut stdout, &mut stderr)
}

/// Run the CLI on `args` (without the program name), writing to `stdout` and `stderr`.
/// Returns the process exit code: 0 on success, 2 on a usage or `ValueError`-class error,
/// 1 where Python dies with an uncaught exception.
pub fn run(args: &[String], stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    let exit = match parse(args) {
        Ok(Parsed::Help(text)) => {
            let _ = stdout.write_all(text.as_bytes());
            0
        }
        Ok(Parsed::Run(run_args)) => match run_command(&run_args, stdout) {
            Ok(()) => 0,
            Err(Failure::Usage(message)) => usage_error(stderr, TOP_USAGE, TOP_PROG, &message),
            Err(Failure::Uncaught(message)) => {
                let _ = writeln!(stderr, "{TOP_PROG}: error: {message}");
                1
            }
        },
        Err(err) => usage_error(stderr, err.usage, err.prog, &err.message),
    };
    let _ = stdout.flush();
    let _ = stderr.flush();
    exit
}

/// `ArgumentParser.error`: the parser's usage, then `{prog}: error: {message}`; exit 2.
fn usage_error(stderr: &mut dyn Write, usage: &str, prog: &str, message: &str) -> i32 {
    let _ = writeln!(stderr, "{usage}{prog}: error: {message}");
    2
}

// --------------------------------------------------------------------- argument parsing

/// An argparse action of this CLI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Help,
    Command,
    Dataset,
    Split,
    Gt,
    Format,
    Pred,
    Metrics,
    OutCsv,
    OutJson,
}

impl Action {
    /// `_get_action_name`: how errors name the argument.
    fn name(self) -> &'static str {
        match self {
            Self::Help => "-h/--help",
            Self::Command => "command",
            Self::Dataset => "--dataset",
            Self::Split => "--split",
            Self::Gt => "--gt",
            Self::Format => "--format",
            Self::Pred => "--pred",
            Self::Metrics => "--metrics",
            Self::OutCsv => "--out-csv",
            Self::OutJson => "--out-json",
        }
    }

    /// How many arguments the action takes: 0 for help, 1 for every stored option.
    fn takes_value(self) -> bool {
        !matches!(self, Self::Help)
    }
}

/// One of the two parsers: `moteval` and its `run` subparser.
struct Parser {
    prog: &'static str,
    usage: &'static str,
    /// `_option_string_actions`, in insertion order.
    options: &'static [(&'static str, Action)],
    /// `_actions`, in insertion order (the order of the required-arguments message).
    actions: &'static [Action],
    /// The positional actions: `command` (nargs `PARSER`) for the top-level parser.
    positionals: &'static [Action],
    required: &'static [Action],
}

const TOP: Parser = Parser {
    prog: TOP_PROG,
    usage: TOP_USAGE,
    options: &[("-h", Action::Help), ("--help", Action::Help)],
    actions: &[Action::Help, Action::Command],
    positionals: &[Action::Command],
    required: &[Action::Command],
};

const RUN: Parser = Parser {
    prog: RUN_PROG,
    usage: RUN_USAGE,
    options: &[
        ("-h", Action::Help),
        ("--help", Action::Help),
        ("--dataset", Action::Dataset),
        ("--split", Action::Split),
        ("--gt", Action::Gt),
        ("--format", Action::Format),
        ("--pred", Action::Pred),
        ("--metrics", Action::Metrics),
        ("--out-csv", Action::OutCsv),
        ("--out-json", Action::OutJson),
    ],
    actions: &[
        Action::Help,
        Action::Dataset,
        Action::Split,
        Action::Gt,
        Action::Format,
        Action::Pred,
        Action::Metrics,
        Action::OutCsv,
        Action::OutJson,
    ],
    positionals: &[],
    required: &[Action::Pred],
};

/// The `run` subcommand's namespace. Paths are already `str(Path(value))`.
#[derive(Clone, Debug)]
struct RunArgs {
    dataset: Option<String>,
    split: Option<String>,
    gt: Option<String>,
    format: String,
    pred: String,
    metrics: String,
    out_csv: Option<String>,
    out_json: Option<String>,
}

/// Each action taken, in order, with its argument strings.
type Taken = Vec<(Action, Vec<String>)>;

enum Parsed {
    /// `-h`/`--help`: print this text to stdout and exit 0.
    Help(&'static str),
    Run(RunArgs),
}

/// An argparse error, reported by the parser that raised it.
#[derive(Debug)]
struct ParseError {
    prog: &'static str,
    usage: &'static str,
    message: String,
}

/// Why a parse stopped early: help printed, or an error.
enum Stop {
    Help(&'static str),
    Error(ParseError),
}

impl From<ParseError> for Stop {
    fn from(err: ParseError) -> Self {
        Self::Error(err)
    }
}

/// `ArgumentError(action, message)` as text.
fn argument_error(action: Option<Action>, message: impl fmt::Display) -> String {
    match action {
        Some(action) => format!("argument {}: {message}", action.name()),
        None => message.to_string(),
    }
}

/// One interpretation of an option string: `(action, option_string, sep, explicit_arg)`.
#[derive(Clone, Debug)]
struct OptionTuple {
    action: Option<Action>,
    option_string: String,
    sep: Option<String>,
    explicit_arg: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pattern {
    /// An argument (`A`).
    Arg,
    /// An option (`O`).
    Opt,
    /// `--` (`-`).
    Dash,
}

/// `_negative_number_matcher`: `^-\d+$|^-\d*\.\d+$`, where `$` also matches before a
/// final newline. Only ASCII digits count here; Python's `\d` also takes other Unicode
/// decimal digits.
fn looks_like_negative_number(arg: &str) -> bool {
    let Some(rest) = arg.strip_prefix('-') else {
        return false;
    };
    let rest = rest.strip_suffix('\n').unwrap_or(rest);
    let digits = |s: &str| s.chars().all(|c| c.is_ascii_digit());
    match rest.split_once('.') {
        None => !rest.is_empty() && digits(rest),
        Some((whole, frac)) => digits(whole) && !frac.is_empty() && digits(frac),
    }
}

/// `str.partition('=')`: `(before, sep, after)`, with `sep` empty when there is no `=`.
fn partition_eq(s: &str) -> (&str, &str, &str) {
    match s.split_once('=') {
        Some((before, after)) => (before, "=", after),
        None => (s, "", ""),
    }
}

impl Parser {
    fn lookup(&self, option_string: &str) -> Option<Action> {
        self.options
            .iter()
            .find(|(name, _)| *name == option_string)
            .map(|(_, action)| *action)
    }

    fn error(&self, message: String) -> ParseError {
        ParseError {
            prog: self.prog,
            usage: self.usage,
            message,
        }
    }

    /// `_get_option_tuples`.
    fn option_tuples(&self, arg: &str) -> Vec<OptionTuple> {
        let mut result = Vec::new();
        let mut chars = arg.chars();
        let (first, second) = (chars.next(), chars.next());
        if first == Some('-') && second == Some('-') {
            let (prefix, sep, explicit) = partition_eq(arg);
            let (sep, explicit) = if sep.is_empty() {
                (None, None)
            } else {
                (Some(sep.to_string()), Some(explicit.to_string()))
            };
            for (option_string, action) in self.options {
                if option_string.starts_with(prefix) {
                    result.push(OptionTuple {
                        action: Some(*action),
                        option_string: option_string.to_string(),
                        sep: sep.clone(),
                        explicit_arg: explicit.clone(),
                    });
                }
            }
        } else {
            // A single-dash option: `-x` takes the rest of the string as its argument.
            let (prefix, sep, explicit) = partition_eq(arg);
            let (sep, explicit) = if sep.is_empty() {
                (None, None)
            } else {
                (Some(sep.to_string()), Some(explicit.to_string()))
            };
            let split = arg.char_indices().nth(2).map_or(arg.len(), |(i, _)| i);
            let (short_prefix, short_explicit) = arg.split_at(split);
            for (option_string, action) in self.options {
                if *option_string == short_prefix {
                    result.push(OptionTuple {
                        action: Some(*action),
                        option_string: option_string.to_string(),
                        sep: Some(String::new()),
                        explicit_arg: Some(short_explicit.to_string()),
                    });
                } else if option_string.starts_with(prefix) {
                    result.push(OptionTuple {
                        action: Some(*action),
                        option_string: option_string.to_string(),
                        sep: sep.clone(),
                        explicit_arg: explicit.clone(),
                    });
                }
            }
        }
        result
    }

    /// `_parse_optional`: `None` for a positional, else the option's interpretations.
    fn parse_optional(&self, arg: &str) -> Option<Vec<OptionTuple>> {
        if !arg.starts_with('-') {
            return None;
        }
        if let Some(action) = self.lookup(arg) {
            return Some(vec![OptionTuple {
                action: Some(action),
                option_string: arg.to_string(),
                sep: None,
                explicit_arg: None,
            }]);
        }
        if arg.chars().count() == 1 {
            return None;
        }
        let (option_string, sep, explicit) = partition_eq(arg);
        if !sep.is_empty()
            && let Some(action) = self.lookup(option_string)
        {
            return Some(vec![OptionTuple {
                action: Some(action),
                option_string: option_string.to_string(),
                sep: Some(sep.to_string()),
                explicit_arg: Some(explicit.to_string()),
            }]);
        }
        let tuples = self.option_tuples(arg);
        if !tuples.is_empty() {
            return Some(tuples);
        }
        // No option of this parser looks like a negative number.
        if looks_like_negative_number(arg) || arg.contains(' ') {
            return None;
        }
        Some(vec![OptionTuple {
            action: None,
            option_string: arg.to_string(),
            sep: None,
            explicit_arg: None,
        }])
    }

    /// `parse_known_args`: the namespace values in order of action, and the extras.
    fn parse_known(&self, args: &[String]) -> Result<(Taken, Vec<String>), Stop> {
        let mut patterns = Vec::with_capacity(args.len());
        let mut option_indices: Vec<Option<Vec<OptionTuple>>> = vec![None; args.len()];
        let mut dash_seen = false;
        for (i, arg) in args.iter().enumerate() {
            if dash_seen {
                patterns.push(Pattern::Arg);
            } else if arg == "--" {
                patterns.push(Pattern::Dash);
                dash_seen = true;
            } else if let Some(tuples) = self.parse_optional(arg) {
                option_indices[i] = Some(tuples);
                patterns.push(Pattern::Opt);
            } else {
                patterns.push(Pattern::Arg);
            }
        }

        let mut state = ParseState {
            parser: self,
            args,
            patterns: &patterns,
            seen: Vec::new(),
            taken: Vec::new(),
            extras: Vec::new(),
            positionals: self.positionals.to_vec(),
        };
        let max_option_index = option_indices.iter().rposition(Option::is_some);
        let mut start = 0;
        if let Some(max_option_index) = max_option_index {
            while start <= max_option_index {
                let mut next = start;
                while next <= max_option_index && option_indices[next].is_none() {
                    next += 1;
                }
                if start != next {
                    let end = state.consume_positionals(start)?;
                    if end > start {
                        start = end;
                        continue;
                    }
                    start = end;
                }
                if option_indices[start].is_none() {
                    state.extras.extend_from_slice(&args[start..next]);
                    start = next;
                }
                let tuples = option_indices[start].as_ref().expect("an option index");
                start = state.consume_optional(start, tuples)?;
            }
        }
        let stop = state.consume_positionals(start)?;
        state.extras.extend_from_slice(&args[stop..]);

        let missing: Vec<&str> = self
            .actions
            .iter()
            .filter(|action| self.required.contains(action) && !state.seen.contains(action))
            .map(|action| action.name())
            .collect();
        if !missing.is_empty() {
            return Err(self
                .error(format!(
                    "the following arguments are required: {}",
                    missing.join(", ")
                ))
                .into());
        }
        Ok((state.taken, state.extras))
    }
}

struct ParseState<'a> {
    parser: &'a Parser,
    args: &'a [String],
    patterns: &'a [Pattern],
    seen: Vec<Action>,
    taken: Taken,
    extras: Vec<String>,
    positionals: Vec<Action>,
}

impl ParseState<'_> {
    /// `consume_positionals`: the `command` positional (nargs `PARSER`, pattern
    /// `-*A[-AO]*`) takes every remaining string when the next non-`--` one is an
    /// argument; a leading `--` is dropped.
    fn consume_positionals(&mut self, start: usize) -> Result<usize, Stop> {
        let Some(&action) = self.positionals.first() else {
            return Ok(start);
        };
        let rest = &self.patterns[start..];
        let dashes = rest.iter().take_while(|p| **p == Pattern::Dash).count();
        if rest.get(dashes) != Some(&Pattern::Arg) {
            return Ok(start);
        }
        let mut strings: Vec<String> = self.args[start..].to_vec();
        if self.patterns[start] == Pattern::Dash {
            let first = strings.iter().position(|s| s == "--").expect("a '--'");
            strings.remove(first);
        }
        self.positionals.clear();
        self.take_action(action, strings)?;
        Ok(self.args.len())
    }

    /// `consume_optional`.
    fn consume_optional(&mut self, start: usize, tuples: &[OptionTuple]) -> Result<usize, Stop> {
        let parser = self.parser;
        if tuples.len() > 1 {
            let matches: Vec<&str> = tuples.iter().map(|t| t.option_string.as_str()).collect();
            return Err(parser
                .error(format!(
                    "ambiguous option: {} could match {}",
                    self.args[start],
                    matches.join(", ")
                ))
                .into());
        }
        let OptionTuple {
            mut action,
            mut option_string,
            mut sep,
            mut explicit_arg,
        } = tuples[0].clone();
        let mut action_tuples: Vec<(Action, Vec<String>)> = Vec::new();
        let stop;
        loop {
            let Some(current) = action else {
                self.extras.push(self.args[start].clone());
                return Ok(start + 1);
            };
            if let Some(explicit) = explicit_arg.clone() {
                let single_dash = option_string.chars().nth(1) != Some('-');
                if !current.takes_value() && single_dash && !explicit.is_empty() {
                    let sep_set = sep.as_deref().is_some_and(|s| !s.is_empty());
                    if sep_set || explicit.starts_with('-') {
                        return Err(parser
                            .error(argument_error(
                                Some(current),
                                format!("ignored explicit argument {}", repr_str(&explicit)),
                            ))
                            .into());
                    }
                    action_tuples.push((current, Vec::new()));
                    let mut rest = explicit.chars();
                    let next_char = rest.next().expect("a non-empty explicit argument");
                    let rest = rest.as_str();
                    option_string = format!("-{next_char}");
                    if let Some(next_action) = parser.lookup(&option_string) {
                        action = Some(next_action);
                        if rest.is_empty() {
                            sep = None;
                            explicit_arg = None;
                        } else if let Some(after) = rest.strip_prefix('=') {
                            sep = Some("=".to_string());
                            explicit_arg = Some(after.to_string());
                        } else {
                            sep = Some(String::new());
                            explicit_arg = Some(rest.to_string());
                        }
                    } else {
                        self.extras.push(format!("-{explicit}"));
                        stop = start + 1;
                        break;
                    }
                } else if current.takes_value() {
                    action_tuples.push((current, vec![explicit]));
                    stop = start + 1;
                    break;
                } else {
                    return Err(parser
                        .error(argument_error(
                            Some(current),
                            format!("ignored explicit argument {}", repr_str(&explicit)),
                        ))
                        .into());
                }
            } else {
                let next = start + 1;
                let count = if current.takes_value() {
                    if self.patterns.get(next) != Some(&Pattern::Arg) {
                        return Err(parser
                            .error(argument_error(Some(current), "expected one argument"))
                            .into());
                    }
                    1
                } else {
                    0
                };
                action_tuples.push((current, self.args[next..next + count].to_vec()));
                stop = next + count;
                break;
            }
        }
        for (action, strings) in action_tuples {
            self.take_action(action, strings)?;
        }
        Ok(stop)
    }

    /// `take_action`: convert and check the value, then act on it.
    fn take_action(&mut self, action: Action, strings: Vec<String>) -> Result<(), Stop> {
        if !self.seen.contains(&action) {
            self.seen.push(action);
        }
        match action {
            Action::Help => {
                return Err(Stop::Help(if self.parser.prog == TOP_PROG {
                    TOP_HELP
                } else {
                    RUN_HELP
                }));
            }
            // `_check_value`: the choices of the command and of `--format`.
            Action::Command | Action::Format => {
                let choices: &[&str] = if action == Action::Command {
                    &["run"]
                } else {
                    FORMAT_CHOICES
                };
                if !choices.contains(&strings[0].as_str()) {
                    let shown: Vec<String> = choices.iter().map(|c| repr_str(c)).collect();
                    let message = format!(
                        "invalid choice: {} (choose from {})",
                        repr_str(&strings[0]),
                        shown.join(", ")
                    );
                    return Err(self
                        .parser
                        .error(argument_error(Some(action), message))
                        .into());
                }
            }
            _ => {}
        }
        self.taken.push((action, strings));
        Ok(())
    }
}

/// `parser.parse_args(argv)` for the whole CLI: the top-level parser hands every string
/// after the command to the `run` subparser, and unrecognized strings from both are
/// reported by the top-level parser.
fn parse(args: &[String]) -> Result<Parsed, ParseError> {
    let (taken, mut extras) = match TOP.parse_known(args) {
        Ok(parsed) => parsed,
        Err(Stop::Help(text)) => return Ok(Parsed::Help(text)),
        Err(Stop::Error(err)) => return Err(err),
    };
    let command_args = taken
        .into_iter()
        .find(|(action, _)| *action == Action::Command)
        .map(|(_, strings)| strings)
        .expect("the required command was parsed");
    let (taken, sub_extras) = match RUN.parse_known(&command_args[1..]) {
        Ok(parsed) => parsed,
        Err(Stop::Help(text)) => return Ok(Parsed::Help(text)),
        Err(Stop::Error(err)) => return Err(err),
    };
    extras.extend(sub_extras);
    if !extras.is_empty() {
        return Err(TOP.error(format!("unrecognized arguments: {}", extras.join(" "))));
    }

    let mut run_args = RunArgs {
        dataset: None,
        split: None,
        gt: None,
        format: "mot".to_string(),
        pred: String::new(),
        metrics: DEFAULT_METRICS.to_string(),
        out_csv: None,
        out_json: None,
    };
    // Later occurrences overwrite earlier ones; `type=Path` normalizes path values.
    for (action, mut strings) in taken {
        let value = strings.pop().expect("a stored option has one value");
        match action {
            Action::Dataset => run_args.dataset = Some(value),
            Action::Split => run_args.split = Some(value),
            Action::Gt => run_args.gt = Some(pyfmt::path_str(&value)),
            Action::Format => run_args.format = value,
            Action::Pred => run_args.pred = pyfmt::path_str(&value),
            Action::Metrics => run_args.metrics = value,
            Action::OutCsv => run_args.out_csv = Some(pyfmt::path_str(&value)),
            Action::OutJson => run_args.out_json = Some(pyfmt::path_str(&value)),
            Action::Help | Action::Command => unreachable!("not a stored option"),
        }
    }
    Ok(Parsed::Run(run_args))
}

// ------------------------------------------------------------------------- the command

/// How `_run` fails: a `ValueError` (reported with the top-level usage, exit 2) or an
/// exception Python does not catch (exit 1).
enum Failure {
    Usage(String),
    Uncaught(String),
}

/// Python's `str.strip()` whitespace: Unicode `White_Space` plus the four information
/// separators U+001C..U+001F, which `str.isspace()` also counts.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `_resolve_metrics`: comma-separated, stripped, lowercased names, in order.
fn resolve_metrics(raw: &str) -> Result<Vec<Box<dyn Metric>>, String> {
    let names: Vec<String> = raw
        .split(',')
        .map(|name| name.trim_matches(is_py_space))
        .filter(|name| !name.is_empty())
        .map(str::to_lowercase)
        .collect();
    let mut unknown: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|name| !METRIC_NAMES.contains(name))
        .collect();
    unknown.sort_unstable();
    unknown.dedup();
    let available = METRIC_NAMES.join(", ");
    if !unknown.is_empty() {
        return Err(format!(
            "unknown metric(s): {}; available: {available}",
            unknown.join(", ")
        ));
    }
    if names.is_empty() {
        return Err(format!("no metrics selected; available: {available}"));
    }
    Ok(names
        .iter()
        .map(|name| -> Box<dyn Metric> {
            match name.as_str() {
                "hota" => Box::new(Hota),
                "clear" => Box::new(Clear),
                "identity" => Box::new(Identity),
                "count" => Box::new(Count),
                "track_map" => Box::new(TrackMap),
                "jf" => Box::new(JAndF),
                other => unreachable!("unknown metric {other} passed the check"),
            }
        })
        .collect())
}

/// `_run` catches `ValueError` (and its subclasses) only; any other exception class is an
/// uncaught traceback, exit 1.
fn failure(exception: &str, message: String) -> Failure {
    if is_value_error(exception) {
        Failure::Usage(message)
    } else {
        Failure::Uncaught(message)
    }
}

fn benchmark_failure(err: BenchmarkError) -> Failure {
    failure(err.python_exception(), err.to_string())
}

fn eval_failure(err: EvalError) -> Failure {
    failure(err.python_exception(), err.to_string())
}

/// `_load`: a built-in benchmark by name, or `--gt` in the `--format` layout.
fn load(args: &RunArgs) -> Result<MOTDataset, Failure> {
    let gt = args.gt.as_deref().map(Path::new);
    if let Some(name) = &args.dataset {
        return benchmarks::load_dataset(name, gt, args.split.as_deref())
            .map_err(benchmark_failure);
    }
    let Some(gt) = gt else {
        return Err(Failure::Usage(
            "--gt is required when --dataset is omitted".to_string(),
        ));
    };
    let split = args.split.as_deref().unwrap_or("train");
    let loaded = if args.format == "mots" {
        // `load_mots(root, split)` evaluates its default class, 2 (MOTS pedestrians).
        benchmarks::mots20::load_mots(gt, split, 2)
    } else {
        benchmarks::motchallenge::load_motchallenge(gt, split)
    };
    loaded.map_err(benchmark_failure)
}

/// `_run`.
fn run_command(args: &RunArgs, stdout: &mut dyn Write) -> Result<(), Failure> {
    let metrics = resolve_metrics(&args.metrics).map_err(Failure::Usage)?;
    if let Some(gt) = &args.gt
        && !Path::new(gt).is_dir()
    {
        return Err(Failure::Usage(format!(
            "ground-truth root not found or not a directory: {gt}"
        )));
    }
    if !Path::new(&args.pred).is_dir() {
        return Err(Failure::Usage(format!(
            "prediction directory not found or not a directory: {}",
            args.pred
        )));
    }

    let dataset = load(args)?;
    let result = evaluate(&dataset, Path::new(&args.pred), &metrics).map_err(eval_failure)?;
    let text = table(&result).map_err(|err| match err {
        TableError::NanToInteger => Failure::Usage(err.to_string()),
        TableError::InfinityToInteger | TableError::MissingScore { .. } => {
            Failure::Uncaught(err.to_string())
        }
    })?;
    let _ = writeln!(stdout, "{text}");
    let _ = stdout.flush();
    if let Some(path) = &args.out_csv {
        write_output(path, &results::to_csv(&result))?;
    }
    if let Some(path) = &args.out_json {
        write_output(
            path,
            &results::to_json(&result, &dataset.name, &dataset.split),
        )?;
    }
    Ok(())
}

/// Create the parent directories, then write `text` (`_write_csv`, `_write_json`).
fn write_output(path: &str, text: &str) -> Result<(), Failure> {
    let path = Path::new(path);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|err| Failure::Uncaught(err.to_string()))?;
    }
    std::fs::write(path, text).map_err(|err| Failure::Uncaught(err.to_string()))
}

// ---------------------------------------------------------------------------- the table

/// Why the headline table cannot be built. Python raises these from `_table`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableError {
    /// `int(nan)` on an integer column (`ValueError`).
    NanToInteger,
    /// `int(inf)` on an integer column (`OverflowError`).
    InfinityToInteger,
    /// A row lacks a headline that the combined scores hold (`KeyError`).
    MissingScore { metric: String, field: String },
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NanToInteger => f.write_str("cannot convert float NaN to integer"),
            Self::InfinityToInteger => f.write_str("cannot convert float infinity to integer"),
            Self::MissingScore { metric, field } => {
                write!(
                    f,
                    "missing score {}",
                    repr_str(&format!("{metric}/{field}"))
                )
            }
        }
    }
}

impl std::error::Error for TableError {}

/// `_format_headline`: `{:d}` of `int(value)` for counts, `{:1.5g}` of `100 * value` for
/// ratios.
fn format_headline(value: f64, label: &str) -> Result<String, TableError> {
    if !INTEGER_HEADLINES.contains(&label) {
        return Ok(pyfmt::format_g(100.0 * value, 5));
    }
    if value.is_nan() {
        return Err(TableError::NanToInteger);
    }
    if value.is_infinite() {
        return Err(TableError::InfinityToInteger);
    }
    let truncated = value.trunc();
    // `{:.0}` of an integral float prints its exact decimal value, as `int()` does.
    Ok(if truncated == 0.0 {
        "0".to_string()
    } else {
        format!("{truncated:.0}")
    })
}

/// `_table`: the headline table the CLI prints, one row per sequence plus `COMBINED`,
/// without the final newline.
///
/// Columns are the `_HEADLINES` whose metric the combined scores (the class-averaged
/// scores for a multi-class run) hold. A multi-class run adds a leading `class` column,
/// repeats the rows for each class, and ends with `class_averaged` and `det_averaged`
/// rows. Label columns are left-aligned, values right-aligned, columns joined by two
/// spaces; widths count characters, as Python's `len` does.
pub fn table(result: &EvaluationResult) -> Result<String, TableError> {
    let available = if result.is_multi_class() {
        &result.class_averaged
    } else {
        &result.combined
    };
    let columns: Vec<&(&str, &str, &str)> = HEADLINES
        .iter()
        .filter(|(metric, _, _)| available.get(metric).is_some())
        .collect();
    let label_names = results::row_label_names(result);
    let mut headers: Vec<String> = label_names.iter().map(|s| s.to_string()).collect();
    headers.extend(columns.iter().map(|(_, _, label)| label.to_string()));

    let mut rows: Vec<Vec<String>> = Vec::new();
    for (labels, scores) in results::scored_rows(result) {
        let mut values = labels;
        for (metric, field, label) in &columns {
            let value = scores
                .get(metric)
                .and_then(|fields| fields.get(field))
                .ok_or_else(|| TableError::MissingScore {
                    metric: metric.to_string(),
                    field: field.to_string(),
                })?;
            values.push(format_headline(results::scalar_value(value), label)?);
        }
        rows.push(values);
    }

    let width = |index: usize| {
        rows.iter()
            .map(|row| row[index].chars().count())
            .chain([headers[index].chars().count()])
            .max()
            .unwrap_or(0)
    };
    let widths: Vec<usize> = (0..headers.len()).map(width).collect();
    let line = |cells: &[String]| {
        let parts: Vec<String> = cells
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let w = widths[index];
                if index < label_names.len() {
                    format!("{cell:<w$}")
                } else {
                    format!("{cell:>w$}")
                }
            })
            .collect();
        parts.join("  ")
    };
    let mut lines = vec![line(&headers)];
    lines.extend(rows.iter().map(|row| line(row)));
    Ok(lines.join("\n"))
}
