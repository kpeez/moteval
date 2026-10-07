# Architecture and test fixes

Status: approved on 2026-10-07, with every default decision. Not started.

This plan comes from the architecture review and the test audit that followed PR #39
(ty → pyrefly) and PR #40 (test-audit prune). Six pull requests carry it. The interactive
version of this plan is [architecture-and-test-fixes.html](architecture-and-test-fixes.html)
(open it in a browser). The workflow that dispatches the work is
[architecture-and-test-fixes.workflow.js](architecture-and-test-fixes.workflow.js).

## Decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| TrackMAP per-sequence entry | Absent from the results | AP pools detections across all sequences. Upstream never reports a per-sequence AP, so it has no parity check. |
| Multi-class result shape | New fields; single-class output unchanged | Every built-in benchmark is single-class. Current users keep the same JSON, CSV and API. |
| How predictions give their class | Column 8 of a MOTChallenge row, read only for multi-class protocols | It matches `gt.txt`. Many prediction files hold -1 in that column, so single-class reading must not change. |
| Missing prediction file | Raise | TrackEval raises (`trackeval/datasets/mot_challenge_2d_box.py:120-126` at `12c8791b`). A silent zero gives plausible wrong scores. |
| Expected values for untested quirks | Oracle scenarios through `scripts/regen_parity_fixtures.py` | These are parity quirks, so the proof must come from TrackEval. |
| Loader default root | Loaders take a required `root` | Only `load_dataset` knows the data location. |
| BDD100K loader | Not in this plan | ADR-0003 (scope boundaries) excludes it. Adding it means revisiting that ADR first, then a plan on top of `pr-multi-class`. |

## Merge order

| Wave | PRs | Waits for |
| --- | --- | --- |
| 1 | `pr-results-fields`, `pr-data-root`, `pr-test-repairs` | nothing; run in parallel |
| 2 | `pr-multi-class` | `pr-results-fields` (`moteval/eval.py`) |
| 3 | `pr-quirk-tests` | `pr-multi-class` (fixture regen) and `pr-data-root` (`tests/scenarios.py`) |
| 3 | `pr-missing-preds` | `pr-multi-class` (`moteval/eval.py`) and `pr-test-repairs` (`tests/test_data.py`) |

Each PR rebases on `main` before it merges. AGENTS.md and README.md take small edits
from several PRs, in different sections.

## Rules for every PR

- Numbers stay bit-identical to TrackEval `12c8791b`. Regenerate parity fixtures with
  `scripts/regen_parity_fixtures.py`; never hand-edit them. A fixture diff may only add
  entries unless the PR says otherwise.
- Gates: `just check`, `just test`, `just test-real`. Point `data/benchmarks` at
  `/data/tracking/external` for the real-data gate. MOTS20 is not on this machine, so its
  real-data test skips; state that in the PR, never call it a pass.
- Every new or changed test must be seen red. Mutate the production owner, run the test,
  then restore the file byte for byte. Run mutation probes with
  `PYTHONDONTWRITEBYTECODE=1` after you delete `__pycache__`: a same-size, same-second
  source swap can reuse a stale `.pyc`. Do not keep scratch tests under `tests/temp/`,
  because pytest collects them.
- Expected values in tests are hand-derived or come from the TrackEval oracle, never
  copied from moteval output.
- An independent reviewer approves each PR before it merges.
- Project ADRs are in `.agents/docs/adrs/` (the llmOS vault). ADR-0007 (CLI output
  contract) requires a deliberate decision for any JSON schema change. This plan is that
  decision for `pr-results-fields` and `pr-multi-class`; each of those PRs also records
  its schema change in ADR-0007, following the vault rules.

## pr-results-fields

Results and exports hold only the fields that each metric declares.

- `evaluate()` stores in `EvaluationResult.per_sequence[seq][metric]` only the keys in
  `metric.fields`. If nothing is left (TrackMAP per sequence), it omits the metric key.
- `EvaluationResult.combined[metric]` is filtered the same way. This hides TrackMAP's
  `_num_dt_*` weights.
- `combine_sequences()` still receives the raw per-sequence state. Keep the raw combined
  results available inside `evaluate()`, because `pr-multi-class` feeds them (with
  `_num_dt_*`) to `combine_classes_det_averaged`.
- JSON, CSV and the CLI follow `EvaluationResult`. The 51 private TrackMAP CSV rows on
  the toy dataset go away.
- Add one parametrized test: per-sequence keys are a subset of `metric.fields`, and
  combined keys equal `metric.fields`, for HOTA, CLEAR, Identity, Count and TrackMAP.
- Update the README results description and the `Metric.fields` docstring. Record in
  ADR-0007 that exports now hold only declared fields.

## pr-data-root

`load_dataset(name)` finds data where the downloader puts it.

- Add `default_data_root()` to `moteval/benchmarks/__init__.py`. It returns
  `MOTEVAL_DATA_ROOT` if set, raises on an empty value, and otherwise returns
  `data/benchmarks`.
- `load_dataset(name, root=None, split=None)` uses `root`, else
  `default_data_root() / name`.
- Every loader takes a required `root`. Remove `MOTChallengeConfig.default_root` and the
  per-module default paths. Update the `DatasetLoader` comment and `load_toy` in
  `tests/conftest.py`.
- `scripts/download_benchmarks.py` keeps `--root` first and uses `default_data_root()`
  for the fallback.
- `tests/scenarios.py` and the real-data gate use `default_data_root()`.
- In the same files: `load_mots` derives its protocol from `MOTS20_PROTOCOL` with
  `dataclasses.replace`, and `_read_seq_length` becomes public for `mots20.py`.
- Test that `load_dataset` honors `MOTEVAL_DATA_ROOT` and rejects an empty value.
- Update AGENTS.md, README.md, `moteval/benchmarks/README.md` and `docs/DATASETS.md`.

## pr-test-repairs

Weak tests get assertions that can fail.

- Add a test that overlapping predicted masks raise (`moteval/data/convert.py:144-145`).
- `test_toy_run_prints_sequence_and_combined_headlines`: every headline is 100, so a swap
  of two columns passes. Use predictions that give each column a distinct, hand-derived
  value.
- `_write_predictions_matching_gt` (`tests/test_data.py`) and the `toy_predictions`
  fixture (`tests/test_cli.py`) copy GT rows as predictions, which AGENTS.md forbids.
  Give the predictions their own track ids.
- Add two AGENTS.md gotchas: pytest collects `test_*.py` under the gitignored
  `tests/temp/`, and mutation probes need `PYTHONDONTWRITEBYTECODE=1`.

## pr-multi-class

`evaluate()` scores each class and both class combinations.

- Remove the single-class guard. For each class in `protocol.eval_classes`, build the
  sequence data with `cls_id`, run every metric, and combine sequences per class. Then
  call `combine_classes_class_averaged` and `combine_classes_det_averaged` on the raw
  per-class combined results.
- `EvaluationResult` keeps `per_sequence` and `combined` unchanged for single-class runs,
  and adds `per_class`, `class_averaged` and `det_averaged`, which stay empty for a
  single class. Single-class JSON, CSV and CLI output stay byte-identical; prove it with
  a test. Define the multi-class CSV rows and CLI table (a class column, plus
  class-averaged and det-averaged rows).
- `read_mot` gains an option to read the class from column 8. `evaluate()` uses it only
  for multi-class protocols.
- Extend the regen script to freeze `class_averaged` and `det_averaged` for every metric
  on the `combine_classes` scenario, then run it. It clones TrackEval and can need
  network access, cv2 and scikit-image. TrackMAP det-averaged stays the one permitted
  divergence: exclude it from the oracle check and pin it with a hand-derived test.
- Fix the comment at `tests/test_parity.py:73`: HOTA det-averaged is not a divergence.
  Replace `test_evaluate_rejects_multi_class_protocol` with tests of the new behavior.
- In the same PR: delete the pass-through `linear_sum_assignment` wrapper in
  `moteval/metrics/_matching.py`, make `_matching.EPS` the one EPS for the metrics, and
  re-export `JAndF` from `moteval/metrics/__init__.py`.
- Update AGENTS.md and README.md, and record the multi-class JSON, CSV and table shape in
  ADR-0007.

## pr-quirk-tests

Three replicated TrackEval quirks get oracle scenarios. Each scenario must turn red under
the named mutation.

- J&F decay bins cast to uint8 wrap past 255 frames (`moteval/metrics/jf.py:225-226`):
  one MOTS sequence of approximately 300 frames. Mutation: drop the cast.
- MOTS `matching_fill = -10000` (`moteval/data/protocol.py:106`): a tie of one 1.0 pair
  against two 0.5 pairs next to an ignore region. Mutation: set the MOTS fill to 0. If no
  scenario can tell the fills apart, report that instead of adding a vacuous one.
- TrackMAP scalar `_track_iou` fallback (`moteval/metrics/track_map.py:176-178`): a track
  pair whose frame set iterates out of order, such as frames {2, 9}, with nonzero
  intersection. Mutation: break the fallback.

## pr-missing-preds

A missing prediction file stops the run with a clear error.

- Before it scores anything, `evaluate()` raises `ValueError` that names the prediction
  directory and every missing sequence. Remove the `is_file()` fallbacks. Add a CLI test
  that the message reaches stderr with no traceback.
- Five tests relied on a missing file meaning "no predictions" (a probe found them on
  `9d1b10d`; probe again after rebase):
  `test_evaluate_with_missing_prediction_file_reports_zero_preds` (becomes the raise
  test), `test_load_motchallenge_drops_conf_zero_gt_rows`,
  `test_mots20_overlapping_gt_masks_raise`,
  `test_multi_sequence_combine_is_detection_weighted` and
  `test_empty_predictions_sequence`. They write an empty file instead.
- Document in README.md and AGENTS.md that every sequence needs a file, and that an
  empty file means no predictions.

## Not changing

- Metric numbers, file formats for single-class runs, and CLI flags.
- The box and mask conversion paths stay separate.
- `box_iou` and `box_ioa` keep their duplicated lines, for ULP parity.
- The built-in loaders stay single-class. No BDD100K loader and no MOTS20 download.

## Dispatch

Run the workflow script from a session in this repository. It creates one worktree for
each PR off `origin/main`, never touches the main checkout, and runs each PR through
implement, independent review, up to two fix rounds, and merge. A PR starts only after
the PRs it waits for have merged. If a PR does not pass review, the PRs that depend on
it are skipped. Expect 18 to 30 agents.
