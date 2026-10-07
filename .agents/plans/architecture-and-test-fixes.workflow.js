export const meta = {
  name: 'moteval-fix-waves',
  description: 'Implement, independently review, fix, and merge the six approved moteval PRs in dependency order',
  whenToUse: 'After the moteval fixes plan was approved with default decisions',
  phases: [
    { title: 'Implement', detail: 'one implementer per PR, own worktree off origin/main, draft PR' },
    { title: 'Review', detail: 'independent reviewer per PR; re-review after fixes' },
    { title: 'Fix', detail: 'address blocking and should-fix findings, at most 2 rounds' },
    { title: 'Merge', detail: 'rebase if needed, wait for CI, squash-merge' },
  ],
}

const REPO = '/home/kylep/repos/moteval'
const PLAN = '.agents/plans/architecture-and-test-fixes.md (in your worktree; the interactive page is architecture-and-test-fixes.html beside it)'

const COMMON = `
You work on the moteval repo (GitHub kpeez/moteval). Read AGENTS.md in your worktree first; its non-negotiables bind you (bit-identical numbers vs TrackEval 12c8791b; parity fixtures in tests/fixtures/*.json are never hand-edited, only regenerated with scripts/regen_parity_fixtures.py; frame-indexing rules; no registration).
The approved plan is ${PLAN}. The user approved it with every default decision.

Hard rules:
- NEVER touch the main checkout at ${REPO}: do not checkout, stash, reset, commit, or edit files there. It holds the user's uncommitted work. Only run read-only git commands there (fetch, worktree add/list/remove).
- Work only in your own worktree (path given below). Inside it run \`uv sync --locked\` and create the gitignored real-data link: \`mkdir -p data && ln -sfn /data/tracking/external data/benchmarks\`.
- Required gates before any push: \`just check\`, \`just test\`, \`just test-real\`. MOTS20 is not on this machine, so its real-data test skips; say so plainly, never call a skip a pass.
- Every new or changed test must be seen red: mutate the production owner, run the test, confirm it fails, restore the file byte for byte. Run such probes with PYTHONDONTWRITEBYTECODE=1 and delete __pycache__ dirs (outside .venv) before each run, because a same-size, same-second source swap can reuse a stale .pyc. Never put scratch test files under tests/temp/ (pytest collects them).
- Expected values in tests must be hand-derived or come from the TrackEval oracle via the regen script, never copied from moteval output.
- Never spawn \`uv run\` from inside a test.
- Commits: task-scoped, clear messages, no attribution lines. Push only your branch to origin.
- Report failures honestly with the command output.`

const PRS = [
  {
    slug: 'results-fields', branch: 'fix/results-fields', deps: [],
    title: 'Results and exports hold only declared metric fields',
    spec: `Plan claim 1 (pr-results-fields).
- evaluate() (moteval/eval.py) stores in EvaluationResult.per_sequence[seq][metric] only the keys in metric.fields. If that leaves nothing (TrackMAP per sequence, whose eval_sequence returns only combine state), omit the metric key for that sequence (decision: the per-sequence TrackMAP entry is absent).
- EvaluationResult.combined[metric] is likewise filtered to metric.fields (drops TrackMAP's _num_dt_* weights).
- combine_sequences() must still receive the raw, unfiltered per-sequence state. Keep the raw combined results available inside evaluate(): a follow-up PR (multi-class) will feed raw per-class combined results, including _num_dt_*, to combine_classes_det_averaged. Make that seam obvious.
- JSON/CSV/CLI follow from EvaluationResult; confirm the 51 private CSV rows for TrackMAP are gone.
- Add one parametrized test: for HOTA, CLEAR, Identity, Count, TrackMAP (and JAndF on a mask dataset if cheap), per-sequence keys are a subset of metric.fields and combined keys equal metric.fields. Mutation-audit it (e.g. remove the filter -> red).
- No numeric change anywhere; all parity tests pass unchanged.
- Update docs the change makes stale (README results description, Metric.fields docstring as the public contract).`,
  },
  {
    slug: 'data-root', branch: 'fix/data-root', deps: [],
    title: 'load_dataset finds data where the downloader puts it',
    spec: `Plan claim 3 (pr-data-root).
- Add default_data_root() in moteval/benchmarks/__init__.py: returns Path(MOTEVAL_DATA_ROOT) if the env var is set (raise ValueError if it is empty or whitespace), else Path("data/benchmarks").
- load_dataset(name, root=None, split=None) uses root if given, else default_data_root() / name.
- Every loader takes a required root (decision: loaders hold no default path). Remove MOTChallengeConfig.default_root and the per-module default-root constants (chimpact, panaf500, mots20, and the configs). Update the DatasetLoader comment (benchmarks/__init__.py:24) and tests/conftest.py load_toy accordingly.
- scripts/download_benchmarks.py: _resolve_root keeps --root precedence and uses default_data_root() for the fallback (one owner of the rule). Keep the existing env-var tests green.
- tests/scenarios.py DATA_ROOT and the real-data gate use default_data_root(), so MOTEVAL_DATA_ROOT also drives \`just test-real\`.
- Small cleanup in the same files: load_mots derives its protocol from MOTS20_PROTOCOL via dataclasses.replace instead of re-declaring values; make motchallenge._read_seq_length public and use it from mots20.py instead of a private import.
- Tests: load_dataset honors MOTEVAL_DATA_ROOT (monkeypatch env + tmp layout) and rejects an empty value; mutation-audit both.
- Update AGENTS.md (data/benchmarks gotcha), README.md, moteval/benchmarks/README.md and docs/DATASETS.md where they describe default locations.`,
  },
  {
    slug: 'test-repairs', branch: 'test/repairs', deps: [],
    title: 'Repair weak tests and record two test traps',
    spec: `Plan claims 5.2-5.5 (pr-test-repairs).
- Add a test that overlapping predicted masks raise (moteval/data/convert.py:144-145; only the GT branch is tested today). Mutation-audit it.
- tests/test_cli.py test_toy_run_prints_sequence_and_combined_headlines: every headline is 100, so a swap of two equal columns passes. Use predictions that differ from GT so the headline columns have distinct values, with expected values hand-derived and the derivation written in a short comment. Mutation-audit by swapping two columns in moteval/cli.py _HEADLINES.
- AGENTS.md says regression tests must number predictions independently of GT. tests/test_data.py _write_predictions_matching_gt and the tests/test_cli.py toy_predictions fixture copy GT rows verbatim. Give predictions their own track ids (boxes may stay equal) and keep every dependent assertion valid; re-run the whole suite.
- AGENTS.md Gotchas: add the two traps from the plan (pytest collects test_*.py under gitignored tests/temp/; same-size same-second source swaps can reuse a stale .pyc, so mutation probes run with PYTHONDONTWRITEBYTECODE=1 after purging __pycache__).`,
  },
  {
    slug: 'multi-class', branch: 'feat/multi-class', deps: ['results-fields'],
    title: 'evaluate() scores each class and both class combinations',
    spec: `Plan claim 2 (pr-multi-class). Builds on the merged results-fields PR.
- evaluate(): remove the single-class guard (eval.py). For each cls_id in protocol.eval_classes, build SequenceData with build_sequence_data(..., cls_id) / build_mask_sequence_data, run every metric, and combine sequences per class. Then compute class_averaged = metric.combine_classes_class_averaged(raw per-class combined) and det_averaged = metric.combine_classes_det_averaged(raw per-class combined). Feed the RAW per-class combined results (TrackMAP needs _num_dt_*); filter to metric.fields only for output.
- Decision: additive schema. EvaluationResult keeps per_sequence and combined with today's exact meaning and values for single-class protocols, and gains per_class (class id -> per_sequence + combined), class_averaged and det_averaged, all empty for a single class. JSON export adds those keys only for multi-class runs; single-class JSON, CSV and CLI output stay byte-identical (prove it with a test). Define and document the multi-class CSV rows and the CLI table (a class column, plus class-averaged and det-averaged COMBINED rows) for multi-class runs only.
- Decision: prediction rows give their class in column 8 (index 7) of MOTChallenge box rows, read only for multi-class protocols (many MOT prediction files hold -1 there). Extend read_mot with an explicit option; single-class reading is unchanged. MOTS rows already carry their class.
- Oracle fixtures: extend scripts/regen_parity_fixtures.py to freeze class_averaged and det_averaged for every metric on the existing combine_classes scenario (and a mask equivalent for JAndF if the oracle path supports it), then run the regen script (it clones TrackEval; network access, and cv2/scikit-image for J&F, may be needed). The fixture diff must only ADD keys: every existing value stays byte-identical; check this with a diff. Fixtures are never hand-edited. TrackMAP det_averaged is the one permitted divergence (upstream copy-paste bug): exclude it from the oracle comparison and pin moteval's detection-weighted result with a hand-derived test.
- Fix the wrong comment at tests/test_parity.py:73 (HOTA det_averaged is not a divergence; only TrackMAP's is). Replace test_data.py test_evaluate_rejects_multi_class_protocol with tests of the new behavior.
- Cleanup in the same PR: delete the pass-through linear_sum_assignment wrapper in moteval/metrics/_matching.py (callers import scipy directly); _matching.EPS becomes the one EPS for metrics (track_map.py and jf.py import it; data/protocol.py keeps its own); moteval/metrics/__init__.py also re-exports JAndF.
- Update AGENTS.md and README.md (multi-class support, result schema, the divergence wording).
- If network or oracle dependencies block the regen, stop and report instead of hand-editing fixtures.`,
  },
  {
    slug: 'quirk-tests', branch: 'test/quirk-oracle', deps: ['multi-class', 'data-root'],
    title: 'Oracle scenarios for three untested TrackEval quirks',
    spec: `Plan claim 5.1 (pr-quirk-tests). Decision: oracle scenarios through the regen script.
Add parity scenarios in tests/scenarios.py, regenerate fixtures with scripts/regen_parity_fixtures.py (never hand-edit), and prove each scenario reaches its quirk with a mutation that turns its parity test red:
1. J&F decay bins cast to uint8 wrap for sequences longer than 255 frames (moteval/metrics/jf.py:225-226): one MOTS sequence of approximately 300 frames. Mutation: drop the uint8 cast.
2. MOTS matching_fill = -10000 vs the box path's 0 (moteval/data/protocol.py:106): a scenario where the fill changes which prediction counts as unmatched for the ignore-region test (for example an assignment tie of one 1.0 pair vs two 0.5 pairs next to an ignore region). Mutation: set the MOTS fill to 0. If no discriminating scenario can be built, report that instead of adding a vacuous scenario.
3. TrackMAP scalar _track_iou fallback (moteval/metrics/track_map.py:176-178): a track pair whose frame-union set iterates non-ascending (e.g. frames {2, 9}) with nonzero intersection. Mutation: break _track_iou or the fallback branch.
The fixture diff must only add entries; existing values stay byte-identical (check with a diff). If network or oracle dependencies block the regen, stop and report.`,
  },
  {
    slug: 'missing-preds', branch: 'fix/missing-predictions', deps: ['multi-class', 'test-repairs'],
    title: 'A missing prediction file stops the run with a clear error',
    spec: `Plan claim 4 (pr-missing-predictions). Decision: raise, like TrackEval (mot_challenge_2d_box.py:120-126 raises TrackEvalException for a missing tracker file).
- evaluate(): before scoring any sequence, collect every missing <seq>.txt and raise ValueError naming the prediction directory and all missing sequence names. Remove the \`if pred_file.is_file() else ()\` fallbacks. The CLI already turns ValueError into parser.error; add a CLI test that the message reaches stderr without a traceback.
- These tests relied on a missing file meaning no predictions (found by a probe on the pre-multi-class main; re-probe after rebasing): tests/test_data.py test_evaluate_with_missing_prediction_file_reports_zero_preds (turn it into the raise test), tests/test_loaders.py test_load_motchallenge_drops_conf_zero_gt_rows, tests/test_masks.py test_mots20_overlapping_gt_masks_raise, tests/test_metrics.py test_multi_sequence_combine_is_detection_weighted and test_empty_predictions_sequence. Make them write an empty file where they mean no predictions.
- Mutation-audit the new raise test.
- Document in README.md and AGENTS.md: every sequence needs a prediction file; an empty file means no predictions.`,
  },
]

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['ok', 'failed'] },
    pr_number: { type: 'integer' },
    branch: { type: 'string' },
    worktree: { type: 'string' },
    head_sha: { type: 'string' },
    summary: { type: 'string' },
    verification: { type: 'string', description: 'gates run with results, mutation audits run with results' },
    gaps: { type: 'string', description: 'anything not done or not verified' },
  },
  required: ['status', 'branch', 'worktree', 'summary', 'verification', 'gaps'],
}
const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['approve', 'changes'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocking', 'should_fix', 'nit'] },
          file: { type: 'string' },
          line: { type: 'integer' },
          summary: { type: 'string' },
          evidence: { type: 'string' },
        },
        required: ['severity', 'summary', 'evidence'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['verdict', 'findings'],
}
const FIX_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['ok', 'failed'] },
    head_sha: { type: 'string' },
    summary: { type: 'string' },
    verification: { type: 'string' },
  },
  required: ['status', 'summary', 'verification'],
}
const MERGE_SCHEMA = {
  type: 'object',
  properties: {
    merged: { type: 'boolean' },
    merge_sha: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['merged', 'notes'],
}

function worktreeOf(pr) { return `${REPO}-${pr.slug}` }

async function runPR(pr) {
  const wt = worktreeOf(pr)
  const impl = await agent(`${COMMON}

TASK: implement PR "${pr.title}" as one draft pull request.
Create your worktree first: \`git -C ${REPO} fetch origin && git -C ${REPO} worktree add -b ${pr.branch} ${wt} origin/main\` (if the path exists from an earlier attempt, inspect it before reusing). Then follow the setup rules above.

SPEC:
${pr.spec}

Then: run all gates, mutation-audit the tests you added or changed, commit, push the branch, and open a DRAFT PR with \`gh pr create --draft --base main --head ${pr.branch}\`. The PR body states the scope, each decision applied, the verification actually run (with the MOTS20 skip stated), and any gaps. Do not merge. Return the PR number, worktree path and head SHA.`,
    { label: `implement:${pr.slug}`, phase: 'Implement', schema: IMPL_SCHEMA, agentType: 'swe:implementer' })
  if (!impl || impl.status !== 'ok' || !impl.pr_number) {
    log(`${pr.slug}: implementation failed or opened no PR`)
    return { slug: pr.slug, merged: false, stage: 'implement', impl }
  }

  let review = null
  const rounds = []
  for (let round = 0; round < 3; round++) {
    review = await agent(`Independent review of moteval PR #${impl.pr_number} (branch ${pr.branch}, worktree ${wt}). You did not write it. Do not edit, commit or push in that worktree. For any probing (running tests, mutation checks) copy it first: \`rm -rf /tmp/claude-1001/review-${pr.slug} && cp -a ${wt} /tmp/claude-1001/review-${pr.slug}\`, and use PYTHONDONTWRITEBYTECODE=1 with __pycache__ purged.
Read AGENTS.md. moteval must stay bit-identical to TrackEval 12c8791b; fixtures are only regenerated, never hand-edited.

The spec this PR must satisfy (approved plan ${PLAN}, default decisions):
${pr.spec}
${round > 0 ? `\nThis is re-review round ${round}. Earlier findings and the fixer's report:\n${JSON.stringify(rounds, null, 1)}\nConfirm each earlier finding is actually resolved, and review any new changes.` : ''}

Check, with evidence (file:line, command output):
1. Does the diff (\`git -C ${wt} diff origin/main...HEAD\`) do what the spec says, and nothing outside it?
2. Numeric preservation: can any change alter a computed value, dtype or evaluation order on any path? Fixture diffs must only add entries; verify existing values are byte-identical.
3. Tests: can each new or changed test fail for its stated bug (try the mutation)? Are expected values hand-derived or oracle-derived, never copied from output?
4. Gates: run \`just check\`, \`just test\` and \`just test-real\` in your copy.
5. Docs made stale by the change are updated.
Severity: blocking = wrong results, spec violated, test cannot fail, gate fails; should_fix = real but contained problem; nit = optional. verdict is 'approve' only if there are no blocking or should_fix findings.`,
      { label: `review${round ? '#' + (round + 1) : ''}:${pr.slug}`, phase: 'Review', schema: REVIEW_SCHEMA, agentType: 'swe:reviewer' })
    if (!review) break
    const serious = review.findings.filter(f => f.severity !== 'nit')
    if (review.verdict === 'approve' && serious.length === 0) break
    if (round === 2) break
    const fix = await agent(`${COMMON}

TASK: address review findings on moteval PR #${impl.pr_number} in your existing worktree ${wt} (branch ${pr.branch}). Do not create a new worktree.
SPEC (for context):
${pr.spec}

FINDINGS (fix every blocking and should_fix item; fix nits if cheap; if you disagree with a finding, explain why with evidence instead of changing code):
${JSON.stringify(review.findings, null, 1)}

Re-run all gates and the relevant mutation audits, commit, and push. Return what you changed and the verification run.`,
      { label: `fix${round ? '#' + (round + 1) : ''}:${pr.slug}`, phase: 'Fix', schema: FIX_SCHEMA, agentType: 'swe:implementer' })
    rounds.push({ round: round + 1, findings: review.findings, fix })
    if (!fix || fix.status !== 'ok') { log(`${pr.slug}: fix round ${round + 1} failed`); break }
  }

  const serious = review ? review.findings.filter(f => f.severity !== 'nit') : [{ summary: 'no review result' }]
  if (!review || review.verdict !== 'approve' || serious.length) {
    log(`${pr.slug}: not approved; left as a draft PR`)
    return { slug: pr.slug, merged: false, stage: 'review', pr: impl.pr_number, review, rounds }
  }

  const merge = await agent(`Finish moteval PR #${impl.pr_number} (branch ${pr.branch}, worktree ${wt}). An independent review approved it. Never touch the main checkout at ${REPO} beyond read-only git commands.
Steps:
1. In the worktree: \`git fetch origin\`. If the branch is behind origin/main, rebase onto origin/main. Resolve any conflict so both sides' intent survives (other approved PRs merged meanwhile), then re-run \`just check\`, \`just test\` and \`just test-real\` and push with --force-with-lease. If a conflict touches numeric code or fixtures and you are not certain, stop and report instead of merging.
2. Wait for CI on the PR head: \`gh pr checks ${impl.pr_number} --watch\`. If any check fails, do not merge; report the failure.
3. Update the PR body with a short Review section (approved; summary of fixes made after review, if any): ${JSON.stringify(rounds.map(r => ({ round: r.round, fixed: r.fix && r.fix.summary })))}.
4. \`gh pr ready ${impl.pr_number}\` then \`gh pr merge ${impl.pr_number} --squash --delete-branch\`. Confirm with \`gh pr view ${impl.pr_number} --json state,mergeCommit\` that the state is MERGED.
5. Remove the worktree: \`git -C ${REPO} worktree remove --force ${wt}\` and delete the local branch if it still exists. (The data/benchmarks entry inside it is only a symlink; removing it does not touch /data.)
Return merged true/false, the merge commit SHA, and notes.`,
    { label: `merge:${pr.slug}`, phase: 'Merge', schema: MERGE_SCHEMA })
  if (!merge || !merge.merged) log(`${pr.slug}: merge did not complete`)
  return { slug: pr.slug, merged: !!(merge && merge.merged), stage: 'merge', pr: impl.pr_number, merge, review, rounds }
}

const done = {}
for (const pr of PRS) {
  done[pr.slug] = Promise.all(pr.deps.map(d => done[d])).then(depResults => {
    const blocked = depResults.filter(r => !r || !r.merged).map(r => (r && r.slug) || '?')
    if (blocked.length) {
      log(`${pr.slug}: skipped; dependency not merged: ${blocked.join(', ')}`)
      return { slug: pr.slug, merged: false, stage: 'skipped', blockedBy: blocked }
    }
    log(`${pr.slug}: starting`)
    return runPR(pr).catch(e => ({ slug: pr.slug, merged: false, stage: 'error', error: String(e) }))
  })
}

const results = await Promise.all(PRS.map(pr => done[pr.slug]))
return results.map(r => ({
  slug: r.slug,
  merged: r.merged,
  stage: r.stage,
  pr: r.pr,
  merge_sha: r.merge && r.merge.merge_sha,
  merge_notes: r.merge && r.merge.notes,
  review_rounds: r.rounds ? r.rounds.length : 0,
  last_findings: r.review ? r.review.findings : undefined,
  blockedBy: r.blockedBy,
  error: r.error,
}))
