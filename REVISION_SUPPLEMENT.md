# Camera-ready revision supplement

Branch `camera-ready-revision` (from `main` at 223abde). Working document for the WSDM 2027
camera-ready of *Capability Is Not Validity*; not part of the public artifact (drop it, or move it
to `notes/`, before the release). Everything below was produced from the committed corpus and the
scorer code; **no new model episode was run and none is required**.

Conventions: E-items are prose edits, in paper order, each with the current sentence quoted from
the submitted PDF and the minimal replacement. "must" corrects a statement the code or corpus does
not support; "should" answers a reviewer with existing data; "optional" is the authors' call.
Reviewer tags: R1 to R4, MR = meta-review.

---

## 1. What this branch already changed

| File | Change | Effect on reported numbers |
| --- | --- | --- |
| `evaluation/figdata.py` | Marks cached copies (`_cached_copy`), adds `corpus_counts()`, excludes `latency_c1/` from grid cells (`CELL_EXCLUDE_DIRS`) | Corpus count: 1,402 rows = 1,382 executed episodes = 1,350 distinct identities. PairAcc/EVPairAcc unchanged in every cell. gpt-5-mini R=32 per-run means now use 10 runs, not 14. |
| `scripts/paper_numbers.py` | §0 prints the three counts; new §1b (Table 1 calls column); §6 drops cached copies and the latency re-runs from cost averages; new §9 (single-pair cells) | §5.2.4 cost averages move in the second or third decimal (see §2.1 below). |
| `scripts/check_m3_threshold.py` (new) | Monotone-threshold test on the superseded v2 M3 traces | New sentence for §5.2.2 (E19). |
| `scripts/diag_m4_payload.py` (new) | M4 audit-boundary sensitivity over cited payloads | New sentence for §5.2.3 / §7 (E20, E26). |
| `evaluation/exec_support.py` | `exec_ev` returns `None` for A0/A1 (not applicable); docstring states the per-interface reach of the checker; fixes an invalid escape warning | None. |
| `agents/runner.py` | Passes the oracle closure time to the live scorer so fresh M4 runs use the anchored rule | None. `scripts/rescore.py --all` is idempotent on the committed corpus (0 rows changed in every directory). |
| `scripts/figure1.py`, `scripts/figure2.py`, `figures/figure1.*`, `figures/figure2.*` | Panel titles and one annotation reworded (E24); figures regenerated | Plotted values unchanged. |
| `PREREGISTRATION_C5.md` | Dated decision record appended (width proxy to anchored rule; boundary sensitivity; provenance caveat) | Cited by E12. |
| `README.md` | Counting rule, M4 scoring note, sampling note, missing controls file, new commands | Artifact only. |

Tests: `selftest_schemas` 46/46, `test_compiler` 18/18, `test_mechanisms` 12/12,
`test_operator_consistency` 11/11, `test_sweep_isolation` 4/4, `test_a0_equivalence` 27/27,
`test_resolution` PASS, `test_m1` PASS, `test_script_a1` runs. `bootstrap_contrast.py` still gives
+0.75 [+0.55, +0.95].

---

## 2. Numbers

### 2.1 Numbers that change (old → new)

| Where | Old | New | Why |
| --- | --- | --- | --- |
| Abstract, §4.5 | 1,402 scored runs / episodes | **1,382 executed episodes** (1,402 scored rows; 1,350 distinct identities) | 20 rows are cached copies of the Opus anchor episodes re-served into `resolution_opus/` |
| §5.2.4 Opus A3 | 6.2 calls and $0.298 per run | **6.1 calls and $0.299** | cached copies removed from the cost average |
| §5.2.4 Opus A1 | 7.6 / $0.414 | **7.7 / $0.412** | same |
| §5.2.4 Opus A2 | 8.3 / $0.399 | unchanged | |
| §5.2.4 mini ordering | $0.007 < $0.011 < $0.021 | unchanged | |
| §5.2.4 events per LLM round, Opus A1 | 107K | unchanged (106.6K rounds to 107K) | |
| §5.2.4 Opus A2 vs A3 | 17.6K vs. 25.9K | **17.6K vs. 26.1K** | |
| §5.2.4 mini A3 vs A2 | 9.4K vs. 11.8K | **9.3K vs. 11.8K** | |
| §5.2.4 interpreter overhead | 81.0 ms of 84.7 ms | **not reproducible from the artifact** (no script computes it; `paper_numbers.py` §6 gives mean backend ms per run, 1,349 Opus / 1,292 mini for A2) | keep only if the authors hold the measurement; otherwise apply E21b |

### 2.2 New numbers (all from committed traces)

**Table 1 calls column** (`paper_numbers.py` §1b): mean tool calls per run, and runs (of 10) that
used all 16 calls. No run in any of these cells ended with `budget_exhausted` (the turn cap).

| R | Opus A1 | at cap | mini A1 | at cap | Opus A2 | Opus A3 | mini A2 | mini A3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 15.8 | 8/10 | 12.6 | 3/10 | 7.7 | 12.9 | 1.9 | 6.3 |
| 128 | 15.8 | 9/10 | 11.9 | 2/10 | n/r | 12.5 | n/r | 4.6 |
| 192 | 16.0 | 10/10 | 12.1 | 3/10 | n/r | n/r | n/r | n/r |
| 256 | 15.9 | 9/10 | 9.8 | 3/10 | 8.0 | n/r | 2.3 | n/r |
| 384 | 16.0 | 10/10 | 11.9 | 2/10 | n/r | n/r | n/r | n/r |
| 512 | 16.0 | 10/10 | 13.2 | 6/10 | 7.4 | 13.5 | 3.7 | 4.9 |

Reading: Opus saturates the budget at every R, succeeding and failing alike, so budget saturation
by itself does not separate success from failure (R1 C4).

**M3 ablation, monotone-threshold test** (`check_m3_threshold.py`, v2 cells, A3, B=16):
gpt-5-mini's labels are not reproducible by any single cut on the returned peak score under any
reading (gold-parameter call, max over all calls, max over cited calls): seeds 1 and 2 both
returned 0.2857 and were labelled present and absent respectively; the lowest score, 0.2105, was
labelled present. Opus 5's labels are consistent with any cut in (0.286, 1.0) on the
gold-parameter reading and (0.613, 1.0) on cited calls (R1 C5).

**M4 payload-level sensitivity** (`diag_m4_payload.py`, A3, B=32): comparing returned arrival
times with [t_c, t_c+Δadm] over the cited reachability/paths payloads yields the gold label in
19 of 20 episodes. Evidence-valid pairs under that variant: gpt-5-mini 4/5 (5/5 if a bounded path
enumeration returning fewer than `max_paths` routes counts as exhaustive), Opus 5 5/5. Under the
reported anchored rule: 0/5 and 1/5. Under the submission-time width proxy: 0/5 and 3/5
(R1 C2, R3, R4 C2).

**Single-pair cells** (`paper_numbers.py` §9): Qwen2.5-14B A1 R=256 PairAcc 0.20 is one pair,
seed 1, whose positive run made 0 tool calls and is classified `fabricated_evidence` (cell
EVPairAcc 0.00). Opus A1 8K-node PairAcc 0.80 is one failed pair, seed 0, whose positive run
abstained (`insufficient`, no reconstruction attempt); all 10 runs in that cell used exactly 16
calls (R3).

**Sampling facts** (code audit): `configs/models.yaml` requests temperature 0; both hosted APIs
reject the parameter and `agents/backends.py` drops it after the first refusal, so Opus 5 and
gpt-5-mini ran at provider-default sampling. Qwen ran at temperature 0 via vLLM (Hermes tool
parser, 32,768 max model length, AWQ for all three sizes, one H100 each). 4,096 output tokens, tool
results truncated at 4,000 characters, turn cap B+4. Each world run once per sweep directory;
32 identities were executed twice across sweeps and some changed decision (R3, R4 C5).

### 2.3 Verified unchanged

Table 1 PairAcc rows and per-seed strings; reconstruction rates 0.90/0.90/0.90/0.90/1.00/0.80;
gpt-5-mini pooled reconstruction 0.13; Table 2 (all cells, NS-cite rates, zero protocol errors,
270 episodes); Table 3 (all 24 numbers); Table 4 (all numbers, including calls 32.0/26.9/12.1/
3.6/7.8/2.4); Table 5 (all numbers, including A1 calls 15.8/16.0/16.0/11.9/12.4/14.4); the M2/M3
budget-axis cells; the bootstrap contrast +0.75 [+0.55, +0.95]; the A0 enumeration counts; the
latency measurement 64.3 / 24.9 / 20.8 s; Figure 3(b) trace values (t_c 24382, interval
[24382, 28382], cascades at 25382 and 34382, query [18400, 36212]).

---

## 3. Prose edits, in paper order

### Abstract

**E1 (must; R2 W2, MR).** Current: "Across four temporal mechanisms, four interface levels and
five models, totaling 1,402 scored runs, the two dimensions separate sharply."
Replace "1,402 scored runs" with "1,382 scored episodes".

**E2 (should; R1 W3, R3).** Current: "Thus, final-answer capability can exceed the validity of
the replayable analytical support recorded in the execution trace."
Replace with: "Thus, final-answer capability can exceed the machine-checkable analytical support
recorded in the execution trace; on the single-stage mechanisms the remaining gaps are citation
omissions, and M4 is where executed support is absent."

**E3 (must; R3).** Current: "Additional nuisance activity and a four-fold increase in graph size
move a previously successful primitive-interface configuration back below the capability
frontier."
Replace with: "Additional nuisance activity and a four-fold increase in graph size degrade a
previously successful primitive-interface configuration (by two and one of five pairs)."

**E4 (must; R1 W1, R3, R4 W1, MR).** Current: "...whether their successful conclusions are
supported by valid analytical evidence."
Replace with: "...whether their successful conclusions are supported by replayable,
machine-checkable analytical evidence."

### §1 Introduction

**E5 (must; R4 C1).** Current: "Validity. When it succeeds, does the supporting computation
establish the intended claim?"
Replace with: "Validity. When it succeeds, does the supporting computation, as recorded in
replayable executed operations, establish the intended claim?"

**E6 (must; R3).** Current: "Within the Qwen2.5-Instruct-AWQ ladder, A1 performance is
non-monotone with checkpoint size, suggesting that adaptive reconstruction is a strategy regime
rather than a smooth function of parameter count."
Replace with: "Within the Qwen2.5-Instruct-AWQ ladder, no checkpoint enters the reconstruction
regime under A1, while A3 lifts the two larger checkpoints to 0.80–1.00: whether adaptive
reconstruction emerges is not a smooth function of parameter count."

### §2 Related Work

**E7 (must; R1 W1).** Current: "whether the cited computation is sufficient for the constrained
proposition the user requested"
Replace with: "whether the cited computation preserves the constraints of the proposition the
user requested"  (this makes §2 agree with §3.4, which states necessary conditions).

**E8 (should; R4 W6).** After "...requires accepted claims to descend from attested tool evidence
[10]." add: "Both describe or enforce provenance; neither measures, on controlled ground truth,
how often a correct answer lacks it. EVPairAcc is deterministic and replayable, needs no proof
kernel and no judge, and is evaluated where task truth is counterfactually controlled."

### §3.4 Evidence-valid analytical correctness

**E9 (must; R1 W1/C1, R3 W2, MR).** Current: "For A2 it verifies that the cited program
explicitly incorporates the task-relevant bound; correctness beyond the encoded constraint is
outside this checker."
Replace with: "For A2 it verifies, by literal inspection of the program text, that the cited
program contains the task-relevant bound; correctness beyond the encoded constraint is outside
this checker. For A0 and A1, whose calls carry no task-constraint parameters, SemValid reduces to
the structural conditions above (cited steps exist, cited windows lie inside the observable
horizon, a positive claim cites a support step). The checker is therefore strictest where the
interface exposes the constraint it tests, and the M4 anchoring rule below applies to A3
operators only."

**E10 (must; R1 W2, R3, R4 C2, MR).** In "Auditable evidence boundary", current: "Payload-level
natural-language derivations lie outside this boundary because they are not represented as
replayable task-constrained computations; M4 is constructed to expose exactly this case."
Append: "The excluded M4 inference, comparing returned arrival times with the admissible
interval, is itself deterministic; the boundary excludes it by design, so EVPairAcc audits
whether the cited operation encodes the composition, not whether the returned data would decide
it (§5.2.3 reports the sensitivity)."

### §4.3 Models

**E11 (must; R3, R4 C5).** Append to the paragraph: "Opus 5 and gpt-5-mini are accessed as
claude-opus-5 and gpt-5-mini through their vendor APIs, which reject a temperature parameter, so
both ran at provider-default sampling; the Qwen2.5-{7,14,32}B-Instruct-AWQ checkpoints are served
by vLLM with the Hermes tool parser at temperature 0. All conditions use 4,096 output tokens,
tool results truncated at 4,000 characters, and a turn cap of B+4. Each world is run once per
cell."

### §4.5 Controls

**E12 (must; R2 W2, R3).** Current: "The reported corpus contains 1,402 scored episodes."
Replace with: "The reported corpus contains 1,382 executed episodes (1,402 scored rows; 20 rows
are cached copies of the preregistered Opus anchor episodes reused by content hash in the
resolution sweep)."
Then append to the paragraph: "The M4 cells, metrics and the diagnostic of whether a run binds
the second stage to t_c were preregistered before any M4 episode; the specific anchored rule of
§3.4 (tolerance Δadm/4, absence coverage) was fixed during pre-submission review after the traces
had been read, replacing a width-only proxy under which Opus scored EVPairAcc 0.60 on M4
(gpt-5-mini, A1 and A2 are unaffected). The dated decision record is in the artifact."

### Table 1 and §5.1.3

**E13 (should; R1 C4, R2 C2).** Add a column "calls (at cap)" to Table 1 with the Opus A1 and
mini A1 values from §2.2 (e.g. "15.8 (8)" ... "16.0 (10)"; "12.6 (3)" ... "13.2 (6)"). Caption,
append: "Calls is the mean number of tool calls per A1 run, with the number of runs (of 10) that
used all 16 in parentheses. A2 and A3 were preregistered at representative R values rather than
the full grid (A2: 32, 256, 512; A3: 32, 128, 512), which is why cells are marked n/r."

**E14 (must; R1 W4/C4, R3).** Current: "Opus continues attempting reconstruction in 0.80–1.00
of runs even where its accuracy is zero, so what degrades is completion inside a fixed budget,
not the strategy itself."
Replace with: "Opus continues attempting localization in 0.80–1.00 of runs even where its
accuracy is zero, and it saturates the 16-call budget at every R, succeeding and failing alike
(Table 1); what degrades is what the attempted strategy accomplishes within a saturated budget,
not whether it is attempted."
Also in the same paragraph, "the persistent reconstruction attempts" → "the persistent
localization attempts".

### §5.1.6 and Table 2

**E15 (must; R3).** Heading, current: "A1 reconstruction is non-monotone across Qwen
checkpoints." Replace with: "No Qwen checkpoint enters the A1 reconstruction regime."
Body, current: "14B reaches only 0.20 in one R=256 cell, and none shows the sustained
reconstruction regime seen with Opus."
Replace with: "14B reaches 0.20 in one R=256 cell, a single pair whose positive run made no tool
call and cited steps it never executed (EVPairAcc 0.00), and none shows the sustained
reconstruction regime seen with Opus."

**E16 (should; R3).** Table 2 caption, append: "The single 14B A1 success at R=256 is a
zero-call run classified as fabricated evidence."

### §5.2.1 and Table 3

**E17 (must; R1 W1, R3 W2, MR).** Heading, current: "Measured cited-support gaps concentrate
under semantic operators." Replace with: "Cited-support gaps are measurable only under semantic
operators."
Current: "Under the checker, A1 and A2 have EVPairAcc = PairAcc in every evaluated cell and are
omitted."
Replace with: "A1 and A2 have EVPairAcc = PairAcc in every evaluated cell and are omitted; for A1
this is a property of the checker, which has no constraint parameter to test on windowed counts
(§3.4), and ExecEVPairAcc is undefined there."
Current: "On M4 the gaps persist: gpt-5-mini executes no task-faithful operation in any run"
Replace "task-faithful operation" with "operation anchored at t_c".

**E18 (must; R3 W2).** Table 3 caption, current: "A1 and A2 have EVPairAcc = PairAcc in every
evaluated cell and are omitted."
Replace with: "A1 and A2 have EVPairAcc = PairAcc in every evaluated cell and are omitted; for A1
the checker has no constraint parameter to test, so equality there reflects the checker's reach,
not validated support."

### §5.2.2

**E19 (should; R1 W5/C5, R3).** After "...following the returned contact lists rather than the
decisive statistic, while Opus remains at 1.00." add: "gpt-5-mini's labels are not reproducible
by any single threshold on the returned statistic: two negatives returned the identical score
0.286 and received opposite labels, and the lowest score (0.211) was labelled present, whereas
Opus's labels are consistent with any cut in (0.29, 1.0). The failure is model-specific: the
weaker model does not infer the well-separated criterion the stronger one applies."
Current: "Interpreting decisive operator output is therefore reliable for both models only when
the criterion is explicit, and the weaker model substitutes its own reading of auxiliary payload
when it is not."
Replace with: "Interpreting decisive operator output is therefore reliable for the weaker model
only when the criterion is explicit; it substitutes its own reading of auxiliary payload when it
is not."

### §5.2.3

**E20 (optional, recommended; R1 C2, R3, R4 C2, MR).** Append to the last paragraph: "The
excluded inference is deterministic: recomputed over the cited reachability payloads, a
comparison of returned arrival times with [t_c, t_c+Δadm] yields the gold label in 19 of 20 A3
episodes (gpt-5-mini 4 of 5 pairs, Opus 5 of 5; artifact script diag_m4_payload). The M4 gap
therefore measures whether the cited operation encodes the composition, not whether the returned
data could decide it."

### §5.2.4

**E21a (must).** Current: "6.2 calls and $0.298 per run, against 7.6/$0.414 for A1 and 8.3/$0.399
for A2" → "6.1 calls and $0.299 per run, against 7.7/$0.412 for A1 and 8.3/$0.399 for A2".
Current: "(17.6K vs. 25.9K under A3) and A3 the fewest for gpt-5-mini (9.4K vs. 11.8K under A2)"
→ "(17.6K vs. 26.1K under A3) and A3 the fewest for gpt-5-mini (9.3K vs. 11.8K under A2)".

**E21b (decision).** Current: "an empty run_python call accounts for 81.0 ms of the observed 84.7
ms, so the portable measure of backend work is events scanned, by which A2 is cheaper than A1."
No script in the artifact produces these two numbers. Keep the sentence only if the measurement
exists outside the artifact; otherwise replace with: "the fixed cost of an empty run_python call
dominates it, so the portable measure of backend work is events scanned, by which A2 is cheaper
than A1." (This is also the cheapest compensating cut for the added text.)

### §5.2.5 and Table 5

**E22 (must; R3).** Current: "In both degraded cells Opus consumes exactly 16.0 calls (the
budget ceiling) while its reconstruction rate stays high, at 0.70 with the burst and 0.90 at 8K."
Replace with: "Each degraded cell loses two (burst) or one (8K) of five pairs; the failed 8K pair
is an abstention with no reconstruction attempt. In both degraded cells Opus consumes exactly
16.0 calls on average (the budget ceiling; all ten 8K runs used 16) while its reconstruction rate
stays high, at 0.70 with the burst and 0.90 at 8K."
Current: "nuisance activity and graph size also move a previously solved setting into the failure
regime, while A3 stays at 1.00 in every tested cell."
Replace with: "nuisance activity and graph size also degrade a previously solved setting, by two
and one pairs of five, while A3 stays at 1.00 in every tested cell."

**E23 (should; R3).** Table 5 caption, append: "Each degraded Opus cell differs from baseline by
two (confounder) or one (scale) matched pairs."

### Figure captions

**E24 (must; R1 W4).** Figure 4 caption, current: "(b) Reconstruction rate over the same cells:
the detector fires on the attempt, never on success, and Opus keeps attempting at 0.80–1.00
precisely where its accuracy is zero."
Replace with: "(b) Reconstruction rate over the same cells: the detector fires on sustained
node-filtered localization (an attempt), never on success, and Opus keeps attempting at 0.80–1.00
precisely where its accuracy is zero." The panel title and annotation in the figure itself are
already regenerated ("but localization is still attempted"; "localization attempts persist where
accuracy is 0"). Figure 3(a) title is regenerated as "perfect answers, no anchored cited
operation"; its caption needs no change.

### §6 Discussion

**E25 (should; R1 C6, R3, MR).** Current: "...and A3 removes it differently, exposing the
failures abstraction cannot eliminate: an implicit decision criterion misapplied to decisive
evidence in the M3 ablation, correct conclusions on computations that cannot establish them in
M4."
Replace "an implicit decision criterion misapplied" with "a weaker model's implicit decision
criterion misapplied". Then append to the paragraph: "Auditability is itself interface-relative:
only an interface that exposes constraint parameters can be audited for them, which is why the
gaps in Table 3 are measurable only under A3."
Current: "Free-form payload inference is hard to replay and validate automatically, whereas..."
Replace with: "Free-form payload inference is hard to replay and validate automatically in
general, even when, as in M4, the particular inference is a deterministic comparison, whereas..."

### §7 Threats to Validity

**E26 (must; R1 W1, R3 W2, MR; R1 C6).** Measurement validity, append: "SemValid is also not
uniformly strict across interfaces: it compares operator arguments under A3, inspects program
text under A2, and has no constraint parameter to test under A0 and A1, so cross-interface
EVPairAcc comparisons are bounded by what each interface lets the checker see, and
ExecEVPairAcc is undefined for A0 and A1. The M4 criterion is a boundary choice: a payload-level
comparison would credit 19 of 20 A3 episodes (§5.2.3). A2 and A3 also differ in information
access and operator alignment, not only in abstraction (on M3 the A3 operator returns the target
statistic directly), so the abstraction claims are restricted to the tested regimes."

**E27 (must; R3, R4 C5).** Statistical and model scope, append: "Each world was run once. The
frontier models ran at provider-default sampling because their APIs reject a temperature
parameter, and thirty-two episodes that were re-executed in a second sweep sometimes changed
decision, so run-level sampling variance is real and unquantified here; with n=5 a single
stochastic outcome moves a cell by 0.20."

### §8 Conclusion

**E28 (must; R3, R4 W1).** Current: "Task accuracy, however, is not trace-grounded analytical
validity:" → "Task accuracy, however, is not machine-checkable analytical provenance:".
Current: "M3 shows that a model can receive decisive semantic evidence and misread it whenever
the decision criterion is not stated alongside it" → "M3 shows that a weaker model can receive
decisive semantic evidence and misread it when the decision criterion is not stated alongside
it".
Current: "while no operation it cites, or executes at all, encodes the required composition."
→ "while no operation it cites, or executes at all, encodes the required composition, although
the returned payloads would decide it."

### Page budget

E9, E11, E12, E19, E20, E26 and E27 add roughly a third of a column. Compensating cuts that lose
nothing: E21b (drop the two unverifiable latency numbers); in §5.1.3 the sentence "The
interaction-limited interpretation is supported jointly by the monotone accuracy decline, the
persistent localization attempts, and the analogous budget effect in M2." can go, since E14 and
§5.1.4 already say it; in §6 the sentence beginning "A2 suggests a complementary design" can be
shortened to its first clause.

---

## 4. Tables and figures to regenerate

- Table 1: add the calls column (E13); PairAcc, per-seed and recon. columns unchanged.
- Tables 2 to 5: numbers unchanged; captions per E16, E18, E23.
- Figure 1 (framework): unchanged.
- Figure 3 and Figure 4: regenerated on this branch (`figures/figure1.*`, `figures/figure2.*`);
  values unchanged, titles reworded. Copy the new PDFs into the LaTeX source.
- Figure 2 (mechanisms): unchanged.

---

## 5. Remaining author tasks (not doable from the artifact)

1. De-anonymise: author block, affiliations, ORCIDs, acknowledgements, funding; replace the
   anonymous artifact URL with the public repository and an archived DOI release; remove the
   "Anonymity" section of the README and the "anonymous artifact" title line.
2. Recover `runs/ladder_<slot>_controls.json` (vLLM, torch, GPU and driver versions) from the
   cluster job logs and commit it; if gone, keep the README sentence that says it is unrecorded
   and say "vLLM (version unrecorded)" in §4.3.
3. Confirm the WSDM 2027 findings-paper page budget and template; CCS concepts, ACM reference
   format, rights statement; update references that now have venues (e.g. AAAI 2026, ICML 2026
   items) and add DOIs; alt text for figures.
4. Decide E20 and E21b.
5. Run `python scripts/paper_numbers.py` once more after the LaTeX edits and tick every number
   in §2 against the PDF; have a co-author who did not write the edits read the abstract, §3.4
   and §7 cold.
6. Remove this file (or move it under `notes/`, which is gitignored) before the public release.

---

## 6. Deliberately not done

- No frontier-model episode re-run (not needed; preregistrations forbid post-hoc cells).
- No Qwen repeat-sampling experiment (would be new, post-hoc evidence; belongs to the next paper).
- No expert audit of traces (R1 C1): cheap, but needs a written sampling rule first; deferred.
- No change to the reported M4 criterion: the anchored rule stays the headline; the payload
  variant is reported as sensitivity only (E20).
- No title change.
