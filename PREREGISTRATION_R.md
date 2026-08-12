# Preregistration — resolution × capability boundary test

**Committed before any R-sweep result was produced or inspected.** The ambiguity sweep and the
Opus anchors are complete and known; nothing at R = 32 or R = 128 has been run for either model.
Git history is the evidence.

## The question

At the anchors both models score A1 = 0.00, but for opposite reasons: gpt-5-mini abstains without
attempting reconstruction, Opus 5 attempts it in 60–100% of runs and exhausts its budget. That is a
behavioural difference, not yet a capability boundary.

The claim we want to test is stronger:

> There exists a regime where **Opus 5 succeeds through reconstruction and gpt-5-mini fails, under
> the identical interface and budget** — and where semantic operators (A3) succeed for both.

If such a regime exists, model capability changes what a low-level interface is *effectively*
capable of, and semantic operators erase that gap. If it does not, the honest claim stays at "the
same interface is used differently by different models", which is weaker and must be reported as
such.

## Cells

| Parameter | Value |
|---|---|
| mechanism | `cascade` |
| temporal resolution R | **32, 128, 512** |
| decisive events k (`hops`) | 5 |
| candidate ambiguity D | **0** |
| tool-call budget B | **16** |
| seeds / polarities | 0–4 / both |

**Primary (confirmatory):** condition **A1**, models **gpt-5-mini and Opus 5** — 3 R × 2 models ×
5 seeds × 2 polarities = 60 episodes, of which the 10 Opus episodes at R = 512 already exist from
the anchors and are reused by content hash.

**Secondary (flat-line check):** condition **A3**, same cells and models. This is my addition, not
in the original request, and is declared here rather than added later: the phase-boundary figure
needs a semantic-interface row, and Finding 2 ("semantic interfaces compress model differences")
requires A3 measured across R, not only at R = 512.

New Opus episodes: 20 (A1) + 20 (A3, R=512 reused) ≈ 40, forecast ≈ $15. New mini episodes: 60,
≈ $1.

R = 32 (span 2 400), R = 128 (span 9 600), R = 512 (span 38 399). The earlier Opus A1 success ran
at the default padding, span ≈ 4 950, i.e. R ≈ 66 — between the first two values, so the bracket
straddles the known success and the known failure.

## Predicted outcomes and what each would mean

| Pattern | Interpretation |
|---|---|
| Opus A1 declines with R (e.g. 1.0 / 0.6 / 0.0) while mini stays 0.00 | **Capability boundary.** The strongest available claim; H4 supported. |
| Both flat 0.00 at every R | No boundary in this range. Claim retreats to "same score, different mechanism"; report the behavioural difference only. |
| Both succeed at low R | A1 is expressive for both when search is cheap; the gap is budget-driven, not capability-driven. |
| mini succeeds where Opus fails | Would falsify the capability ordering; report as-is. |

A3 is expected flat near 1.00 across R for both models. A decline would mean semantic operators are
themselves resolution-sensitive, which would be a finding against our own framing and reported.

## Rules

* No cells added after results are seen. Any amendment appended below with date and reason.
* The protocol gate (`scripts/sanity_check_sweep.py`) must pass before the Opus half is admitted.
* mini runs first because it is ~$1; Opus runs only after the gate passes on the mini half.

## Amendments

**2026-08-11 — before running.** The `resolution` sweep as originally configured held
`n_decoy_paths: 6` and included R = 2048. Amended to `n_decoy_paths: 0` and R ∈ {32, 128, 512}.
Reasons, both independent of any R observation: the ambiguity sweep showed D has no effect on the
outcome (a negative result), and D = 0 makes these cells identical to the existing Opus anchors at
D = 0 so the R = 512 column is reused rather than repurchased. R = 2048 dropped because R = 512
already defeats reconstruction.
