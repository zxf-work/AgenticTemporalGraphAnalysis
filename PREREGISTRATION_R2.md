# Preregistration — fine bracket for the capability boundary

**Committed before any R ∈ {192, 256, 384} episode was run for either model.** The coarse curve
(R = 32, 128, 512) is complete and known. Nothing between 128 and 512 exists yet.

## The question

The coarse sweep showed Opus 5 A1 at 1.00 / 1.00 / 0.00 for R = 32 / 128 / 512, with gpt-5-mini
flat at 0.00 and A3 flat at 1.00. So a boundary exists somewhere in (128, 512]. This experiment
converts "there is a boundary" into "the boundary lies near R ≈ X".

## Cells

| Parameter | Value |
|---|---|
| mechanism | `cascade` |
| temporal resolution R | **192, 256, 384** (spans 14 400, 19 200, 28 800) |
| decisive events k / ambiguity D / budget B | 5 / 0 / 16 — identical to the coarse columns |
| condition | **A1 only** |
| models | **Opus 5 and gpt-5-mini** |
| seeds / polarities | **0–4** / both |

Deliberately matching the coarse columns' seeds and budget so all six R values form one uniform
curve rather than a spliced one.

**A3 is excluded.** It is flat at 1.00 across a 16× span change for both models; three more cells
would re-measure a constant. **mini is included** despite being flat at 0.00, because the figure's
claim is a *contrast* and a gap in one row would invite the reader to wonder what was omitted; at
~$1.20 it is cheap completeness rather than a hypothesis test.

Episodes: Opus 3 × 5 × 2 = 30 (~$13); mini 30 (~$1.20).

## Declared analysis — before seeing the curve

To prevent defining the boundary after seeing where it would be most flattering:

* **Boundary location** := the midpoint (geometric mean) between the largest R with pair accuracy
  ≥ 0.8 and the smallest R with pair accuracy ≤ 0.2, over the pooled six-value curve.
* **Sharp** := those two R values are adjacent in the tested set. **Gradual** := at least one tested
  R falls strictly between 0.2 and 0.8.
* Report per-seed outcomes (5 binary pair results per cell), not only the mean — at n = 5 the mean
  is a coarse statistic and the raw counts carry the information.
* Paired bootstrap CIs over the pooled curve, 10 000 resamples.

## Predicted outcomes and what each would mean

| Pattern | Interpretation |
|---|---|
| Monotone decline, one intermediate value strictly between 0.2 and 0.8 | Gradual degradation as search cost rises; boundary located, reconstruction fails progressively. |
| 1.00 at 192/256 then 0.00 at 384 (or similar step) | Sharp threshold; reconstruction is all-or-nothing within a budget. |
| Non-monotone (e.g. 1.0, 0.4, 1.0, 0.0) | The axis is noisier than n = 5 resolves; report as such and do **not** add seeds to smooth it. |
| All three at 1.00 | Boundary sits between 384 and 512, narrower than expected; report the narrowed interval. |
| All three at 0.00 | Boundary sits between 128 and 192; report the narrowed interval. |

## Rules

* **No post-hoc seed addition.** If the curve is noisy at n = 5, that is the reported result. Adding
  seeds only where the curve looks ragged would be optimising precision against the outcome.
* No R values added after seeing these three.
* The protocol gate must pass before the Opus half is admitted; mini runs first.

## Amendments

None.
