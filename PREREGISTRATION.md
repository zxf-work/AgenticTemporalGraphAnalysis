# Preregistration — strong-model anchor cells

**Committed before the corrected gpt-5-mini ambiguity sweep was read.** The sweep was running at
the time of this commit; its `results.jsonl` had not been inspected, summarised, or plotted. The
git history is the evidence: this file's commit precedes any commit or transcript containing the
sweep's numbers.

## Why preregister these

Budget forces a division of labour between two different scientific jobs:

* **Dense characterisation** — how the interfaces behave across the whole information-complexity
  grid. This needs many cells and tolerates a cheaper subject; `gpt-5-mini` runs the full grid.
* **Cross-model validation** — whether the effect is a property of the interfaces or an artifact of
  one model's habits. This needs a frontier subject but only a few, fixed points.

Replicating every cell at frontier cost would spend hundreds of dollars to answer the second
question with the first question's design. Separating them is the better experiment, not merely the
affordable one.

The risk in a small validation set is that the cells get chosen after the fact, where the cheap
model's results look most favourable. Declaring them in advance removes that possibility, and
removes the reviewer's grounds for suspecting it.

## The anchor cells

Fixed at the centre of the other two axes, varying only candidate ambiguity:

| Parameter | Value |
|---|---|
| mechanism | `cascade` |
| temporal resolution R | 512 |
| decisive events k (`hops`) | 5 |
| candidate ambiguity D (`n_decoy_paths`) | **0, 16, 64** |
| tool-call budget B | **16** |
| conditions | **A1, A3** at every D; **A0 at D=16 only** |
| seeds | 0–4 (5 pairs) |
| polarities | positive + negative |

Episode count: `3 D x 2 conditions x 5 seeds x 2 polarities = 60`, plus `1 x 5 x 2 = 10` for A0 at
the centre anchor = **70 Opus 5 episodes**, forecast ≈ $25.

D = {0, 16, 64} spans the ambiguity axis endpoints and its midpoint; B = 16 is the middle of the
declared budget ladder {4, 8, 16, 32}; R = 512 and k = 5 are the centre values the other sweeps
already hold. A0 appears only at the centre anchor because its role is an identifiability control,
not a competitor — one confirmation that it sits on its floor is sufficient, and spending frontier
budget on it at all three D values would buy nothing.

## Commands

```bash
python scripts/run_matrix.py anchors --models primary --conditions A1,A3 --budget 40
python scripts/run_matrix.py anchors --models primary --conditions A0 \
       --budgets 16 --out runs/anchors_a0 --budget 10   # centre anchor only; D filtered in analysis
```

## What would falsify the cross-model claim

If Opus 5 at these anchors shows the *same* A1/A3 ordering and a similar cost gap as gpt-5-mini,
the interface effect is model-general. If the ordering inverts or collapses at any anchor, the
effect is capability-dependent — which is H4, and is reported as such rather than explained away.
Either outcome is publishable; neither licenses adding anchors afterwards.

## Amendments

None. Any change to the cells above must be appended here with its date and reason, and the
original left in place.
