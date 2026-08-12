# Preregistration — C4 open-weight capability ladder

**Committed before any open-weight episode is run.** Cells, models, and the analysis are fixed
here; the results are read afterwards. Same discipline as the earlier preregistrations, and for
the same reason: an open-weight ladder has many defensible configurations, and choosing among them
after seeing which makes the story cleanest would not be a measurement.

## Question

Does the capability profile — strategy discovery, the A1 frontier, and A3's compression of it —
vary systematically with model capability, or is it an artifact of comparing two proprietary
models?

## Subjects

One family, three sizes, **one quantization**, so scale is the only variable that moves:

| slot | checkpoint | role |
|---|---|---|
| `qwen7b` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | small |
| `qwen14b` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | medium |
| `qwen32b` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | large |

Mixing an fp16 small model with AWQ large ones would confound scale with precision, so all three
are AWQ. No cross-family checkpoint is added: the milestone permits at most one and explicitly
warns against a model zoo, and one more family would not answer the scale question.

Served by vLLM through an OpenAI-compatible endpoint with Hermes-style tool parsing, so the agent
loop, prompts, tools, and scoring are byte-identical to the hosted subjects. **The comparison is
between models, not between harnesses.**

## Cells (verbatim from milestone C4)

| sweep | mechanism | interface | axis | values | B |
|---|---|---|---|---|---|
| `ladder_m1_a1` | M1 cascade | A1 | R | 32, 128, 256, 512 | 16 |
| `ladder_m1_a3` | M1 cascade | A3 | R | 32, 512 | 16 |
| `ladder_m2_a1` | M2 closure wave | A1 | B | — | 8, 32 |
| `ladder_m2_a3` | M2 closure wave | A3 | B | — | 16 |

5 seeds × 2 polarities throughout. 90 episodes per model, 270 total. All local GPU, no API cost;
the binding constraint is wall-clock, not money.

Held at the same values as the corresponding hosted-model cells (`hops: 5`, `n_decoy_paths: 0` on
M1), so the ladder columns drop directly beside the existing Opus and mini columns without
re-running anything.

## Required controls, recorded per checkpoint

Exact checkpoint, inference engine and version, quantization, context limit, sampling
configuration, chat/tool template, GPU configuration. Written to the run manifest, not prose.

## The measurement that must not be conflated

**Tool-protocol failure is not analytical failure.** An open-weight model that cannot emit a
well-formed tool call has not been measured on temporal reasoning at all, and scoring it as a
wrong answer would silently convert a formatting deficiency into a scientific claim about
reasoning. Implemented before this run: malformed emissions are counted as `tool_protocol_errors`,
the model is told what was wrong and allowed to retry, and the retry is **not** charged to the
tool budget, which measures analysis rather than syntax. `PairAcc` is reported alongside the
protocol-failure rate, and any cell whose protocol-failure rate exceeds 20% is reported as
*protocol-limited* rather than compared on accuracy.

## Predictions (recorded before the run)

1. A1 PairAcc increases with model size at low R and stays at 0.00 for all three at R=512.
2. A3 compresses the differences: all three closer together, and higher, than under A1.
3. Reconstruction/strategy-discovery rate increases with size — the small model mostly does not
   attempt reconstruction, matching mini rather than Opus.

## Outcomes, all publishable

* **Monotone ladder** — the profile tracks capability; the strongest version of C4.
* **Non-monotone** — reported honestly as such. The claim is *model-dependent interface
  capability*, never parameter-count scaling.
* **Floor effect** — if all three sit at 0.00 under A1, that bounds the frontier below Opus and is
  itself informative, provided protocol failures are low enough for the cells to be interpretable.
* **Protocol-limited** — if the small model cannot hold the tool contract, that is reported as an
  interface-usability finding and its accuracy cells are withheld, not spun.

## Rules

* No cell, seed, model, or quantization added after results are seen.
* Failure to reach a prediction is reported, not re-run with different settings.
* Results update the *scope* sentences of `PAPER_CLAIMS.md`; the three findings are frozen.
* Smallest model runs first, as a protocol shakedown, before the larger two are launched.
