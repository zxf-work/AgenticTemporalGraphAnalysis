# Paper build

`main.tex` is a WSDM-format (ACM `sigconf`) draft. Every number in it is emitted by

```
python scripts/paper_numbers.py
```

and both figures by `scripts/figure1.py` / `scripts/figure2.py`, all of which read
`runs/*/results.jsonl` directly. **If a number appears in the LaTeX and not in that output, it is
unverified.** Regenerate figures into `paper/figures/` before a final build.

## Building

No TeX installation exists on the project machines (`compute-host`, `gpu-cluster`), so this draft has **not
been compiled**. It has been checked structurally — balanced environments, no dangling `\ref`,
matching `tabular` column counts, resolvable figure paths — but that is not a substitute for a
compile pass.

Two options:

* **Overleaf** — upload `main.tex` plus `figures/`, select a TeX Live 2021+ compiler. `acmart` is
  available there by default.
* **Local** — `acmart.cls` must be installed (`tlmgr install acmart`), then `latexmk -pdf main`.

## Known gaps before submission

1. **No bibliography.** Related Work is written descriptively with no `\cite` commands and no
   `.bib` file. This is the largest remaining hole: the positioning arguments are in place but the
   citations that support them are not. Roughly 25–35 references are needed across agent-interface
   design, graph/temporal-graph agent benchmarks, process-level and trajectory evaluation, and
   typed temporal systems.
2. **Anonymisation.** The class is loaded with `anonymous,review`; drop both for camera-ready.
3. **Two robustness cells not yet run** — one global-burst confounder and one 8K-node cell. The
   Threats section currently states their absence honestly; if they run, update that section and
   the corpus count.
4. **Length.** ~4{,}830 words plus 6 tables and 2 figures, which lands near but probably under the
   9-page limit. Related Work is the natural place to expand once citations exist.
