"""Turn one trace into one fully-populated RunRow.

Single place where a trace becomes a scored row, so every runner reports the same fields and no
metric is silently missing from one experiment and present in another.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluation.checker import check
from evaluation.error_taxonomy import classify, parameterisation_report, reconstruction_stats
from evaluation.metrics import RunRow
from tasks.schema import TaskRecord
from traces.schema import Trace


def score(trace: Trace, rec: TaskRecord, condition: str,
          compound_tc: int | None = None) -> RunRow:
    """``compound_tc`` is the oracle closure-completion time for this world (compound only);
    when provided, M4 evidence citations are judged by the anchored claim-sensitive rule."""
    res = check(trace, rec.gold, condition=condition, horizon=rec.task.horizon)
    par = parameterisation_report(trace, rec.task, rec.mechanism, compound_tc=compound_tc)
    rec_stats = reconstruction_stats(trace)
    cat = classify(trace, rec.task, rec.mechanism, res.decision_correct, res.evidence_grounded)

    return RunRow(
        pair_id=rec.pair_id, polarity=rec.polarity, gold=rec.gold, condition=condition,
        model=trace.model, mechanism=rec.mechanism, confounder=rec.confounder,
        decision=(trace.final_claim or {}).get("decision", "none"), check=res,
        tool_calls=len(trace.steps), input_tokens=trace.total_input_tokens,
        output_tokens=trace.total_output_tokens, latency_ms=trace.total_latency_ms,
        status=trace.status,
        tool_protocol_errors=trace.tool_protocol_errors,
        rows_scanned=sum(s.rows_scanned for s in trace.steps),
        backend_ms=sum(s.latency_ms for s in trace.steps),
        llm_calls=trace.llm_calls,
        param_error_rate=par.error_rate, cited_bad_support=par.cited_bad_support,
        decomposition_rate=par.decomposition_rate,
        n_temporal_calls=par.n_temporal_calls,
        used_reconstruction=bool(rec_stats["used_reconstruction"]),
        range_calls=int(rec_stats["range_calls"]),
        windows_per_call=float(rec_stats["mean_windows_per_call"]),
        min_window_width=rec_stats["min_window_width"],
        failure_category=cat,
    )
