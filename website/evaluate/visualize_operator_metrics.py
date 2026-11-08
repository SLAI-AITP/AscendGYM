#!/usr/bin/env python3
"""Export operator evaluation summary into a CSV file.

Usage:
    python evaluate/visualize_operator_metrics.py --input operator/eval_summary.json --output-dir operator/export
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compute_summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in summary.get("runs", []):
        accuracy = run.get("accuracy", {})
        cost = run.get("cost", {})
        sts = run.get("sts", {})
        paos = run.get("paos", {})
        profiling = paos.get("profiling", {})
        baseline_profile = profiling.get("baseline", {})
        agent_profile = profiling.get("agent", {})
        rows.append({
            "task_id": run.get("task_id"),
            "status": run.get("status", "unknown"),
            "TaskPass": accuracy.get("TaskPass"),
            "CandidatePassRate": accuracy.get("CandidatePassRate"),
            "CorrectnessGate": accuracy.get("CorrectnessGate"),
            "RunCompleted": accuracy.get("RunCompleted"),
            "FailureReason": accuracy.get("FailureReason"),
            "CandidatesTotal": accuracy.get("CandidatesTotal"),
            "CandidatesPassed": accuracy.get("CandidatesPassed"),
            "BaselineEvidence": accuracy.get("BaselineEvidence"),
            "TotalTokens": cost.get("TotalTokens"),
            "TotalWallClockSec": cost.get("TotalWallClockSec"),
            "SearchWallClockSec": cost.get("SearchWallClockSec"),
            "LLMRounds": cost.get("LLMRounds"),
            "ToolCalls": cost.get("ToolCalls"),
            "STS_status": sts.get("status"),
            "STS_reason": sts.get("reason"),
            "Speedup": sts.get("Speedup"),
            "VLR": sts.get("VLR"),
            "AULS_token": sts.get("AULS_token"),
            "AULS_time": sts.get("AULS_time"),
            "SpeedupSearch": sts.get("SpeedupSearch"),
            "VLRSearch": sts.get("VLRSearch"),
            "STS_token": sts.get("STS_token"),
            "STS_time": sts.get("STS_time"),
            "S_equiv": sts.get("S_equiv"),
            "VerificationPassRate": sts.get("VerificationPassRate"),
            "VerificationMedianMs": sts.get("VerificationMedianMs"),
            "VerificationCV": sts.get("VerificationCV"),
            "SpeedupConfirmed": sts.get("SpeedupConfirmed"),
            "VLRConfirmed": sts.get("VLRConfirmed"),
            "TokensTo1%": sts.get("TokensTo1%"),
            "TimeTo1%": sts.get("TimeTo1%"),
            "TokensTo5%": sts.get("TokensTo5%"),
            "TimeTo5%": sts.get("TimeTo5%"),
            "TokensTo10%": sts.get("TokensTo10%"),
            "TimeTo10%": sts.get("TimeTo10%"),
            "PAOS_status": paos.get("status"),
            "PAOS_reason": paos.get("reason"),
            "G": paos.get("G"),
            "runtime_scenario": paos.get("runtime_scenario"),
            "batch_size": paos.get("batch_size"),
            "chunk_len": paos.get("chunk_len"),
            "T_other_ms": paos.get("T_other_ms"),
            "calls_per_step": paos.get("calls_per_step"),
            "T_step_base_ms": paos.get("T_step_base_ms"),
            "T_step_agent_ms": paos.get("T_step_agent_ms"),
            "eTPS_base": paos.get("eTPS_base"),
            "eTPS_agent": paos.get("eTPS_agent"),
            "S_i_lat": paos.get("S_i_lat"),
            "PTEI": paos.get("PTEI"),
            "PKAI": paos.get("PKAI"),
            "PAOS_base": paos.get("PAOS_base"),
            "PAOS_recipe": paos.get("recipe"),
            "PAOS_input_source": paos.get("input_source"),
            "profiling_baseline_status": baseline_profile.get("status"),
            "profiling_baseline_active_time_us": baseline_profile.get("ActiveTimeUs"),
            "profiling_baseline_source": baseline_profile.get("source"),
            "profiling_agent_status": agent_profile.get("status"),
            "profiling_agent_active_time_us": agent_profile.get("ActiveTimeUs"),
            "profiling_agent_source": agent_profile.get("source"),
            "S_i_aicore": profiling.get("S_i_aicore"),
            "U_i": profiling.get("U_i"),
            "U_step": profiling.get("U_step"),
            "VecRatio": profiling.get("VecRatio"),
            "CubeRatio": profiling.get("CubeRatio"),
            "MTE2Ratio": profiling.get("MTE2Ratio"),
        })
    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path):
    fieldnames = list(rows[0].keys()) if rows else []

    def csv_value(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return value

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k)) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Export operator evaluation summary JSON into CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Path to eval_summary.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Folder for the CSV output")
    args = parser.parse_args()

    summary = load_json(args.input)
    rows = compute_summary_rows(summary)
    out_dir = args.output_dir or args.input.parent / "export"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "operator_metrics.csv"
    write_csv(rows, csv_path)

    print(f"CSV exported: {csv_path}")


if __name__ == "__main__":
    main()
