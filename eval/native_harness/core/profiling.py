"""Versioned sampling contract, shared by native configuration and evaluator."""


def validate_profiling(value):
    if not isinstance(value, dict) or set(value) != {"total_runs", "warmup_runs", "statistic"}:
        raise ValueError("profiling requires exactly total_runs, warmup_runs, statistic")
    total, warmup = value["total_runs"], value["warmup_runs"]
    if type(total) is not int or total < 1:
        raise ValueError("profiling.total_runs must be a positive integer")
    if type(warmup) is not int or not 0 <= warmup < total:
        raise ValueError("profiling.warmup_runs must be an integer in [0, total_runs)")
    if value["statistic"] != "mean":
        raise ValueError("profiling.statistic must be mean")
    return dict(value)


def default_profiling():
    return {"total_runs": 8, "warmup_runs": 3, "statistic": "mean"}
