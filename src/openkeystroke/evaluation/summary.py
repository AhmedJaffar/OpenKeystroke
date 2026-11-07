"""Aggregate per-seed evaluation rows into statistically interpretable summaries."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from ..config import ExperimentConfig
from ..defense.metrics import pareto_frontier
from .metrics import paired_comparison, summarize_seeds

KEYS = ("accuracy", "balanced_accuracy", "f1_macro", "ece")
MIN_SEEDS_FOR_INFERENCE = 3


def _g(rows: list[dict[str, Any]], *fields: str) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[tuple(r[f] for f in fields)].append(r)
    return out


def _by_seed(rs: list[dict[str, Any]], split: str = "test", key: str = "accuracy") -> dict[int, float]:
    return {r["seed"]: r[split][key] for r in rs}


def summarize_experiment(
    rows: list[dict[str, Any]],
    cfg: ExperimentConfig,
    defense_quality: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    conf = cfg.evaluation.confidence
    n_classes = len(dataset["class_names"])
    chance = 1.0 / n_classes
    seeds = sorted({r["seed"] for r in rows})
    caveats: list[str] = []
    if dataset["is_synthetic"]:
        caveats.append(
            "SYNTHETIC dataset: these numbers measure behaviour on controlled simulated transients and are not evidence of real-world effectiveness."
        )
    if len(seeds) < MIN_SEEDS_FOR_INFERENCE:
        caveats.append(
            f"only {len(seeds)} seed(s): no statistical inference is drawn (need >= {MIN_SEEDS_FOR_INFERENCE})."
        )
    if any(r["protocol"] == "in_distribution" for r in rows):
        caveats.append(
            "in_distribution results share sessions/devices between train and test; they are NOT evidence of cross-domain generalisation."
        )
    caveats.append(
        "Reported metrics are on held-out test partitions only; model selection used the separate validation partition."
    )

    # ---- main table
    table = []
    for (variant, protocol, model, condition, attacker), rs in sorted(
        _g(rows, "variant", "protocol", "model", "condition", "attacker").items()
    ):
        ent: dict[str, Any] = {
            "variant": variant,
            "protocol": protocol,
            "model": model,
            "condition": condition,
            "attacker": attacker,
            "n_seeds": len({r["seed"] for r in rs}),
            "chance": chance,
        }
        for k in KEYS:
            ent[k] = summarize_seeds([r["test"][k] for r in rs], conf)
        ent["val_accuracy"] = summarize_seeds([r["val"]["accuracy"] for r in rs], conf)
        ci = [r["test_ci"].get("accuracy") for r in rs if r.get("test_ci")]
        if ci:
            ent["accuracy_within_seed_ci_mean"] = {
                "lo": float(np.mean([c["lo"] for c in ci])),
                "hi": float(np.mean([c["hi"] for c in ci])),
                "method": ci[0]["method"],
            }
        table.append(ent)

    # ---- degradation vs clean baseline (paired by seed)
    degradation = []
    for (variant, protocol, model, condition, attacker), rs in sorted(
        _g(rows, "variant", "protocol", "model", "condition", "attacker").items()
    ):
        if condition == "clean":
            continue
        base_rows = [
            r
            for r in rows
            if (r["variant"], r["protocol"], r["model"], r["condition"], r["attacker"])
            == (variant, protocol, model, "clean", "static")
        ]
        if not base_rows:
            continue
        b, a = _by_seed(base_rows), _by_seed(rs)
        common = sorted(set(a) & set(b))
        base_v, cond_v = [b[s] for s in common], [a[s] for s in common]
        cmp = paired_comparison(base_v, cond_v)
        n = len(common)
        if n < MIN_SEEDS_FOR_INFERENCE:
            evidence = "insufficient_seeds"
        elif cmp["p_value"] is not None and cmp["p_value"] < 0.05 and cmp["mean_diff"] > 0:
            evidence = "significant_reduction"
        else:
            evidence = "no_significant_reduction"
        degradation.append(
            {
                "variant": variant,
                "protocol": protocol,
                "model": model,
                "condition": condition,
                "attacker": attacker,
                "n_seeds": n,
                "baseline_accuracy": float(np.mean(base_v)),
                "defended_accuracy": float(np.mean(cond_v)),
                "accuracy_drop": cmp["mean_diff"],
                "relative_drop": float(cmp["mean_diff"] / max(np.mean(base_v), 1e-9)),
                "paired_test": cmp,
                "defended_vs_chance": float(np.mean(cond_v) - chance),
                "evidence": evidence,
            }
        )

    # ---- defense trade-off (Pareto): privacy gain = accuracy drop of the strongest available attacker
    defense_points, curves = [], defaultdict(list)
    dcfg = {d.name: d for d in cfg.defenses}
    by_def: dict[tuple, dict[str, Any]] = {}
    for d in degradation:
        by_def.setdefault((d["variant"], d["protocol"], d["model"], d["condition"]), {})[d["attacker"]] = d
    pareto_by_group: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for (variant, protocol, model, condition), att in by_def.items():
        q = defense_quality.get(condition, {})
        gain_static = att.get("static", {}).get("accuracy_drop")
        gain_adapt = att.get("adaptive", {}).get("accuracy_drop")
        gains = [g for g in (gain_static, gain_adapt) if g is not None]
        pt = {
            "name": condition,
            "variant": variant,
            "protocol": protocol,
            "model": model,
            "gain_static": gain_static,
            "gain_adaptive": gain_adapt,
            "privacy_gain": min(gains) if gains else None,
            "log_spectral_distance_db": q.get("log_spectral_distance_db"),
            "waveform_snr_db": q.get("waveform_snr_db"),
            "real_time_factor": q.get("real_time_factor"),
            "latency_ms_per_sample": q.get("latency_ms_per_sample"),
            "evidence": {k: v["evidence"] for k, v in att.items()},
        }
        # only defenses with a positive worst-case gain and no attacker showing "no significant reduction" may be called effective
        pt["eligible"] = bool(
            pt["privacy_gain"] is not None
            and pt["privacy_gain"] > 0
            and "no_significant_reduction" not in pt["evidence"].values()
        )
        pareto_by_group[(variant, protocol, model)].append(pt)
        defense_points.append(pt)
        step = dcfg[condition].steps[0] if condition in dcfg else None
        if step is not None:
            pname = sorted(step.params)[0] if step.params else None
            curves[(variant, protocol, model, step.type, pname)].append(
                {
                    "x": step.params[pname] if pname else None,
                    "condition": condition,
                    "static_accuracy": att.get("static", {}).get("defended_accuracy"),
                    "adaptive_accuracy": att.get("adaptive", {}).get("defended_accuracy"),
                    "baseline_accuracy": next(iter(att.values()))["baseline_accuracy"],
                }
            )
    pareto = []
    for (variant, protocol, model), pts in pareto_by_group.items():
        eligible = [p for p in pts if p["eligible"]]
        front = pareto_frontier(eligible, "privacy_gain", "log_spectral_distance_db")
        pareto.append(
            {
                "variant": variant,
                "protocol": protocol,
                "model": model,
                "frontier": front,
                "n_candidates": len(pts),
                "n_eligible": len(eligible),
                "objective": "maximise worst-case accuracy drop (min over attackers) while minimising log-spectral distance",
            }
        )
    effectiveness_curves = [
        {
            "variant": k[0],
            "protocol": k[1],
            "model": k[2],
            "defense_type": k[3],
            "parameter": k[4],
            "points": sorted(v, key=lambda p: (p["x"] is None, p["x"])),
        }
        for k, v in curves.items()
    ]

    # ---- domain shift (clean/static): protocol vs in_distribution
    shift = []
    clean = [r for r in rows if r["condition"] == "clean" and r["attacker"] == "static"]
    for (variant, model), rs in _g(clean, "variant", "model").items():
        per = _g(rs, "protocol")
        accs = {p[0]: summarize_seeds([r["test"]["accuracy"] for r in v], conf) for p, v in per.items()}
        if ("in_distribution",) in per:
            base = _by_seed(per[("in_distribution",)])
            for (proto,), v in per.items():
                if proto == "in_distribution":
                    continue
                common = sorted(set(base) & set(_by_seed(v)))
                cmp = paired_comparison([base[s] for s in common], [_by_seed(v)[s] for s in common])
                shift.append(
                    {
                        "variant": variant,
                        "model": model,
                        "protocol": proto,
                        "in_distribution_accuracy": accs["in_distribution"]["mean"],
                        "shifted_accuracy": accs[proto]["mean"],
                        "degradation": cmp["mean_diff"],
                        "paired_test": cmp,
                        "note": "test partitions differ between protocols; the difference mixes domain shift with split composition",
                    }
                )
        else:
            for proto, acc in accs.items():
                shift.append(
                    {
                        "variant": variant,
                        "model": model,
                        "protocol": proto,
                        "shifted_accuracy": acc["mean"],
                        "degradation": None,
                        "note": "no in_distribution run to compare against",
                    }
                )

    # ---- model comparison (clean/static), paired per seed
    comparisons = []
    for (variant, protocol), rs in _g(clean, "variant", "protocol").items():
        models = _g(rs, "model")
        for ma, mb in combinations(sorted(m[0] for m in models), 2):
            sa, sb = _by_seed(models[(ma,)]), _by_seed(models[(mb,)])
            common = sorted(set(sa) & set(sb))
            c = paired_comparison([sa[s] for s in common], [sb[s] for s in common])
            comparisons.append(
                {
                    "variant": variant,
                    "protocol": protocol,
                    "model_a": ma,
                    "model_b": mb,
                    **c,
                    "significant_at_0.05": bool(
                        c.get("p_value") is not None
                        and c["p_value"] < 0.05
                        and len(common) >= MIN_SEEDS_FOR_INFERENCE
                    ),
                }
            )

    # ---- selection on validation partition
    selection = []
    variants = sorted({r["variant"] for r in rows})
    if len(variants) > 1:
        for (protocol, model), rs in _g(clean, "protocol", "model").items():
            scored = [
                (summarize_seeds([r["val"]["accuracy"] for r in v], conf)["mean"], variant, v)
                for (variant,), v in _g(rs, "variant").items()
            ]
            val, best, vr = max(scored, key=lambda t: (t[0], t[1]))
            selection.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "selected_variant": best,
                    "selected_by": "validation_accuracy",
                    "validation_accuracy": val,
                    "test_accuracy": summarize_seeds([r["test"]["accuracy"] for r in vr], conf),
                    "n_variants": len(scored),
                }
            )

    # ---- resources
    resources = []
    for (variant, protocol, model), rs in _g(clean, "variant", "protocol", "model").items():
        res = [r["resource"] for r in rs if r.get("resource")]
        if res:
            resources.append(
                {
                    "variant": variant,
                    "protocol": protocol,
                    "model": model,
                    **{
                        k: float(np.median([x[k] for x in res]))
                        for k in res[0]
                        if isinstance(res[0][k], (int, float))
                    },
                }
            )

    ranking = sorted(
        [t for t in table if t["condition"] == "clean" and t["attacker"] == "static"],
        key=lambda t: -t["accuracy"]["mean"],
    )
    return {
        "dataset": {
            k: dataset[k] for k in ("dataset_id", "version", "fingerprint", "is_synthetic", "class_names")
        },
        "seeds": seeds,
        "chance_accuracy": chance,
        "table": table,
        "ranking_clean_static": [
            (t["variant"], t["protocol"], t["model"], t["accuracy"]["mean"]) for t in ranking
        ],
        "degradation": degradation,
        "defense_points": defense_points,
        "pareto": pareto,
        "effectiveness_curves": effectiveness_curves,
        "domain_shift": shift,
        "model_comparisons": comparisons,
        "selection": selection,
        "resources": resources,
        "caveats": caveats,
    }
