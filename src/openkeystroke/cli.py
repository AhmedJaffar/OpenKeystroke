"""openkeystroke command-line interface. Exit codes: 0 ok, 1 runtime failure, 2 invalid input, 3 not found, 4 cancelled."""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import click

from . import __version__
from .config import SyntheticConfig, load_config
from .errors import NotFoundError, OpenKeystrokeError

_EXPERIMENT_EXIT = {"COMPLETED": 0, "CANCELLED": 4, "FAILED": 1}


def _emit(ctx: click.Context, obj: Any, text: str | None = None) -> None:
    if ctx.obj["json"]:
        click.echo(json.dumps(obj, indent=1, default=str))
    elif text is not None:
        click.echo(text)


def _metric_key(m: dict[str, Any]) -> tuple:
    return (
        m["variant"],
        m["protocol"],
        m["model"],
        m["condition"],
        m["attacker"],
        m["seed"],
        m["split"],
        m["name"],
    )


def _handled(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        ctx = click.get_current_context()
        try:
            return fn(*a, **kw)
        except OpenKeystrokeError as exc:
            if ctx.obj and ctx.obj.get("json"):
                click.echo(json.dumps(exc.to_dict()), err=True)
            else:
                click.echo(f"error[{exc.code}]: {exc.message}", err=True)
            ctx.exit(exc.exit_code)

    return wrapper


def _ws(ctx: click.Context):
    from .registry import Workspace

    return Workspace(ctx.obj["home"])


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--home",
    type=click.Path(file_okay=False),
    default=None,
    envvar="OPENKEYSTROKE_HOME",
    help="Workspace directory (registry, artifacts, datasets, cache). Default: ./.openkeystroke",
)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON output (and JSON errors).")
@click.version_option(__version__)
@click.pass_context
def main(ctx: click.Context, home: str | None, as_json: bool) -> None:
    """OpenKeystroke: research platform for authorized acoustic side-channel experiments and privacy-defense evaluation.

    Intended for controlled laboratory data and consented recordings only."""
    ctx.obj = {"home": home, "json": as_json}


# ----------------------------------------------------------------------------- dataset
@main.group()
def dataset() -> None:
    """Generate, ingest, validate and inspect datasets."""


@dataset.command("generate")
@click.option("--out", required=True, type=click.Path(file_okay=False), help="Output dataset directory.")
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--classes", "n_classes", default=6, show_default=True, type=click.IntRange(2, 26))
@click.option("--devices", "n_devices", default=4, show_default=True, type=click.IntRange(1, 16))
@click.option("--sessions", "sessions_per_device", default=3, show_default=True, type=click.IntRange(1, 16))
@click.option(
    "--per-class", "samples_per_class_per_session", default=8, show_default=True, type=click.IntRange(1, 500)
)
@click.option("--sample-rate", default=16000, show_default=True, type=int)
@click.option(
    "--config",
    "cfg_file",
    type=click.Path(exists=True, dir_okay=False),
    help="YAML/JSON SyntheticConfig (overrides flags).",
)
@click.option("--overwrite", is_flag=True)
@click.pass_context
@_handled
def dataset_generate(
    ctx,
    out,
    seed,
    n_classes,
    n_devices,
    sessions_per_device,
    samples_per_class_per_session,
    sample_rate,
    cfg_file,
    overwrite,
):
    """Generate a deterministic SYNTHETIC acoustic-transient dataset."""
    import yaml

    from .data import generate_synthetic

    try:
        cfg = (
            SyntheticConfig.model_validate(yaml.safe_load(Path(cfg_file).read_text()))
            if cfg_file
            else SyntheticConfig(
                seed=seed,
                n_classes=n_classes,
                n_devices=n_devices,
                sessions_per_device=sessions_per_device,
                samples_per_class_per_session=samples_per_class_per_session,
                sample_rate=sample_rate,
            )
        )
    except Exception as exc:
        from .errors import ConfigError

        raise ConfigError(f"invalid synthetic config: {exc}") from exc
    ds = generate_synthetic(cfg)
    ds.save(out, overwrite=overwrite)
    info = {
        "path": str(Path(out).resolve()),
        "dataset_id": ds.manifest.dataset_id,
        "n_samples": len(ds),
        "fingerprint": ds.fingerprint,
        "is_synthetic": True,
    }
    _emit(ctx, info, f"generated {len(ds)} synthetic samples -> {out}\nfingerprint {ds.fingerprint}")


@dataset.command("ingest")
@click.argument("manifest", type=click.Path(dir_okay=False))
@click.option("--out", required=True, type=click.Path(file_okay=False))
@click.option("--sample-rate", default=16000, show_default=True)
@click.option("--window-s", default=0.1, show_default=True)
@click.pass_context
@_handled
def dataset_ingest(ctx, manifest, out, sample_rate, window_s):
    """Ingest consented lab recordings listed in a consent manifest (nothing is discovered or downloaded)."""
    from .data import ingest_consented_recordings

    ds = ingest_consented_recordings(manifest, out, target_sample_rate=sample_rate, window_s=window_s)
    _emit(
        ctx,
        {"path": out, "n_samples": len(ds), "fingerprint": ds.fingerprint},
        f"ingested {len(ds)} recordings -> {out}",
    )


@dataset.command("validate")
@click.argument("path", type=click.Path(file_okay=False))
@click.pass_context
@_handled
def dataset_validate(ctx, path):
    """Check dataset integrity, schema, fingerprint, consent and quality. Exit 2 if invalid."""
    from .data import Dataset, validate_dataset

    rep = validate_dataset(Dataset.load(path))
    lines = (
        [f"{'VALID' if rep.ok else 'INVALID'}: {path}"]
        + [f"  error:   {e}" for e in rep.errors]
        + [f"  warning: {w}" for w in rep.warnings]
    )
    _emit(ctx, rep.to_dict(), "\n".join(lines))
    if not rep.ok:
        ctx.exit(2)


@dataset.command("stats")
@click.argument("path", type=click.Path(file_okay=False))
@click.pass_context
@_handled
def dataset_stats(ctx, path):
    """Print dataset statistics."""
    from .data import Dataset

    st = Dataset.load(path).stats()
    _emit(ctx, st, json.dumps(st, indent=1, default=str))


# ----------------------------------------------------------------------------- experiment
@main.group()
def experiment() -> None:
    """Create, run, list, inspect, cancel, export and reproduce experiments."""


@experiment.command("run")
@click.option(
    "--config", "cfg_file", type=click.Path(exists=True, dir_okay=False), help="Experiment YAML/JSON config."
)
@click.option("--preset", type=str, help="Preset kind (single, model_comparison, defense_comparison, ...).")
@click.option("--id", "eid", type=str, help="Run an existing CREATED/VALIDATED experiment.")
@click.option("--seeds", type=str, help="Comma-separated seeds override, e.g. 0,1,2.")
@click.option(
    "--background", is_flag=True, help="Launch in a bounded worker subprocess and return immediately."
)
@click.pass_context
@_handled
def experiment_run(ctx, cfg_file, preset, eid, seeds, background):
    """Run a complete experiment (dataset -> features -> train -> evaluate -> defenses -> artifacts)."""
    from .config import ConfigError
    from .experiments import LocalExecutor, create_experiment, run_experiment
    from .presets import preset_config

    ws = _ws(ctx)
    if sum(x is not None and x is not False for x in (cfg_file, preset, eid)) != 1:
        raise ConfigError("provide exactly one of --config, --preset, --id")
    if eid is None:
        cfg = load_config(cfg_file) if cfg_file else preset_config(preset)
        if seeds:
            try:
                cfg = cfg.model_copy(update={"seeds": [int(s) for s in seeds.split(",")]})
                cfg = type(cfg).model_validate(cfg.model_dump())
            except Exception as exc:
                raise ConfigError(f"bad --seeds: {exc}") from exc
        eid = create_experiment(ws, cfg)
    if background:
        LocalExecutor(ws, ws.store.get_config(eid).resources.max_concurrent_experiments).submit(eid)
        _emit(ctx, {"id": eid, "status": "LAUNCHED"}, f"launched {eid}")
        return
    t = time.time()

    def prog(i, n, msg):
        if not ctx.obj["json"]:
            click.echo(f"[{i + 1}/{n}] {msg}", err=True)

    status = run_experiment(ws, eid, progress=prog)
    rec = ws.store.get(eid)
    _emit(
        ctx,
        {"id": eid, "status": status, "error": rec["error"], "seconds": round(time.time() - t, 1)},
        f"{eid}: {status} in {time.time() - t:.1f}s" + (f"\n  error: {rec['error']}" if rec["error"] else ""),
    )
    ctx.exit(_EXPERIMENT_EXIT[status])


@experiment.command("list")
@click.option(
    "--status", type=click.Choice(["CREATED", "VALIDATED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"])
)
@click.option("--kind", type=str)
@click.option("--model", type=str)
@click.option("--search", type=str)
@click.option("--limit", default=50, show_default=True, type=int)
@click.pass_context
@_handled
def experiment_list(ctx, status, kind, model, search, limit):
    """List experiments in the registry."""
    rows = _ws(ctx).store.list_experiments(status=status, kind=kind, model=model, search=search, limit=limit)
    text = (
        "\n".join(
            f"{r['id']}  {r['status']:<9} {r['kind']:<18} {r['name'][:40]:<40} {'SYN' if r['dataset_is_synthetic'] else 'REAL' if r['dataset_is_synthetic'] == 0 else '-':<4} {r['created_at']}"
            for r in rows
        )
        or "(no experiments)"
    )
    _emit(ctx, rows, text)


@experiment.command("inspect")
@click.argument("experiment_id")
@click.option("--events", is_flag=True, help="Include execution event log.")
@click.pass_context
@_handled
def experiment_inspect(ctx, experiment_id, events):
    """Show status, provenance, headline results and artifacts for one experiment."""
    st = _ws(ctx).store
    rec = st.get(experiment_id)
    out: dict[str, Any] = {k: v for k, v in rec.items() if k != "config"}
    out["artifacts"] = [a["name"] for a in st.list_artifacts(experiment_id)]
    lines = [
        f"{rec['id']}  [{rec['status']}]  {rec['name']}",
        f"  kind={rec['kind']} dataset={rec['dataset_id']} v{rec['dataset_version']} synthetic={bool(rec['dataset_is_synthetic'])}",
        f"  code={rec['code_revision'][:10]}{' (dirty)' if rec['code_dirty'] else ''} config_hash={rec['config_hash']}",
        f"  progress {rec['progress_done']}/{rec['progress_total']}",
    ]
    if rec["error"]:
        lines.append(f"  error: {rec['error']}")
    if st.has_artifact(experiment_id, "summary.json"):
        s = st.get_json(experiment_id, "summary.json")
        out["summary"] = s
        lines.append(f"  chance={s['chance_accuracy']:.3f}  seeds={s['seeds']}")
        for t in s["table"]:
            if t["condition"] == "clean" and t["attacker"] == "static":
                a = t["accuracy"]
                lines.append(
                    f"  {t['protocol']:<20} {t['model']:<18} variant={t['variant']:<12} acc={a['mean']:.3f}±{a['std']:.3f}  (n={a['n']}, 95% CI {a['ci_lo']:.3f}-{a['ci_hi']:.3f})"
                )
        for c in s["caveats"]:
            lines.append(f"  ! {c}")
    if events:
        out["events"] = st.events(experiment_id)
        lines += [f"  {e['ts']} {e['level']:<5} {e['message'][:150]}" for e in out["events"]]
    _emit(ctx, out, "\n".join(lines))


@experiment.command("cancel")
@click.argument("experiment_id")
@click.pass_context
@_handled
def experiment_cancel(ctx, experiment_id):
    """Cancel an experiment (immediate if not started, cooperative if running)."""
    r = _ws(ctx).store.request_cancel(experiment_id)
    _emit(ctx, {"id": experiment_id, "result": r}, f"{experiment_id}: {r}")


@experiment.command("export")
@click.argument("experiment_ids", nargs=-1, required=True)
@click.option("--out", required=True, type=click.Path())
@click.option(
    "--format", "fmt", type=click.Choice(["csv", "json", "bundle"]), default="csv", show_default=True
)
@click.pass_context
@_handled
def experiment_export(ctx, experiment_ids, out, fmt):
    """Export metrics (CSV/JSON) or a full bundle directory (config, metrics, summary, plots)."""
    st = _ws(ctx).store
    if fmt == "bundle":
        if len(experiment_ids) != 1:
            raise click.UsageError("bundle export takes exactly one experiment")
        p = st.export_experiment(experiment_ids[0], Path(out))
    else:
        p = st.export_metrics(list(experiment_ids), Path(out), fmt)
    _emit(ctx, {"path": str(p)}, f"exported -> {p}")


@experiment.command("reproduce")
@click.argument("experiment_id")
@click.option(
    "--tol", default=1e-6, show_default=True, type=float, help="Max allowed absolute metric difference."
)
@click.pass_context
@_handled
def experiment_reproduce(ctx, experiment_id, tol):
    """Re-run a stored experiment from its immutable config and compare every metric to the original."""
    from .evaluation import SCALAR_KEYS
    from .experiments import create_experiment, run_experiment

    ws = _ws(ctx)
    cfg = ws.store.get_config(experiment_id)
    new = create_experiment(ws, cfg, parent_id=experiment_id)
    status = run_experiment(ws, new)
    if status != "COMPLETED":
        raise NotFoundError(f"reproduction run {new} ended {status}: {ws.store.get(new)['error']}")
    a = {_metric_key(m): m["value"] for m in ws.store.metrics(experiment_id) if m["name"] in SCALAR_KEYS}
    b = {_metric_key(m): m["value"] for m in ws.store.metrics(new) if m["name"] in SCALAR_KEYS}
    missing = set(a) ^ set(b)
    worst = max((abs(a[k] - b[k]) for k in a.keys() & b.keys()), default=0.0)
    ok = not missing and worst <= tol
    same_fp = ws.store.get(experiment_id)["dataset_fingerprint"] == ws.store.get(new)["dataset_fingerprint"]
    res = {
        "original": experiment_id,
        "reproduction": new,
        "metrics_compared": len(a),
        "max_abs_diff": worst,
        "tolerance": tol,
        "keys_mismatch": len(missing),
        "dataset_fingerprint_match": same_fp,
        "reproduced": ok,
    }
    _emit(
        ctx,
        res,
        f"{'REPRODUCED' if ok else 'NOT REPRODUCED'}: {len(a)} metrics, max |diff| = {worst:.3g} (tol {tol:g}), dataset fingerprint match = {same_fp}\n  new run: {new}",
    )
    ctx.exit(0 if ok else 1)


# ----------------------------------------------------------------------------- benchmark
@main.group()
def benchmark() -> None:
    """Defense / model benchmarks and result comparison."""


@benchmark.command("run")
@click.option("--config", "cfg_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--preset", default="defense_comparison", show_default=True)
@click.option("--seeds", type=str, help="Comma-separated seeds, e.g. 0,1,2")
@click.pass_context
@_handled
def benchmark_run(ctx, cfg_file, preset, seeds):
    """Run a (defense) benchmark and print effectiveness vs distortion with Pareto-optimal configurations."""
    from .experiments import create_experiment, run_experiment
    from .presets import preset_config

    ws = _ws(ctx)
    cfg = load_config(cfg_file) if cfg_file else preset_config(preset)
    if seeds:
        cfg = type(cfg).model_validate({**cfg.model_dump(), "seeds": [int(s) for s in seeds.split(",")]})
    eid = create_experiment(ws, cfg)
    status = run_experiment(
        ws,
        eid,
        progress=None if ctx.obj["json"] else (lambda i, n, m: click.echo(f"[{i + 1}/{n}] {m}", err=True)),
    )
    if status != "COMPLETED":
        raise NotFoundError(f"{eid} ended {status}: {ws.store.get(eid)['error']}")
    s = ws.store.get_json(eid, "summary.json")
    _emit(ctx, {"id": eid, "summary": s}, f"{eid}: COMPLETED\n" + _defense_table(s))


def _defense_table(s: dict[str, Any]) -> str:
    if not s["defense_points"]:
        return "(no defense conditions in this experiment)"
    front = {(p["variant"], p["protocol"], p["model"]): set(p["frontier"]) for p in s["pareto"]}
    lines = [
        f"{'model':<14}{'defense':<26}{'static drop':>12}{'adaptive drop':>14}{'LSD dB':>9}{'SNR dB':>8}{'RTF':>8}  pareto  evidence"
    ]
    for p in sorted(s["defense_points"], key=lambda p: (p["model"], -(p["privacy_gain"] or -9))):
        f = lambda v, w, d=3: f"{v:>{w}.{d}f}" if v is not None else " " * (w - 1) + "-"  # noqa: E731
        star = "*" if p["name"] in front.get((p["variant"], p["protocol"], p["model"]), ()) else ""
        ev = ",".join(f"{k[:3]}:{v}" for k, v in p["evidence"].items())
        lines.append(
            f"{p['model']:<14}{p['name']:<26}{f(p['gain_static'], 12)}{f(p['gain_adaptive'], 14)}{f(p['log_spectral_distance_db'], 9, 1)}{f(p['waveform_snr_db'], 8, 1)}{f(p['real_time_factor'], 8, 4)}  {star:^6}  {ev}"
        )
    lines.append(
        f"chance accuracy = {s['chance_accuracy']:.3f}; drop = clean accuracy - defended accuracy (held-out); '*' = Pareto-optimal"
    )
    return "\n".join(lines + [f"! {c}" for c in s["caveats"]])


@benchmark.command("compare")
@click.argument("experiment_ids", nargs=-1, required=True)
@click.pass_context
@_handled
def benchmark_compare(ctx, experiment_ids):
    """Compare clean held-out accuracy (mean ± std over seeds) across experiments."""
    st = _ws(ctx).store
    rows = []
    for e in experiment_ids:
        rec = st.get(e)
        if not st.has_artifact(e, "summary.json"):
            raise NotFoundError(f"{e} has no summary (status {rec['status']})")
        for t in st.get_json(e, "summary.json")["table"]:
            if t["condition"] == "clean" and t["attacker"] == "static":
                rows.append(
                    {
                        "experiment": e,
                        "name": rec["name"],
                        "variant": t["variant"],
                        "protocol": t["protocol"],
                        "model": t["model"],
                        "accuracy_mean": t["accuracy"]["mean"],
                        "accuracy_std": t["accuracy"]["std"],
                        "n_seeds": t["accuracy"]["n"],
                        "synthetic": bool(rec["dataset_is_synthetic"]),
                    }
                )
    text = "\n".join(
        f"{r['experiment'][-17:]:<18}{r['protocol']:<19}{r['model']:<18}{r['variant'][:16]:<17}{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} (n={r['n_seeds']}){' [synthetic]' if r['synthetic'] else ''}"
        for r in rows
    )
    _emit(ctx, rows, text)


@benchmark.command("profile")
@click.option("--samples-per-class", default=20, show_default=True)
@click.pass_context
@_handled
def benchmark_profile(ctx, samples_per_class):
    """Measure real throughput/latency of each pipeline stage on this machine."""
    from .profiling import profile_pipeline

    res = profile_pipeline(_ws(ctx), samples_per_class)
    _emit(ctx, res, "\n".join(f"{k:<38}{v}" for k, v in res.items()))


# ----------------------------------------------------------------------------- dashboard / doctor
@main.command()
@click.option("--port", default=8501, show_default=True, type=int)
@click.option("--headless", is_flag=True, help="Do not open a browser.")
@click.pass_context
@_handled
def dashboard(ctx, port, headless):
    """Launch the research dashboard (Streamlit) against this workspace."""
    app = Path(__file__).parent / "dashboard" / "app.py"
    env = {**os.environ, "OPENKEYSTROKE_HOME": str(_ws(ctx).root)}
    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true" if headless else "false",
    ]
    os.execvpe(args[0], args, env)


@main.command()
@click.option(
    "--self-test", is_flag=True, help="Also run a tiny end-to-end synthetic experiment in a temp workspace."
)
@click.pass_context
@_handled
def doctor(ctx, self_test):
    """Check the installation, workspace and (optionally) the full pipeline. Exit 1 on any failure."""
    from .doctor import run_doctor

    checks = run_doctor(_ws(ctx), self_test)
    bad = [c for c in checks if not c["ok"]]
    _emit(
        ctx, checks, "\n".join(f"[{'ok' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in checks)
    )
    if bad:
        ctx.exit(1)
