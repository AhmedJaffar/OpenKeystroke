"""OpenKeystroke research dashboard. Every view reads the real registry, artifact store and datasets of the workspace."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from pydantic import ValidationError

from openkeystroke.config import ExperimentConfig, PreprocessConfig, SplitConfig, SyntheticConfig
from openkeystroke.dashboard import backend as be
from openkeystroke.dashboard import ui
from openkeystroke.data import Dataset, detect_leakage, make_split
from openkeystroke.defense import DEFENSES, DefenseChain, distortion_metrics
from openkeystroke.dsp import PreprocessPipeline, ops
from openkeystroke.errors import OpenKeystrokeError
from openkeystroke.experiments import build_jobs
from openkeystroke.features import EXTRACTORS, make_extractor
from openkeystroke.models import MODEL_REGISTRY
from openkeystroke.presets import PRESET_KINDS, default_defenses, preset_config

ui.setup()


@st.cache_resource
def workspace(home: str):
    return be.get_workspace(home)


WS = workspace(os.environ.get("OPENKEYSTROKE_HOME", ""))


@st.cache_data(show_spinner=False)
def cached_json(eid: str, name: str, mtime: float) -> dict[str, Any]:
    return WS.store.get_json(eid, name)


def summary_of(eid: str) -> dict[str, Any]:
    return be.normalize_summary(cached_json(eid, "summary.json", be.artifact_mtime(WS, eid, "summary.json")))


def results_of(eid: str) -> dict[str, Any]:
    return cached_json(eid, "results.json", be.artifact_mtime(WS, eid, "results.json"))


@st.cache_resource(show_spinner=False)
def load_ds(path: str, mtime: float) -> Dataset:
    return Dataset.load(path, mmap=True)


def pick_dataset(key: str) -> tuple[Dataset | None, dict[str, Any] | None]:
    dss = be.list_datasets(WS)
    if not dss:
        ui.empty_state(
            "No datasets in this workspace", "Generate a labelled SYNTHETIC dataset to get started."
        )
        if st.button("Generate synthetic demo dataset", key=f"gen_{key}"):
            with st.spinner("Generating deterministic synthetic dataset…"):
                be.generate_demo_dataset(WS)
            st.rerun()
        return None, None
    labels = [
        f"{d['name']} · {d['n_samples']} samples · {'synthetic' if d['is_synthetic'] else 'real (consented)'}"
        for d in dss
    ]
    i = st.selectbox("Dataset", range(len(dss)), format_func=lambda k: labels[k], key=f"ds_{key}")
    d = dss[int(i or 0)]
    return load_ds(d["path"], d["mtime"]), d


def df_pm(m: dict[str, Any]) -> str:
    return f"{m['mean']:.3f} ± {m['std']:.3f}"


# ============================================================================ Overview
def page_overview() -> None:
    ui.header(
        "Overview", "Workspace inventory, experiment activity and headline results from stored artifacts."
    )
    ov = be.experiment_overview(WS)
    c = ov["counts"]
    dss = be.list_datasets(WS)
    cols = st.columns(5)
    cols[0].metric("Datasets", len(dss))
    cols[1].metric("Experiments", sum(c.values()))
    cols[2].metric("Running", c.get("RUNNING", 0))
    cols[3].metric("Completed", c.get("COMPLETED", 0))
    cols[4].metric("Failed", c.get("FAILED", 0))
    if not ov["recent"] and not dss:
        ui.empty_state(
            "Nothing here yet",
            "Run a quick end-to-end demo on SYNTHETIC data, or open Experiment Studio to configure your own.",
        )
        if st.button("Run quick synthetic demo experiment"):
            cfg = preset_config("model_comparison", name="Quick demo (synthetic)")
            cfg = be.config_from_form(
                cfg,
                models=[
                    {"name": "logreg"},
                    {"name": "random_forest"},
                    {"name": "cnn1d", "training": {"epochs": 8}},
                ],
            )
            st.session_state["monitor_id"] = be.launch(WS, cfg)
            st.rerun()
    left, right = st.columns([5, 4])
    with left:
        st.subheader("Recent experiments")
        if ov["recent"]:
            df = pd.DataFrame(ov["recent"])[["name", "kind", "status", "dataset_is_synthetic"]]
            df["data"] = df.pop("dataset_is_synthetic").map({1: "synthetic", 0: "real"}).fillna("–")
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            ui.empty_state("No experiments yet", "Launch one from Experiment Studio.")
        st.subheader("Model performance overview")
        lb = pd.DataFrame(be.model_leaderboard(WS))
        if len(lb):
            best = lb.sort_values("accuracy", ascending=False).drop_duplicates(["model", "protocol"])
            fig = go.Figure()
            for p, g in best.groupby("protocol"):
                fig.add_bar(
                    x=g["model"], y=g["accuracy"], name=p, error_y=dict(type="data", array=g["std"].fillna(0))
                )
            fig.update_yaxes(title="held-out accuracy", range=[0, 1])
            ui.show(fig, "ov_models")
            st.caption(
                "Best stored result per model and evaluation protocol (mean ± std over seeds; held-out test partitions). "
                + ("Includes synthetic data." if lb["synthetic"].any() else "")
            )
        else:
            ui.empty_state(
                "No completed experiments with results",
                "Model performance appears here once an experiment completes.",
            )
    with right:
        st.subheader("Defense effectiveness")
        dv = pd.DataFrame(be.defense_overview(WS))
        if len(dv):
            top = dv[dv["eligible"]].sort_values("privacy_gain", ascending=False).head(8)
            if len(top):
                st.dataframe(
                    top[["name", "model", "privacy_gain", "log_spectral_distance_db"]]
                    .rename(columns={"privacy_gain": "drop", "log_spectral_distance_db": "LSD dB"})
                    .round(3),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "drop = worst-case accuracy drop over attackers. Only defenses with a positive drop and no non-significant attacker verdict are listed."
                )
            else:
                st.info(
                    "No defense in stored benchmarks showed a significant reduction against all attackers."
                )
        else:
            ui.empty_state("No defense benchmarks", "Run a defense comparison from Defense Laboratory.")
        st.subheader("System")
        s = be.system_status(WS)
        st.dataframe(
            pd.DataFrame(
                {
                    "": [
                        "Compute device",
                        "CPU cores / load (1m)",
                        "Memory available",
                        "Disk free",
                        "Registry schema",
                        "Workspace",
                    ],
                    " ": [
                        s["device"],
                        f"{s['cpu_count']} / {s['load_1m']:.2f}"
                        if s["load_1m"] is not None
                        else str(s["cpu_count"]),
                        f"{(s['mem_available_mb'] or 0) / 1024:.1f} of {(s['mem_total_mb'] or 0) / 1024:.1f} GB",
                        f"{s['disk_free_gb']:.1f} GB",
                        f"v{s['registry_schema']}",
                        s["workspace"],
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )


# ============================================================================ Signal Explorer
def _defense_form(prefix: str) -> list[dict[str, Any]]:
    steps = []
    sel = st.multiselect(
        "Defense chain (applied in order)", list(DEFENSES), default=["lowpass"], key=f"{prefix}_sel"
    )
    for t in sel:
        c = st.columns(3)
        if t == "lowpass":
            steps.append(
                {
                    "type": t,
                    "params": {
                        "cutoff_hz": c[0].slider(
                            "Low-pass cutoff (Hz)", 300, 7500, 2000, 100, key=f"{prefix}_lp"
                        )
                    },
                }
            )
        elif t == "bandpass":
            lo = c[0].slider("Band low (Hz)", 50, 3000, 500, 50, key=f"{prefix}_bl")
            hi = c[1].slider("Band high (Hz)", 1000, 7900, 3000, 100, key=f"{prefix}_bh")
            steps.append({"type": t, "params": {"low_hz": lo, "high_hz": max(hi, lo + 100)}})
        elif t == "noise":
            steps.append(
                {
                    "type": t,
                    "params": {"snr_db": c[0].slider("Noise SNR (dB)", -5, 40, 10, key=f"{prefix}_ns")},
                }
            )
        elif t == "transient_attenuation":
            steps.append(
                {
                    "type": t,
                    "params": {
                        "attenuation_db": c[0].slider("Attenuation (dB)", 3, 40, 12, key=f"{prefix}_ta"),
                        "window_ms": c[1].slider("Window (ms)", 5, 60, 20, key=f"{prefix}_tw"),
                    },
                }
            )
        elif t == "spectral_smoothing":
            steps.append(
                {
                    "type": t,
                    "params": {
                        "width_bins": c[0].slider("Smoothing width (bins)", 1, 51, 9, key=f"{prefix}_ss")
                    },
                }
            )
        elif t == "tf_mask":
            steps.append(
                {
                    "type": t,
                    "params": {
                        "n_masks": c[0].slider("Masks", 1, 8, 3, key=f"{prefix}_tm"),
                        "freq_frac": c[1].slider("Freq fraction", 0.02, 0.4, 0.15, key=f"{prefix}_tf"),
                        "time_frac": c[2].slider("Time fraction", 0.02, 0.4, 0.2, key=f"{prefix}_tt"),
                    },
                }
            )
    return steps


def _heat(z: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, zlabel: str, key: str, **kw: Any) -> None:
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale="Viridis", colorbar=dict(title=zlabel), **kw))
    fig.update_layout(title=title, xaxis_title="time (ms)", yaxis_title="frequency (Hz)")
    ui.show(fig, key, legend=False)


def page_signal() -> None:
    ui.header(
        "Signal Explorer",
        "Inspect waveforms, spectra and spectrograms; compare preprocessing and defenses on the same sample.",
    )
    ds, _ = pick_dataset("sig")
    if ds is None:
        return
    ui.synthetic_notice(ds.is_synthetic)
    f1, f2, f3 = st.columns(3)
    cls = f1.selectbox("Class", ["(all)", *ds.class_names], key="sig_cls")
    dev = f2.selectbox("Device", ["(all)", *sorted(set(r.device_id for r in ds.records))], key="sig_dev")
    cand = [
        i
        for i, r in enumerate(ds.records)
        if (cls == "(all)" or r.label == cls) and (dev == "(all)" or r.device_id == dev)
    ]
    if not cand:
        ui.empty_state("No samples match these filters", "Relax the class or device filter.")
        return
    idx = f3.selectbox(
        "Sample",
        cand[:300],
        format_func=lambda i: ds.records[i].sample_id[-12:] + f" · {ds.records[i].label}",
        key="sig_idx",
    )
    sv = be.sample_view(ds, int(idx if idx is not None else cand[0]))
    x, sr = sv["x"][None, :], sv["sr"]

    with st.expander("Preprocessing", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        dc = p1.checkbox("Remove DC", True, key="pp_dc")
        norm = p2.selectbox("Normalise", ["peak", "rms", "none"], key="pp_norm")
        lp = p3.slider("Low-pass (Hz, 0=off)", 0, 7500, 0, 100, key="pp_lp")
        bp = p4.checkbox("Band-pass 300–6000 Hz", False, key="pp_bp")
    pcfg = PreprocessConfig(
        target_sample_rate=sr,
        remove_dc=dc,
        normalize=norm,
        lowpass_hz=float(lp) if lp else None,
        bandpass_hz=(300.0, 6000.0) if bp else None,
    )
    proc = PreprocessPipeline(pcfg)(x, sr)
    y = proc.waveforms
    t_ms = np.arange(x.shape[1]) / sr * 1000
    onset = int(ops.detect_transient(x, sr)[0])
    m = st.columns(5)
    m[0].metric("Duration", f"{x.shape[1] / sr * 1000:.0f} ms")
    m[1].metric("Sample rate", f"{sr} Hz")
    m[2].metric("Onset", f"{onset / sr * 1000:.1f} ms" if onset >= 0 else "none")
    m[3].metric("Blind SNR", f"{ops.estimate_snr_db(x, sr)[0]:.1f} dB")
    m[4].metric("Noise floor (RMS)", f"{ops.noise_floor(x, sr)[0]:.4f}")

    tabs = st.tabs(
        ["Waveform", "Spectrogram", "Mel", "Spectrum", "Time–frequency", "Defense before/after", "Metadata"]
    )
    with tabs[0]:
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("original (as recorded)", "after preprocessing"),
        )
        fig.add_scatter(x=t_ms, y=x[0], name="original", line=dict(width=1), row=1, col=1)
        fig.add_scatter(x=t_ms[: y.shape[1]], y=y[0], name="preprocessed", line=dict(width=1), row=2, col=1)
        if onset >= 0:
            fig.add_vline(x=onset / sr * 1000, line_dash="dot")
        fig.update_xaxes(title_text="time (ms)", row=2, col=1)
        fig.update_yaxes(title_text="amplitude")
        ui.show(fig, "sig_wave", height=420, legend=False)
        st.caption("Preprocessing lineage: " + " → ".join(s["op"] for s in proc.lineage))
    n_fft = st.session_state.get("sig_nfft", 256)
    with tabs[1]:
        n_fft = st.select_slider("FFT size", [128, 256, 512, 1024], 256, key="sig_nfft")
        S = ops.log_magnitude(ops.stft(y, n_fft, n_fft // 4))[0]
        _heat(
            S,
            np.arange(S.shape[1]) * (n_fft // 4) / sr * 1000,
            np.linspace(0, sr / 2, S.shape[0]),
            "Log-magnitude spectrogram (preprocessed)",
            "dB",
            "sig_spec",
        )
    with tabs[2]:
        nm = st.slider("Mel bands", 8, 128, 40, key="sig_nmel")
        fb = ops.mel_filterbank(sr, 256, nm, 50.0)
        mel = 10 * np.log10(ops.mel_spectrogram(ops.power_spectrogram(ops.stft(y, 256, 64)), fb)[0] + 1e-10)
        _heat(
            mel, np.arange(mel.shape[1]) * 64 / sr * 1000, np.arange(nm), "Mel spectrogram", "dB", "sig_mel"
        )
    with tabs[3]:
        f = np.fft.rfftfreq(x.shape[1], 1 / sr)
        fig = go.Figure()
        fig.add_scatter(x=f, y=20 * np.log10(np.abs(np.fft.rfft(x[0])) + 1e-9), name="original")
        fig.add_scatter(
            x=np.fft.rfftfreq(y.shape[1], 1 / sr),
            y=20 * np.log10(np.abs(np.fft.rfft(y[0])) + 1e-9),
            name="preprocessed",
        )
        fig.update_layout(xaxis_title="frequency (Hz)", yaxis_title="magnitude (dB)")
        ui.show(fig, "sig_fft")
    with tabs[4]:
        S2 = ops.stft(y, 256, 64)
        k = st.slider(
            "Frame",
            0,
            S2.shape[2] - 1,
            min(S2.shape[2] - 1, max(0, onset // 64)) if onset >= 0 else 0,
            key="sig_frame",
        )
        fig = go.Figure(
            go.Scatter(x=np.linspace(0, sr / 2, S2.shape[1]), y=20 * np.log10(np.abs(S2[0, :, k]) + 1e-9))
        )
        fig.update_layout(
            title=f"Spectrum at {k * 64 / sr * 1000:.1f} ms", xaxis_title="frequency (Hz)", yaxis_title="dB"
        )
        ui.show(fig, "sig_frame_fig", legend=False)
    with tabs[5]:
        steps = _defense_form("sigdef")
        if steps:
            try:
                cfg = ExperimentConfig  # noqa: F841  (schema reuse)
                from openkeystroke.config import DefenseConfig

                chain = DefenseChain(DefenseConfig(name="interactive", steps=steps))
                z = chain.apply(x, sr, seed=0)
                batch_idx = np.sort(np.random.default_rng(0).choice(len(ds), min(32, len(ds)), replace=False))
                bw = np.asarray(ds.waveforms)[batch_idx]
                q = distortion_metrics(bw, chain.apply(bw, sr, 0), sr)
                fig = go.Figure()
                fig.add_scatter(x=t_ms, y=x[0], name="original", line=dict(width=1))
                fig.add_scatter(x=t_ms, y=z[0], name="defended", line=dict(width=1))
                fig.update_layout(xaxis_title="time (ms)", yaxis_title="amplitude")
                ui.show(fig, "sig_def_wave")
                d = ops.log_magnitude(ops.stft(z, 256, 64))[0] - ops.log_magnitude(ops.stft(x, 256, 64))[0]
                _heat(
                    d,
                    np.arange(d.shape[1]) * 64 / sr * 1000,
                    np.linspace(0, sr / 2, d.shape[0]),
                    "Spectrogram change (defended − original)",
                    "dB",
                    "sig_def_diff",
                )
                st.dataframe(
                    pd.DataFrame([q]).T.rename(columns={0: "mean over 32 dataset samples"}), width="stretch"
                )
                st.caption(
                    "Distortion alone does not show a defense works. Evaluate leakage reduction against trained attackers in Defense Laboratory."
                )
            except OpenKeystrokeError as e:
                st.error(f"{e.code}: {e.message}")
        else:
            st.info("Select at least one defense.")
    with tabs[6]:
        st.json(sv["meta"])


# ============================================================================ Dataset Explorer
def page_dataset() -> None:
    ui.header("Dataset Explorer", "Composition, quality, split integrity and feature distributions.")
    ds, d = pick_dataset("dsx")
    if ds is None:
        return
    ui.synthetic_notice(ds.is_synthetic)
    st.caption(
        f"id `{ds.manifest.dataset_id}` · version {ds.manifest.version} · fingerprint `{ds.fingerprint[:16]}…`"
    )
    q = be.dataset_quality(ds)
    if q["errors"]:
        st.error("Validation errors: " + "; ".join(q["errors"]))
    for w in q["warnings"]:
        if not w.startswith("SYNTHETIC"):
            st.warning(w)
    if q["ok"]:
        st.success("Integrity check passed (fingerprint, schema, labels, consent record where required).")
    stats = ds.stats()
    t = st.tabs(["Distributions", "Split integrity", "Samples", "Features"])
    with t[0]:
        c = st.columns(4)
        for col, key, title in zip(
            c,
            ("classes", "devices", "sessions", "environments"),
            ("Class", "Device", "Session", "Environment"),
            strict=True,
        ):
            fig = go.Figure(go.Bar(x=list(stats[key].keys()), y=list(stats[key].values())))
            fig.update_layout(title=f"{title} distribution")
            with col:
                ui.show(fig, f"dist_{key}", height=260, legend=False)
        if stats["snr_db"]:
            snr = [r.noise["snr_db"] for r in ds.records if "snr_db" in r.noise]
            ui.show(
                go.Figure(go.Histogram(x=snr, nbinsx=30)).update_layout(title="Recorded SNR (dB)"),
                "dist_snr",
                height=260,
                legend=False,
            )
    with t[1]:
        c1, c2, c3 = st.columns(3)
        proto = c1.selectbox(
            "Protocol",
            ["in_distribution", "session_disjoint", "device_disjoint", "environment_disjoint"],
            index=1,
        )
        seed = c2.number_input("Seed", 0, 10_000, 0)
        vf = c3.slider("Val / test fraction", 0.1, 0.4, 0.2, 0.05)
        try:
            sp = make_split(
                ds, proto or "session_disjoint", SplitConfig(val_fraction=vf, test_fraction=vf), int(seed)
            )
            rep = detect_leakage(ds, sp)
            st.dataframe(pd.DataFrame([sp.sizes()]), hide_index=True)
            (st.success if rep.ok else st.error)(
                "No leakage detected" if rep.ok else "Leakage: " + "; ".join(rep.errors)
            )
            for w in rep.warnings:
                st.warning(w)
            if sp.groups:
                st.dataframe(
                    pd.DataFrame({k: [", ".join(v)] for k, v in sp.groups.items()}),
                    hide_index=True,
                    width="stretch",
                )
            y = ds.labels
            cm = pd.DataFrame(
                {
                    k: np.bincount(y[getattr(sp, k)], minlength=len(ds.class_names))
                    for k in ("train", "val", "test")
                },
                index=ds.class_names,
            )
            st.bar_chart(cm)
        except OpenKeystrokeError as e:
            st.error(f"{e.code}: {e.message}")
    with t[2]:
        df = pd.DataFrame(
            [
                {
                    **{
                        k: v
                        for k, v in r.model_dump().items()
                        if k not in ("environment", "noise", "lineage")
                    },
                    "environment": r.environment_name,
                    "snr_db": r.noise.get("snr_db"),
                }
                for r in ds.records
            ]
        )
        f = st.multiselect("Filter by class", ds.class_names)
        st.dataframe(df[df["label"].isin(f)] if f else df, hide_index=True, width="stretch", height=360)
    with t[3]:
        name = st.selectbox(
            "Extractor",
            [n for n, c in EXTRACTORS.items() if c.kind == "vector" and not c.requires_fit],
            index=2,
        )
        sample = np.sort(np.random.default_rng(0).choice(len(ds), min(400, len(ds)), replace=False))
        from openkeystroke.config import FeatureConfig

        wf = PreprocessPipeline(PreprocessConfig(target_sample_rate=ds.sample_rate))(
            np.asarray(ds.waveforms)[sample], ds.sample_rate
        ).waveforms
        F = make_extractor(FeatureConfig(name=name), ds.sample_rate).batched(wf)
        j = st.slider("Feature index", 0, F.shape[1] - 1, 0)
        fig = go.Figure()
        yl = ds.labels[sample]
        for k, cn in enumerate(ds.class_names):
            fig.add_histogram(x=F[yl == k, j], name=cn, opacity=0.6)
        fig.update_layout(barmode="overlay", xaxis_title=f"{name}[{j}]")
        ui.show(fig, "feat_hist")
        st.caption(
            f"Computed on {len(sample)} randomly chosen samples (not a train-only statistic; for exploration only)."
        )


# ============================================================================ Experiment Studio
def _monitor() -> None:
    @st.fragment(run_every=2 if be.active_runs(WS) else None)
    def body() -> None:
        runs = be.active_runs(WS)
        recent = [
            r
            for r in WS.store.list_experiments(limit=6)
            if r["status"] in ("COMPLETED", "FAILED", "CANCELLED")
        ][:4]
        if not runs and not recent:
            ui.empty_state("No runs yet", "Launch an experiment above to monitor it here.")
            return
        for r in runs + recent:
            with st.container(border=True):
                a, b = st.columns([4, 1])
                a.markdown(
                    f"{ui.badge(r['status'])} &nbsp; **{r['name']}** &nbsp; `{r['id']}`",
                    unsafe_allow_html=True,
                )
                if r["status"] in ("RUNNING", "VALIDATED", "CREATED") and b.button(
                    "Cancel", key=f"cancel_{r['id']}"
                ):
                    st.toast(f"{r['id']}: {WS.store.request_cancel(r['id'])}")
                tot = max(r["progress_total"], 1)
                if r["status"] == "RUNNING":
                    st.progress(
                        min(r["progress_done"] / tot, 1.0),
                        text=f"{r['progress_done']}/{r['progress_total']} jobs · {r['progress_message'] or ''}",
                    )
                if r["error"]:
                    (st.error if r["status"] == "FAILED" else st.info)(r["error"])
                if r["status"] == "COMPLETED":
                    st.caption("Open Model Evaluation or Defense Laboratory to inspect results.")

    body()


def _studio_config() -> ExperimentConfig | None:
    kind = st.selectbox(
        "Experiment type", PRESET_KINDS, index=PRESET_KINDS.index("model_comparison"), key="st_kind"
    )
    base = preset_config(kind or "single")
    c1, c2 = st.columns(2)
    name = c1.text_input("Name", f"{kind} run", key="st_name")
    hyp = c2.text_input("Hypothesis", base.hypothesis, key="st_hyp")
    with st.expander("Dataset", expanded=True):
        src = st.radio(
            "Source",
            ["Synthetic (generated deterministically)", "Stored dataset"],
            horizontal=True,
            key="st_src",
        )
        if (src or "").startswith("Synthetic"):
            a = st.columns(5)
            syn = SyntheticConfig(
                seed=int(a[0].number_input("Seed", 0, 99999, 0, key="sy_seed")),
                n_classes=int(a[1].number_input("Classes", 2, 26, 6, key="sy_c")),
                n_devices=int(a[2].number_input("Devices", 1, 16, 4, key="sy_d")),
                sessions_per_device=int(a[3].number_input("Sessions/device", 1, 16, 3, key="sy_s")),
                samples_per_class_per_session=int(
                    a[4].number_input("Samples/class/session", 1, 200, 8, key="sy_n")
                ),
            )
            dref: dict[str, Any] = {"synthetic": syn.model_dump(mode="json")}
        else:
            dss = be.list_datasets(WS)
            if not dss:
                st.warning("No stored datasets. Generate or ingest one first.")
                return None
            pick = st.selectbox(
                "Dataset", dss, format_func=lambda d: f"{d['name']} ({d['n_samples']} samples)", key="st_ds"
            )
            dref = {"path": pick["path"]}
    with st.expander("Models & training", expanded=True):
        default_models = [m.name for m in base.models]
        models = st.multiselect(
            "Models", list(MODEL_REGISTRY), default=default_models, key=f"st_models_{kind}"
        )
        epochs = st.slider("Epochs (deep models)", 1, 100, 12, key="st_epochs")
        seeds_txt = st.text_input(
            "Seeds (comma-separated)", ",".join(map(str, base.seeds)), key=f"st_seeds_{kind}"
        )
    with st.expander("Preprocessing, features, evaluation", expanded=False):
        p = st.columns(4)
        norm = p[0].selectbox("Normalise", ["peak", "rms", "none"], key="st_norm")
        dc = p[1].checkbox("Remove DC", True, key="st_dc")
        vec = p[2].selectbox(
            "Vector features (classical)",
            [n for n, c in EXTRACTORS.items() if c.kind == "vector"],
            index=2,
            key="st_vec",
        )
        ten = p[3].selectbox(
            "Spectrogram features (CNN)",
            [n for n, c in EXTRACTORS.items() if c.kind == "tensor"],
            index=2,
            key="st_ten",
        )
        protos = st.multiselect(
            "Evaluation protocols",
            ["in_distribution", "session_disjoint", "device_disjoint", "environment_disjoint"],
            default=base.effective_protocols(),
            key=f"st_protos_{kind}",
        )
        boot = st.slider("Bootstrap resamples", 0, 2000, 300, 50, key="st_boot")
    defenses, noise = list(base.defenses), list(base.noise_snr_db)
    if kind in ("defense_comparison",):
        with st.expander("Defenses", expanded=True):
            names = [d.name for d in default_defenses()]
            chosen = st.multiselect("Defenses to benchmark", names, default=names, key="st_defs")
            defenses = [d for d in default_defenses() if d.name in chosen]
            att = st.multiselect(
                "Attacker models", ["static", "adaptive"], default=["static", "adaptive"], key="st_att"
            )
    else:
        att = list(base.attacker_modes)
    if kind == "noise_robustness":
        noise = [
            float(x)
            for x in st.text_input(
                "Test SNR levels (dB)", ",".join(f"{n:g}" for n in noise), key="st_noise"
            ).split(",")
            if x.strip()
        ]
    try:
        seeds = [int(s) for s in seeds_txt.split(",") if s.strip()]
        mdl = []
        for m in models:
            e: dict[str, Any] = {"name": m}
            if MODEL_REGISTRY[m].family == "deep":
                e["training"] = {"epochs": epochs, "patience": min(8, epochs)}
            old = next((x for x in base.models if x.name == m), None)
            if old is not None and old.params:
                e["params"] = old.params
            mdl.append(e)
        raw = base.model_dump(mode="json")
        raw.update(
            name=name,
            hypothesis=hyp,
            dataset=dref,
            models=mdl,
            seeds=seeds,
            protocols=protos or None,
            defenses=[d.model_dump(mode="json") for d in defenses],
            noise_snr_db=noise,
            attacker_modes=att,
        )
        raw["preprocess"].update(normalize=norm, remove_dc=dc)
        raw["vector_features"]["name"], raw["tensor_features"]["name"] = vec, ten
        raw["evaluation"]["bootstrap_samples"] = boot
        if kind == "single":
            raw["seeds"] = seeds[:1] or [0]
        cfg = ExperimentConfig.model_validate(raw)
        n = len(build_jobs(cfg))
    except (ValidationError, OpenKeystrokeError, ValueError) as e:
        st.error(f"Configuration invalid: {getattr(e, 'message', e)}")
        return None
    with st.expander("Resulting configuration (same schema as CLI / Python API)"):
        st.code(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), language="yaml")
    st.caption(f"This configuration expands to **{n} training jobs**.")
    return cfg


def page_studio() -> None:
    ui.header(
        "Experiment Studio",
        "Configure, launch and monitor experiments. Runs execute in bounded local worker processes.",
    )
    cfg = _studio_config()
    if (
        st.button("Launch experiment", type="primary", disabled=cfg is None, key="st_launch")
        and cfg is not None
    ):
        try:
            with st.spinner("Validating dataset, splits and leakage checks…"):
                eid = be.launch(WS, cfg)
            st.session_state["monitor_id"] = eid
            st.success(f"Launched `{eid}`")
        except OpenKeystrokeError as e:
            st.error(f"{e.code}: {e.message}")
    st.subheader("Runs")
    _monitor()
    with st.expander("Execution log of a run"):
        runs = WS.store.list_experiments(limit=20)
        if runs:
            r = st.selectbox("Run", runs, format_func=lambda r: f"{r['id']} · {r['status']}", key="st_log")
            for ev in WS.store.events(r["id"])[-40:]:
                st.text(f"{ev['ts']} {ev['level']:<5} {ev['message'][:240]}")


# ============================================================================ Model Evaluation
def _pick_experiment(key: str, with_defenses: bool = False) -> dict[str, Any] | None:
    exps = be.completed_with_summary(WS, with_defenses=with_defenses)
    if not exps:
        ui.empty_state(
            "No completed experiments with " + ("defense conditions" if with_defenses else "results"),
            "Launch one from "
            + (
                "the form below."
                if with_defenses
                else "Experiment Studio; results appear here when it completes."
            ),
        )
        return None
    return st.selectbox(
        "Experiment",
        exps,
        format_func=lambda r: f"{r['name']} · {r['id'][-17:]} · {r['kind']}",
        key=f"pick_{key}",
    )


def page_eval() -> None:
    ui.header(
        "Model Evaluation",
        "Held-out performance with uncertainty. Training accuracy is never shown as evidence of generalisation.",
    )
    rec = _pick_experiment("eval")
    if rec is None:
        return
    eid = rec["id"]
    s, res = summary_of(eid), results_of(eid)
    ui.synthetic_notice(s["dataset"]["is_synthetic"])
    for c in s["caveats"]:
        if not c.startswith("SYNTHETIC"):
            st.info(c)
    rows = res["rows"]
    f1, f2 = st.columns(2)
    attackers = sorted({r["attacker"] for r in rows})
    conds = sorted({r["condition"] for r in rows})
    cond = f1.selectbox(
        "Condition", conds, index=conds.index("clean") if "clean" in conds else 0, key="ev_cond"
    )
    att = f2.selectbox("Attacker training", attackers, key="ev_att")
    tab = [t for t in s["table"] if t["condition"] == cond and t["attacker"] == att]
    t = st.tabs(
        [
            "Comparison",
            "Confusion",
            "Training curves",
            "Calibration",
            "Seeds & CIs",
            "Domain shift",
            "Resources",
        ]
    )
    with t[0]:
        if not tab:
            st.info("No results for this condition/attacker combination.")
        else:
            df = pd.DataFrame(
                [
                    {
                        "variant": x["variant"],
                        "protocol": x["protocol"],
                        "model": x["model"],
                        "seeds": x["n_seeds"],
                        "accuracy": df_pm(x["accuracy"]),
                        "95% CI (t over seeds)": f"{x['accuracy']['ci_lo']:.3f}–{x['accuracy']['ci_hi']:.3f}",
                        "balanced acc.": df_pm(x["balanced_accuracy"]),
                        "macro F1": df_pm(x["f1_macro"]),
                        "ECE": df_pm(x["ece"]),
                        "val acc.": df_pm(x["val_accuracy"]),
                    }
                    for x in tab
                ]
            )
            st.dataframe(df, hide_index=True, width="stretch")
            fig = go.Figure()
            for p in sorted({x["protocol"] for x in tab}):
                g = [x for x in tab if x["protocol"] == p]
                fig.add_bar(
                    x=[f"{x['model']} [{x['variant']}]" if x["variant"] != "base" else x["model"] for x in g],
                    y=[x["accuracy"]["mean"] for x in g],
                    name=p,
                    error_y=dict(type="data", array=[x["accuracy"]["std"] for x in g]),
                )
            fig.add_hline(y=s["chance_accuracy"], line_dash="dot", annotation_text="chance")
            fig.update_yaxes(range=[0, 1], title="held-out accuracy (mean ± std over seeds)")
            ui.show(fig, "ev_bar")
            if s["model_comparisons"]:
                st.markdown("**Paired model comparisons** (per-seed accuracy, same splits)")
                st.dataframe(
                    pd.DataFrame(s["model_comparisons"]).drop(columns=["note"], errors="ignore"),
                    hide_index=True,
                    width="stretch",
                )
    sel_rows = [r for r in rows if r["condition"] == cond and r["attacker"] == att]
    labels = [f"{r['variant']} | {r['protocol']} | {r['model']} | seed {r['seed']}" for r in sel_rows]
    with t[1]:
        if sel_rows:
            i = int(
                st.selectbox("Run", range(len(sel_rows)), format_func=lambda k: labels[k], key="ev_cm") or 0
            )
            cm = np.array(sel_rows[i]["test"]["confusion_matrix"])
            cn = s["dataset"]["class_names"]
            fig = go.Figure(
                go.Heatmap(
                    z=cm / np.maximum(cm.sum(1, keepdims=True), 1),
                    x=cn,
                    y=cn,
                    text=cm,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title="row-normalised"),
                )
            )
            fig.update_layout(xaxis_title="predicted", yaxis_title="true", yaxis_autorange="reversed")
            ui.show(fig, "ev_cm_fig", height=430, legend=False)
            pc = pd.DataFrame(sel_rows[i]["test"]["per_class"]).T
            st.dataframe(pc, width="stretch")
    with t[2]:
        jobs = res["jobs"]
        j = st.selectbox("Training job", jobs, format_func=lambda j: j["job_id"], key="ev_job")
        ep = j["history"]["epochs"]
        if len(ep) > 1:
            fig = go.Figure()
            for k in ("train_loss", "val_loss"):
                if k in ep[0]:
                    fig.add_scatter(x=[e["epoch"] for e in ep], y=[e[k] for e in ep], name=k)
            if j["history"]["best_epoch"]:
                fig.add_vline(
                    x=j["history"]["best_epoch"], line_dash="dot", annotation_text="best (validation)"
                )
            fig.update_layout(xaxis_title="epoch", yaxis_title="loss")
            ui.show(fig, "ev_curve")
            st.caption(
                f"device {j['history']['device']} · {j['history']['seconds']:.1f}s · early stop: {j['history']['stopped_early']} · NaN recoveries: {j['history']['nan_recoveries']}. Selection and early stopping used the validation partition."
            )
        else:
            st.info("Classical model: single fit, no iterative training curve.")
    with t[3]:
        if sel_rows:
            i = int(
                st.selectbox("Run", range(len(sel_rows)), format_func=lambda k: labels[k], key="ev_cal") or 0
            )
            tr = sel_rows[i]["test"]
            rel = tr["reliability"]
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dot"))
            fig.add_scatter(
                x=[b["confidence"] for b in rel],
                y=[b["accuracy"] for b in rel],
                mode="lines+markers",
                name="model",
                marker=dict(size=[6 + 18 * b["count"] / max(x["count"] for x in rel) for b in rel]),
            )
            fig.update_layout(
                xaxis_title="mean confidence",
                yaxis_title="empirical accuracy",
                xaxis_range=[0, 1],
                yaxis_range=[0, 1],
            )
            ui.show(fig, "ev_cal_fig")
            c = st.columns(3)
            c[0].metric("ECE", f"{tr['ece']:.3f}")
            c[1].metric("Brier", f"{tr['brier']:.3f}")
            c[2].metric("NLL", f"{tr['nll']:.3f}")
    with t[4]:
        if sel_rows:
            df = pd.DataFrame(
                [
                    {
                        "label": f"{r['model']} · {r['protocol']}",
                        "seed": str(r["seed"]),
                        "accuracy": r["test"]["accuracy"],
                    }
                    for r in sel_rows
                ]
            )
            fig = go.Figure()
            for k, g in df.groupby("label"):
                fig.add_scatter(
                    x=[k] * len(g),
                    y=g["accuracy"],
                    mode="markers",
                    name=k,
                    text=g["seed"],
                    hovertemplate="seed %{text}: %{y:.3f}",
                )
            fig.update_yaxes(range=[0, 1], title="held-out accuracy per seed")
            ui.show(fig, "ev_seeds", legend=False)
            ci = pd.DataFrame(
                [
                    {
                        **{k: r[k] for k in ("variant", "protocol", "model", "seed")},
                        **{
                            f"{m} CI": f"{v['lo']:.3f}–{v['hi']:.3f}"
                            for m, v in r["test_ci"].items()
                            if m == "accuracy"
                        },
                        "CI method": r["test_ci"].get("accuracy", {}).get("method", "–"),
                        "clusters": r["test_ci"].get("accuracy", {}).get("n_clusters", "–"),
                    }
                    for r in sel_rows
                ]
            )
            st.markdown(
                "**Within-run bootstrap CIs** (cluster bootstrap over sessions: samples within a session are not independent)"
            )
            st.dataframe(ci, hide_index=True, width="stretch")
    with t[5]:
        if s["domain_shift"]:
            d = pd.DataFrame(
                [
                    {k: v for k, v in x.items() if k not in ("paired_test", "note")}
                    | {"p (paired t)": (x.get("paired_test") or {}).get("p_value")}
                    for x in s["domain_shift"]
                ]
            )
            st.dataframe(d, hide_index=True, width="stretch")
            st.caption(s["domain_shift"][0]["note"])
        else:
            st.info(
                "Domain-shift analysis needs both in_distribution and group-disjoint protocols (e.g. a cross_device or cross_session experiment)."
            )
    with t[6]:
        if s["resources"]:
            st.dataframe(pd.DataFrame(s["resources"]), hide_index=True, width="stretch")


# ============================================================================ Defense Laboratory
def page_defense() -> None:
    ui.header(
        "Defense Laboratory",
        "Measure how much class information survives a transformation, against static and adaptive attackers, and what it costs in audio quality.",
    )
    rec = _pick_experiment("def", with_defenses=True)
    if rec is not None:
        s = summary_of(rec["id"])
        ui.synthetic_notice(s["dataset"]["is_synthetic"])
        pts = s["defense_points"]
        if not pts:
            st.info(
                "This experiment has no defense conditions. Pick a defense_comparison or noise_robustness run."
            )
        else:
            models = sorted({p["model"] for p in pts})
            protos = sorted({p["protocol"] for p in pts})
            c = st.columns(2)
            mdl = c[0].selectbox("Model", models, key="df_model")
            pro = c[1].selectbox("Protocol", protos, key="df_proto")
            P = [p for p in pts if p["model"] == mdl and p["protocol"] == pro]
            front: list[str] = next(
                (x["frontier"] for x in s["pareto"] if x["model"] == mdl and x["protocol"] == pro), []
            )
            df = pd.DataFrame(P)
            t = st.tabs(
                [
                    "Accuracy degradation",
                    "Quality vs privacy (Pareto)",
                    "Effectiveness curves",
                    "Latency & overhead",
                    "Evidence",
                ]
            )
            with t[0]:
                fig = go.Figure()
                fig.add_bar(x=df["name"], y=df["gain_static"], name="static attacker")
                fig.add_bar(
                    x=df["name"],
                    y=df["gain_adaptive"],
                    name="adaptive attacker (retrained on defended audio)",
                )
                fig.add_hline(y=0, line_color="grey")
                fig.update_yaxes(title="accuracy drop vs clean baseline")
                ui.show(fig, "df_drop")
                st.caption(
                    "A defense that only fools a static attacker may be undone by retraining. Compare both bars; negative values mean accuracy did not fall."
                )
            with t[1]:
                fig = go.Figure()
                for flag, nm, sym in (
                    (True, "eligible (positive gain, significant)", "circle"),
                    (False, "not shown to be effective", "x"),
                ):
                    g = df[df["eligible"] == flag]
                    fig.add_scatter(
                        x=g["log_spectral_distance_db"],
                        y=g["privacy_gain"],
                        mode="markers+text",
                        text=g["name"],
                        textposition="top center",
                        textfont=dict(size=9),
                        name=nm,
                        marker=dict(size=10, symbol=sym),
                    )
                f = df[df["name"].isin(front)].sort_values("log_spectral_distance_db")
                fig.add_scatter(
                    x=f["log_spectral_distance_db"],
                    y=f["privacy_gain"],
                    mode="lines",
                    name="Pareto frontier",
                    line=dict(dash="dash"),
                )
                fig.update_layout(
                    xaxis_title="distortion: log-spectral distance (dB) — lower is better",
                    yaxis_title="privacy gain: worst-case accuracy drop — higher is better",
                )
                ui.show(fig, "df_pareto", height=440)
                st.markdown(
                    "**Pareto-optimal configurations:** "
                    + (", ".join(f"`{n}`" for n in front) if front else "none met the effectiveness criteria")
                )
            with t[2]:
                curves = [
                    c
                    for c in s["effectiveness_curves"]
                    if c["model"] == mdl
                    and c["protocol"] == pro
                    and any(p["x"] is not None for p in c["points"])
                ]
                if curves:
                    fig = go.Figure()
                    for c in curves:
                        xs = [p["x"] for p in c["points"]]
                        fig.add_scatter(
                            x=xs,
                            y=[p["static_accuracy"] for p in c["points"]],
                            mode="lines+markers",
                            name=f"{c['defense_type']} ({c['parameter']}) static",
                        )
                        if all(p["adaptive_accuracy"] is not None for p in c["points"]):
                            fig.add_scatter(
                                x=xs,
                                y=[p["adaptive_accuracy"] for p in c["points"]],
                                mode="lines+markers",
                                line=dict(dash="dash"),
                                name=f"{c['defense_type']} adaptive",
                            )
                    fig.add_hline(y=s["chance_accuracy"], line_dash="dot", annotation_text="chance")
                    fig.update_layout(
                        xaxis_title="defense parameter",
                        yaxis_title="held-out attacker accuracy",
                        yaxis_range=[0, 1],
                    )
                    ui.show(fig, "df_curves", height=420)
                else:
                    st.info("Curves need several parameter settings of the same defense type.")
            with t[3]:
                st.dataframe(
                    df[
                        [
                            "name",
                            "waveform_snr_db",
                            "log_spectral_distance_db",
                            "real_time_factor",
                            "latency_ms_per_sample",
                        ]
                    ].rename(
                        columns={
                            "waveform_snr_db": "waveform SNR (dB)",
                            "log_spectral_distance_db": "LSD (dB)",
                            "real_time_factor": "real-time factor",
                            "latency_ms_per_sample": "latency (ms/clip)",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            with t[4]:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "defense": p["name"],
                                **{f"{k} attacker": v for k, v in p["evidence"].items()},
                                "eligible": p["eligible"],
                            }
                            for p in P
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "significant_reduction: paired t-test over seeds, p<0.05, ≥3 seeds. Defenses are marked effective only if no attacker shows ‘no_significant_reduction’."
                )
    st.subheader("Run a new defense benchmark")
    a = st.columns([3, 2, 1])
    names = [d.name for d in default_defenses()]
    chosen = a[0].multiselect("Defenses", names, default=names[:6], key="dl_defs")
    mods = a[1].multiselect(
        "Attacker models",
        ["logreg", "random_forest", "svm", "cnn1d", "cnn2d"],
        default=["random_forest", "cnn1d"],
        key="dl_models",
    )
    seeds = a[2].text_input("Seeds", "0,1,2", key="dl_seeds")
    if st.button("Launch benchmark", key="dl_launch", disabled=not (chosen and mods)):
        try:
            base = preset_config("defense_comparison")
            cfg = be.config_from_form(
                base,
                name="Defense benchmark",
                models=[
                    {"name": m, "training": {"epochs": 12, "patience": 5}}
                    if MODEL_REGISTRY[m].family == "deep"
                    else {"name": m}
                    for m in mods
                ],
                defenses=[d.model_dump(mode="json") for d in default_defenses() if d.name in chosen],
                seeds=[int(x) for x in seeds.split(",") if x.strip()],
            )
            eid = be.launch(WS, cfg)
            st.success(f"Launched `{eid}` — follow progress in Experiment Studio.")
        except (OpenKeystrokeError, ValidationError, ValueError) as e:
            st.error(f"{getattr(e, 'message', e)}")


# ============================================================================ Registry
def page_registry() -> None:
    ui.header("Experiment Registry", "Search, inspect, compare, export and reproduce stored experiments.")
    f = st.columns(5)
    status = f[0].multiselect(
        "Status", ["CREATED", "VALIDATED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"], key="rg_status"
    )
    kind = f[1].selectbox("Type", ["(any)", *PRESET_KINDS], key="rg_kind")
    model = f[2].selectbox("Model", ["(any)", *MODEL_REGISTRY], key="rg_model")
    dsid = f[3].text_input("Dataset id contains", key="rg_ds")
    q = f[4].text_input("Search name / id", key="rg_q")
    rows: list[dict[str, Any]] = []
    for r in WS.store.list_experiments(
        kind=None if kind == "(any)" else kind,
        model=None if model == "(any)" else model,
        search=q or None,
        limit=500,
    ):
        if (not status or r["status"] in status) and (not dsid or dsid in (r["dataset_id"] or "")):
            rows.append(r)
    if not rows:
        ui.empty_state(
            "No experiments match", "Adjust the filters or launch an experiment from Experiment Studio."
        )
        return
    df = pd.DataFrame(rows)[
        ["id", "name", "kind", "status", "dataset_id", "dataset_is_synthetic", "created_at", "finished_at"]
    ]
    df["data"] = df.pop("dataset_is_synthetic").map({1: "synthetic", 0: "real"}).fillna("–")
    st.dataframe(df, hide_index=True, width="stretch", height=min(380, 38 + 35 * len(df)))
    ids = [r["id"] for r in rows]
    cmp_ids = st.multiselect("Compare runs", ids, key="rg_cmp")
    if len(cmp_ids) >= 1:
        tbl = []
        for e in cmp_ids:
            if WS.store.has_artifact(e, "summary.json"):
                for x in summary_of(e)["table"]:
                    if x["condition"] == "clean" and x["attacker"] == "static":
                        tbl.append(
                            {
                                "experiment": e[-17:],
                                "protocol": x["protocol"],
                                "model": x["model"],
                                "variant": x["variant"],
                                "accuracy": x["accuracy"]["mean"],
                                "std": x["accuracy"]["std"],
                                "seeds": x["n_seeds"],
                            }
                        )
        if tbl:
            tdf = pd.DataFrame(tbl)
            st.dataframe(tdf, hide_index=True, width="stretch")
            fig = go.Figure()
            for e, g in tdf.groupby("experiment"):
                fig.add_bar(
                    x=g["model"] + " · " + g["protocol"],
                    y=g["accuracy"],
                    name=e,
                    error_y=dict(type="data", array=g["std"]),
                )
            fig.update_yaxes(range=[0, 1], title="held-out accuracy")
            ui.show(fig, "rg_cmp_fig")
        else:
            st.info("Selected runs have no stored summary yet.")
        exp_c = st.columns(2)
        for fmt, col in (("csv", exp_c[0]), ("json", exp_c[1])):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as d:
                p = WS.store.export_metrics(cmp_ids, Path(d) / f"metrics.{fmt}", fmt)
                col.download_button(
                    f"Export metrics ({fmt.upper()})",
                    p.read_bytes(),
                    file_name=f"metrics.{fmt}",
                    mime="text/csv" if fmt == "csv" else "application/json",
                    key=f"rg_dl_{fmt}",
                )
    st.divider()
    eid = st.selectbox("Inspect", ids, key="rg_inspect")
    rec = WS.store.get(eid)
    st.markdown(f"{ui.badge(rec['status'])} &nbsp; **{rec['name']}**", unsafe_allow_html=True)
    ui.synthetic_notice(bool(rec["dataset_is_synthetic"]))
    t = st.tabs(["Provenance", "Configuration", "Metrics", "Events", "Artifacts", "Reproduce"])
    with t[0]:
        st.json({k: v for k, v in rec.items() if k not in ("config", "environment")})
        st.markdown("**Environment**")
        st.json(rec["environment"])
    with t[1]:
        st.code(yaml.safe_dump(rec["config"], sort_keys=False), language="yaml")
    with t[2]:
        m = pd.DataFrame(WS.store.metrics(eid, split="test"))
        if len(m):
            st.dataframe(
                m[m["name"].isin(["accuracy", "balanced_accuracy", "f1_macro", "ece"])],
                hide_index=True,
                width="stretch",
                height=320,
            )
        else:
            st.info("No metrics stored (yet).")
    with t[3]:
        for e in WS.store.events(eid):
            st.text(f"{e['ts']} {e['level']:<5} {e['message'][:400]}")
    with t[4]:
        arts = WS.store.list_artifacts(eid)
        if st.button("Verify artifact integrity (sha256)", key="rg_verify"):
            bad = WS.store.verify_artifacts(eid)
            (st.error if bad else st.success)(
                f"corrupted/missing: {bad}" if bad else f"all {len(arts)} artifacts verified"
            )
        st.dataframe(
            pd.DataFrame(arts)[["name", "kind", "size_bytes", "sha256"]] if arts else pd.DataFrame(),
            hide_index=True,
            width="stretch",
        )
        for a in arts:
            p = WS.store.artifacts_dir / a["relpath"]
            if a["kind"] == "png" and p.is_file():
                st.image(str(p), caption=a["name"], width=420)
            elif a["kind"] in ("json", "npz") and p.is_file() and p.stat().st_size < 50_000_000:
                st.download_button(
                    f"Download {a['name']}",
                    p.read_bytes(),
                    file_name=os.path.basename(a["name"]),
                    key=f"dl_{eid}_{a['name']}",
                )
    with t[5]:
        st.write(
            "Re-run this experiment from its immutable stored configuration; metrics should match within numerical tolerance."
        )
        if st.button("Reproduce (launch new run)", key="rg_repro"):
            try:
                new = be.relaunch(WS, eid)
                st.success(
                    f"Launched `{new}` (parent `{eid}`). Compare with: `openkeystroke experiment reproduce {eid}`"
                )
            except OpenKeystrokeError as e:
                st.error(f"{e.code}: {e.message}")


PAGES = {
    "Overview": page_overview,
    "Signal Explorer": page_signal,
    "Dataset Explorer": page_dataset,
    "Experiment Studio": page_studio,
    "Model Evaluation": page_eval,
    "Defense Laboratory": page_defense,
    "Experiment Registry": page_registry,
}

with st.sidebar:
    st.markdown("### OpenKeystroke")
    st.caption("Authorized acoustic research & privacy-defense evaluation")
    page = st.radio("Workspace", list(PAGES), key="nav", label_visibility="collapsed")
    st.divider()
    st.caption(f"Workspace: `{WS.root}`")
    st.caption("Recordings stay local. Nothing is uploaded.")

try:
    PAGES[page or "Overview"]()
except OpenKeystrokeError as exc:
    st.error(f"{exc.code}: {exc.message}")
    if st.button("Retry"):
        st.rerun()
