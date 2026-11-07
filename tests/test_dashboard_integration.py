import os
import time

import pytest
from streamlit.testing.v1 import AppTest

from conftest import SMALL, make_cfg
from openkeystroke.dashboard import backend as be
from openkeystroke.errors import RegistryError
from openkeystroke.experiments import LocalExecutor, create_experiment, run_experiment
from openkeystroke.registry import Workspace

APP = os.path.join(os.path.dirname(__file__), "..", "src", "openkeystroke", "dashboard", "app.py")
PAGES = [
    "Overview",
    "Signal Explorer",
    "Dataset Explorer",
    "Experiment Studio",
    "Model Evaluation",
    "Defense Laboratory",
    "Experiment Registry",
]


def wait(ws, eid, timeout=180):
    t = time.time()
    while time.time() - t < timeout:
        st = ws.store.get(eid)["status"]
        if st in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
        time.sleep(0.5)
    raise TimeoutError(eid)


@pytest.fixture()
def populated(tmp_path, monkeypatch):
    from openkeystroke.config import DefenseConfig, DefenseStep

    ws = Workspace(tmp_path / "h")
    be.generate_demo_dataset(ws, SMALL)
    cfg = make_cfg(
        "defense_comparison",
        defenses=[
            DefenseConfig(name="lp1k", steps=[DefenseStep(type="lowpass", params={"cutoff_hz": 1000})]),
            DefenseConfig(name="lp2k", steps=[DefenseStep(type="lowpass", params={"cutoff_hz": 2000})]),
        ],
    )
    eid = create_experiment(ws, cfg)
    assert run_experiment(ws, eid) == "COMPLETED"
    monkeypatch.setenv("OPENKEYSTROKE_HOME", str(ws.root))
    return ws, eid


def app():
    return AppTest.from_file(APP, default_timeout=120)


def test_every_page_renders_real_stored_results(populated):
    ws, eid = populated
    at = app().run()
    assert not at.exception
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])
    at.sidebar.radio[0].set_value("Defense Laboratory").run()
    assert len(at.get("plotly_chart")) >= 3  # degradation, pareto, curves rendered from the stored summary
    at.sidebar.radio[0].set_value("Model Evaluation").run()
    assert at.dataframe and any(
        "synthetic" in str(m.value).lower() for m in at.markdown
    )  # labelled synthetic


def test_overview_numbers_come_from_registry(populated):
    ws, eid = populated
    at = app().run()
    vals = {m.label: m.value for m in at.metric}
    assert (
        vals["Experiments"] == "1"
        and vals["Completed"] == "1"
        and vals["Datasets"] == "1"
        and vals["Failed"] == "0"
    )


def test_empty_workspace_shows_empty_states_not_fake_data(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENKEYSTROKE_HOME", str(tmp_path / "empty"))
    at = app().run()
    assert not at.exception and {m.label: m.value for m in at.metric}["Experiments"] == "0"
    assert any("Nothing here yet" in str(m.value) for m in at.markdown)
    for page in ("Model Evaluation", "Defense Laboratory", "Experiment Registry", "Signal Explorer"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception and not at.get("plotly_chart")


@pytest.mark.slow
def test_launch_and_inspect_experiment_through_dashboard(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "h")
    monkeypatch.setenv("OPENKEYSTROKE_HOME", str(ws.root))
    at = app().run()
    at.sidebar.radio[0].set_value("Experiment Studio").run()
    at.selectbox(key="st_kind").set_value("single").run()
    at.number_input(key="sy_c").set_value(3).run()
    at.number_input(key="sy_d").set_value(3).run()
    at.number_input(key="sy_s").set_value(2).run()
    at.number_input(key="sy_n").set_value(5).run()
    at.multiselect(key="st_models_single").set_value(["logreg"]).run()
    at.text_input(key="st_name").set_value("dashboard-launched").run()
    assert not at.exception and not at.error
    at.button(key="st_launch").click().run()
    assert not at.exception and any("Launched" in s.value for s in at.success)
    runs = ws.store.list_experiments()
    assert len(runs) == 1 and runs[0]["name"] == "dashboard-launched"
    assert wait(ws, runs[0]["id"]) == "COMPLETED"  # executed by a separate worker process
    at = app().run()
    at.sidebar.radio[0].set_value("Experiment Registry").run()
    assert not at.exception and at.selectbox(key="rg_inspect").value == runs[0]["id"]
    at.sidebar.radio[0].set_value("Model Evaluation").run()
    assert not at.exception and any("dashboard-launched" in o for o in at.selectbox(key="pick_eval").options)


@pytest.mark.slow
def test_dashboard_cancel_and_concurrency_cap(tmp_path):
    ws = Workspace(tmp_path / "h")
    cfg = make_cfg("repeated_seed", seeds=[0, 1, 2, 3, 4, 5, 6, 7]).model_copy(update={"name": "long"})
    e1, e2 = create_experiment(ws, cfg), create_experiment(ws, cfg)
    ex = LocalExecutor(ws, max_concurrent=1)
    ex.submit(e1)
    with pytest.raises(RegistryError, match="concurrency limit"):
        ex.submit(e2)
    t = time.time()
    while ws.store.get(e1)["status"] != "RUNNING" and time.time() - t < 60:
        time.sleep(0.3)
    assert ws.store.request_cancel(e1) in ("CANCEL_REQUESTED", "CANCELLED")
    assert wait(ws, e1) in ("CANCELLED", "COMPLETED")
    ex.submit(e2, wait_for_slot=True)  # slot freed
    ws.store.request_cancel(e2)
    wait(ws, e2)
