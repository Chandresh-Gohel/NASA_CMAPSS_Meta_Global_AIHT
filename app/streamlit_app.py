"""Engine Health Agent - Streamlit demo app.

    streamlit run app/streamlit_app.py

Tabs: Fleet (triage + planner agent), Engine (fan chart, decision, sensor drivers, fault injection),
Ask the agent (Muse Spark analyst + safety reviewer), Evaluation.
Every feature has a sidebar switch, so anything that breaks during a live demo can be hidden.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from ehagent.config import load_config, load_dotenv_if_present, path_for  # noqa: E402
from ehagent.constants import ACTION_LABELS, sensor_label  # noqa: E402
from ehagent.faults import FAULT_KINDS  # noqa: E402
from ehagent.upload import UploadError, mismatch_warning, parse_engine_file, suggest_subset  # noqa: E402
from ehagent.viz import ACTION_ICONS, CATEGORICAL, CHROME, DIVERGING, action_color, hex_to_rgba  # noqa: E402

load_dotenv_if_present()
st.set_page_config(page_title="Engine Health Agent", page_icon="🛠️", layout="wide")
CFG = load_config()


def theme_mode() -> str:
    try:
        t = st.context.theme.type
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return "dark" if st.get_option("theme.base") == "dark" else "light"


def stretch(fn, *args, **kwargs):
    """Full-width rendering across Streamlit versions (width='stretch' vs use_container_width)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def show_plot(fig: go.Figure) -> None:
    stretch(st.plotly_chart, fig, theme=None)


def action_text(action: str) -> str:
    return f"{ACTION_ICONS.get(action, '')} {ACTION_LABELS.get(action, action)}"


@st.cache_resource(show_spinner="Loading models and fleet data ...")
def get_service(multitask: bool):
    from ehagent.service import EngineService

    return EngineService(load_config(), multitask=multitask)


@st.cache_resource(show_spinner=False)
def get_orchestrator(multitask: bool):
    from ehagent.agent.orchestrator import AgentOrchestrator

    svc = get_service(multitask)
    return AgentOrchestrator(svc.cfg, svc)


def init_state() -> None:
    for key, value in {"faults": {}, "chat": [], "plan": None, "uploads": []}.items():
        st.session_state.setdefault(key, value)


APP_BG = {"light": "#ffffff", "dark": "#0e1117"}


def page_bg(mode: str) -> str:
    try:
        return st.get_option("theme.backgroundColor") or APP_BG[mode]
    except Exception:
        return APP_BG[mode]


def base_layout(fig: go.Figure, mode: str, height: int) -> None:
    """Transparent backgrounds so charts sit on the app's own surface in both themes."""
    ink = CHROME[mode]
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink["text2"], family="system-ui, -apple-system, Segoe UI, sans-serif"),
        hoverlabel=dict(bgcolor=ink["surface"], font_color=ink["text"]),
    )


def fan_chart(traj: pd.DataFrame, horizon: int, cap: float, mode: str) -> go.Figure:
    """Predicted RUL with its calibrated band, plus a decision strip underneath (same x axis)."""
    ink = CHROME[mode]
    blue = CATEGORICAL[mode][0]
    x = traj["cycle"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.86, 0.14], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=x, y=traj["hi"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=traj["lo"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=hex_to_rgba(blue, 0.12),
        name="90% band", customdata=traj[["hi"]].to_numpy(),
        hovertemplate="90% band %{y:.0f} to %{customdata[0]:.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=traj["mu"], mode="lines", line=dict(color=blue, width=2), name="Predicted RUL",
                             hovertemplate="Predicted RUL %{y:.1f}<extra></extra>"), row=1, col=1)
    if "rul_target" in traj.columns:
        fig.add_trace(go.Scatter(x=x, y=traj["rul_target"], mode="lines", line=dict(color=ink["text2"], width=2),
                                 name=f"Answer key (capped at {cap:.0f})",
                                 hovertemplate="Answer key, capped %{y:.0f}<extra></extra>"), row=1, col=1)
    if "rul_true" in traj.columns and bool((traj["rul_true"] > cap).any()):
        fig.add_trace(go.Scatter(x=x, y=traj["rul_true"], mode="lines",
                                 line=dict(color=ink["muted"], width=1, dash="dot"), opacity=0.7,
                                 name="Actual RUL (uncapped)",
                                 hovertemplate="Actual RUL, uncapped %{y:.0f}<extra></extra>"), row=1, col=1)
    last = traj.iloc[-1]
    fig.add_trace(go.Scatter(x=[last["cycle"]], y=[last["mu"]], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=10, color=blue, line=dict(color=page_bg(mode), width=2))), row=1, col=1)
    fig.add_hline(y=horizon, line=dict(color=ink["text2"], width=1, dash="dash"), row=1, col=1,
                  annotation_text=f"Horizon {horizon} cycles", annotation_position="top left",
                  annotation_font_color=ink["text2"])
    fig.add_hline(y=cap, line=dict(color=ink["muted"], width=1, dash="dot"), row=1, col=1,
                  annotation_text=f"RUL cap {cap:.0f}", annotation_position="top right",
                  annotation_font_color=ink["muted"])
    for action in ("INSPECT", "ABSTAIN", "NO_INSPECTION"):
        m = traj["action"] == action
        if m.any():
            fig.add_trace(go.Bar(x=x[m], y=[1] * int(m.sum()), width=1.0, marker=dict(color=action_color(action), line=dict(width=0)),
                                 name=action_text(action), hovertemplate=f"cycle %{{x}}: {ACTION_LABELS[action]}<extra></extra>"),
                          row=2, col=1)
    base_layout(fig, mode, 480)
    fig.update_layout(barmode="overlay", bargap=0, hovermode="x unified", margin=dict(l=64, r=10, t=64, b=48),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=ink["text2"])))
    fig.update_xaxes(showgrid=False, linecolor=ink["axis"], row=1, col=1)
    fig.update_xaxes(title_text="Cycle", showgrid=False, linecolor=ink["axis"], row=2, col=1)
    fig.update_yaxes(title_text="Remaining useful life (cycles)", gridcolor=ink["grid"], gridwidth=1, zeroline=False,
                     rangemode="tozero", row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1], title_text="Decision", row=2, col=1)
    return fig


def contributions_chart(contribs: list[dict], mode: str) -> go.Figure:
    """Signed sensor contributions (red lowers the estimate, blue raises it; sign printed too)."""
    ink = CHROME[mode]
    div = DIVERGING[mode]
    rows = list(reversed(contribs))
    vals = [float(c["contribution_cycles"]) for c in rows]
    lo_v, hi_v = min([0.0] + vals), max([0.0] + vals)
    pad = 0.25 * max(hi_v - lo_v, 1.0)
    fig = go.Figure(go.Bar(
        x=vals, y=[f"{c['name']} ({c['sensor']})" for c in rows], orientation="h", width=0.55,
        marker=dict(color=[div["neg"] if v < 0 else div["pos"] for v in vals]),
        text=[f"{v:+.1f}" for v in vals], textposition="outside", textfont=dict(color=ink["text"]), cliponaxis=False,
        customdata=[c["description"] for c in rows], hovertemplate="%{y}: %{x:+.1f} cycles<br>%{customdata}<extra></extra>",
    ))
    base_layout(fig, mode, 70 + 34 * len(rows))
    fig.update_layout(margin=dict(l=10, r=48, t=10, b=40), showlegend=False)
    fig.update_xaxes(title_text="Contribution to the RUL estimate vs a healthy engine (cycles)", gridcolor=ink["grid"],
                     zeroline=True, zerolinecolor=ink["axis"],
                     range=[lo_v - (pad if lo_v < 0 else 0.0), hi_v + (pad if hi_v > 0 else 0.0)])
    fig.update_yaxes(showgrid=False)
    return fig


def decision_card(d: dict) -> None:
    st.markdown(f"### {action_text(d['action'])}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated RUL", f"{d['rul_mean']:.0f} cycles")
    band = f"{d['rul_lo']:.0f} to {d['rul_hi']:.0f}" + ("+" if d["healthy_region"] else "")
    c2.metric(f"{round(100 * d['interval_coverage'])}% band", f"{band} cycles")
    c3.metric(f"P(fail within {d['horizon_cycles']} cycles)", f"{100 * d['p_fail_within_horizon']:.1f}%")
    ep = d.get("epistemic_percentile")
    gate = float(CFG.decision.epistemic_percentile_abstain)
    c4.metric("Unfamiliarity percentile", "n/a" if ep is None else f"{ep:.0f}",
              help=f"Epistemic uncertainty ranked against held-out calibration engines. At {gate:.0f} or above the system abstains.")
    st.write(d["resolution"])
    if d["healthy_region"]:
        st.caption("The band's upper edge reaches the RUL cap: the engine is in the healthy region, and remaining life "
                   "above the cap can't be estimated from sensor data.")
    for f in d.get("sensor_flags", []):
        st.warning(f"⚠️ {f['name']} ({f['sensor']}): {f['check'].replace('_', ' ')}. {f['detail']}.")


def render_answer(res: dict, by_muse: bool = True) -> None:
    if res.get("error"):
        st.error(res["error"])
        return
    rep = res["report"]
    action = res["final_action"]
    st.markdown(f"**{action_text(action)}: {rep['headline']}**")
    st.caption("Written by Muse Spark." if not res.get("used_fallback") else "Shown with the deterministic template.")
    st.write(rep["summary"])
    if rep.get("abstain_reason"):
        st.warning(rep["abstain_reason"])
    if rep.get("key_numbers"):
        st.table(pd.DataFrame(rep["key_numbers"]).reindex(columns=["label", "value", "unit", "source"]))
    if rep.get("sensor_drivers"):
        st.table(pd.DataFrame(rep["sensor_drivers"]).reindex(columns=["name", "sensor", "contribution_cycles", "plain_language"]))
    if rep.get("next_steps"):
        st.markdown("**Next steps**\n" + "\n".join(f"- {s}" for s in rep["next_steps"]))
    for c in rep.get("caveats", []):
        st.caption(c)
    rv = res.get("review")
    if rv:
        icon = {"approve": "✅", "revise": "✏️", "force_abstain": "⚠️"}.get(rv["verdict"], "")
        st.markdown(f"**Safety reviewer:** {icon} {rv['verdict'].replace('_', ' ')}. {rv.get('rationale', '')}")
    if res.get("used_fallback"):
        st.info("Shown with the deterministic template (agent off or unavailable, or its draft failed the number check).")
    if res.get("violations"):
        with st.expander("Number-check findings on the first draft"):
            for v in res["violations"]:
                st.write(f"- {v}")
    for e in res.get("errors", []):
        st.caption(f"Agent note: {e}")
    with st.expander("Agent trace: tool calls and results"):
        for t in res.get("trace", []):
            st.markdown(f"`{t['tool']}` `{json.dumps(t['arguments'])}`")
            st.json(t["result"], expanded=False)


init_state()
mode = theme_mode()
features = CFG.features
st.sidebar.title("Engine Health Agent")
st.sidebar.caption("Calibrated RUL, safe decisions and Muse Spark agents on NASA C-MAPSS (simulated engines).")
multitask = st.sidebar.checkbox("Multitask GP across subsets", value=bool(features.multitask_model))
agents_on = st.sidebar.checkbox("Muse Spark agents", value=bool(features.agents))
reviewer_on = st.sidebar.checkbox("Safety reviewer agent", value=bool(features.reviewer), disabled=not agents_on)
planner_on = st.sidebar.checkbox("Planner agent", value=bool(features.planner_agent), disabled=not agents_on)
faults_on = st.sidebar.checkbox("Fault injection", value=bool(features.fault_injection))
show_truth = st.sidebar.checkbox("Show answer key (true RUL)", value=bool(features.show_answer_key))
default_h = int(CFG.decision.horizon_cycles)
horizon_ui = st.sidebar.slider("Planning horizon (cycles)", 10, 100, default_h, step=5)
H = None if horizon_ui == default_h else horizon_ui

try:
    service = get_service(multitask)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Missing artifacts for this model: {exc}")
    st.info("Run the pipeline first (README, 'Run order'): prepare -> baseline -> dkl -> calibrate -> precompute.")
    st.stop()
orch = get_orchestrator(multitask)
subset_choice = st.sidebar.selectbox("Fleet subset", ["ALL"] + service.subsets)
subsets = None if subset_choice == "ALL" else [subset_choice]
api_state = (f"connected ({orch.client.model})" if orch.client.available else "offline: cached responses or templates")
st.sidebar.caption(f"Model: {service.predictor.tag} | Muse Spark: {api_state}")
if st.session_state.faults:
    st.sidebar.warning(f"{len(st.session_state.faults)} engine(s) with injected sensor faults")
    if st.sidebar.button("Clear all faults"):
        st.session_state.faults = {}
        st.rerun()

tab_fleet, tab_engine, tab_agent, tab_eval = st.tabs(["Fleet", "Engine", "Ask the agent", "Evaluation"])

with tab_fleet:
    rows = service.fleet(subsets, horizon=H, faults=st.session_state.faults)
    counts = pd.Series([r["action"] for r in rows]).value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engines", len(rows))
    c2.metric(action_text("INSPECT"), int(counts.get("INSPECT", 0)))
    c3.metric(action_text("ABSTAIN"), int(counts.get("ABSTAIN", 0)))
    c4.metric(action_text("NO_INSPECTION"), int(counts.get("NO_INSPECTION", 0)))
    table = pd.DataFrame([{
        "engine": r["engine_id"],
        "decision": action_text(r["action"]),
        "urgency": r["urgency"],
        "RUL (cycles)": r["rul_mean"],
        "90% band": f"{r['rul_lo']:.0f} to {r['rul_hi']:.0f}" + ("+" if r["healthy_region"] else ""),
        f"P(fail within {r['horizon_cycles']})": f"{100 * r['p_fail_within_horizon']:.1f}%",
        "why": ", ".join(x.replace("_", " ") for x in r["reasons"]),
    } for r in rows])
    stretch(st.dataframe, table, hide_index=True, height=420)
    st.caption("Table view: same fleet ranking as above (accessible twin of the tiles).")
    st.caption("These are NASA's official C-MAPSS test engines. The model never saw them in training, tuning or calibration.")

    st.markdown("#### Next week's maintenance plan")
    pc1, pc2, pc3 = st.columns(3)
    days = pc1.number_input("Days", 1, 30, int(CFG.planner.horizon_days))
    slots = pc2.number_input("Hangar inspection slots per day", 1, 20, int(CFG.planner.slots_per_day))
    review_slots = pc3.number_input("Engineering-review slots per day", 1, 20, int(CFG.planner.review_slots_per_day))
    if st.button("Build plan" + (" with the planner agent" if agents_on and planner_on else " (deterministic template)")):
        with st.spinner("Planning ..."):
            st.session_state.plan = orch.plan(faults=st.session_state.faults, horizon=H, days=int(days),
                                              slots_per_day=int(slots), review_slots_per_day=int(review_slots),
                                              subset=subset_choice, use_agents=agents_on and planner_on)
    res = st.session_state.plan
    if res and res.get("plan"):
        plan = res["plan"]
        st.write(plan["summary"])
        st.caption("Planner: " + ("Muse Spark agent (reconciled with the deterministic schedule)." if not res.get("used_fallback") else "deterministic template."))
        if plan["work_orders"]:
            stretch(st.dataframe, pd.DataFrame(plan["work_orders"]), hide_index=True)
        for w in plan.get("capacity_warnings", []):
            st.warning(w)
        if res.get("used_fallback"):
            st.caption("Work orders written from the deterministic schedule (planner agent off or unavailable).")
        for v in res.get("violations", []):
            st.caption(f"Reconciled: {v}")

with tab_engine:
    upload_on = bool(features.get("engine_upload", False))
    pool_options = ["Fleet engines (latest data)", "Replay engines (held out, full life)"]
    if upload_on:
        pool_options.append("Upload an engine file")
    pool = st.radio("Engines", pool_options, horizontal=True)
    engine_id = None
    is_up = False
    split = "test"
    if upload_on and pool == "Upload an engine file":
        up = st.file_uploader(
            "Engine file", type=["txt", "csv"],
            help="Same format as NASA's test_FD00x.txt: 26 columns (unit, cycle, 3 operating settings, 21 sensors), space- or comma-separated.")
        if up is None:
            st.info("Upload an engine file to run the full pipeline on it: decision, band, explanation, faults and briefing.")
        else:
            data = up.getvalue()
            try:
                udf = parse_engine_file(data, CFG)
            except UploadError as exc:
                st.error(str(exc))
            else:
                file_digest = hashlib.sha256(data).hexdigest()[:6]
                suggested, why = suggest_subset(udf, service.datasets)
                st.caption(f"Suggested engine type: {suggested} ({why}).")
                type_order = ["FD001", "FD002", "FD003", "FD004"]
                type_descr = {
                    "FD001": "FD001: one operating condition, one fault mode",
                    "FD002": "FD002: six conditions, one fault mode",
                    "FD003": "FD003: one condition, two fault modes",
                    "FD004": "FD004: six conditions, two fault modes",
                }
                subset_up = st.selectbox(
                    "Engine type", type_order,
                    index=type_order.index(suggested) if suggested in type_order else 0,
                    format_func=lambda s: type_descr.get(s, s), key=f"upload_subset_{file_digest}")
                warn = mismatch_warning(udf, subset_up, service.datasets)
                if warn:
                    st.warning(warn)
                try:
                    ids_up = service.add_upload(udf, subset_up)
                except (KeyError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    known = [e for e in st.session_state.get("uploads", []) if service.has_engine(e)]
                    for e in ids_up:
                        if e not in known:
                            known.append(e)
                    st.session_state.uploads = known
                    if known:
                        engine_id = st.selectbox("Engine", known, key=f"upload_engine_{file_digest}_{subset_up}")
                        st.session_state["current_engine"] = engine_id
                        is_up = True
    else:
        split = "test" if pool.startswith("Fleet") else "sim"
        if split == "sim":
            st.caption("Held-out engines from the training files, run to failure. Never used for training or tuning.")
        ids = [e for s in (subsets or service.subsets) for e in service.datasets[s].splits[split]]
        if split == "test" and rows:
            ranked = [r["engine_id"] for r in rows]
            seen = set(ranked)
            ids = ranked + [e for e in ids if e not in seen]
        if ids:
            engine_id = st.selectbox("Engine", ids, key="engine_select")
            st.session_state["current_engine"] = engine_id
    if engine_id is not None:
        if not is_up:
            try:
                raw_text = service.export_raw(engine_id)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                st.caption(f"Download unavailable: {exc}")
            else:
                st.download_button("Download this engine's data", raw_text, file_name=f"{engine_id}.txt")
        fault = st.session_state.faults.get(engine_id)
        if fault:
            st.warning(f"Injected fault: {fault['kind']} on {sensor_label(fault['sensor'])}, starting {fault['start_offset']} cycles ago.")
        analysis = service.analyze(engine_id, horizon=H, fault=fault)
        decision_card(analysis)

        truth = False if is_up else (split == "sim" or show_truth)
        if is_up:
            st.caption("Uploaded engine: the model has never seen it, and there is no answer key.")
        traj = service.trajectory(engine_id, fault=fault, horizon=H, include_truth=truth)
        if truth and split == "test":
            st.caption(f"Answer key: true RUL at the latest cycle is {service.answer_key(engine_id):.0f} cycles.")
        st.markdown(f"#### {engine_id}: remaining useful life over time")
        if "rul_true" in traj.columns:
            traj["rul_target"] = traj["rul_true"].clip(upper=float(CFG.data.rul_cap))
        show_plot(fan_chart(traj, int(H or default_h), float(CFG.data.rul_cap), mode))
        if "rul_target" in traj.columns:
            st.caption("Above 125 cycles healthy engines look alike, so the model reports 'healthy' instead of guessing. "
                       "The solid answer-key line is capped at 125 (what the model aims at); the faint dotted line is the uncapped truth.")
        with st.expander("Table view of this chart"):
            stretch(st.dataframe, traj, hide_index=True)

        left, right = st.columns([3, 2])
        with left:
            st.markdown("#### Which sensors drive this estimate")
            with st.spinner("Integrated Gradients ..."):
                ex = service.explain(engine_id, fault=fault)
            show_plot(contributions_chart(ex["contributions"], mode))
            f = ex["faithfulness"]
            st.caption(f"Red lowers the estimate, blue raises it (vs an average healthy engine at {ex['healthy_reference_rul']:.0f} cycles). "
                       f"Faithfulness check: resetting the top {f['top_k']} sensors moves the estimate {f['top_k_change_cycles']} cycles, "
                       f"random sensors {f['random_k_change_cycles']} cycles.")
        with right:
            if faults_on:
                st.markdown("#### Inject a sensor fault")
                sd = service.datasets[service.subset_of(engine_id)]
                sensor = st.selectbox("Sensor", sd.sensors, format_func=sensor_label,
                                      index=sd.sensors.index(str(CFG.faults.default_sensor)) if str(CFG.faults.default_sensor) in sd.sensors else 0)
                kind = st.selectbox("Fault", list(FAULT_KINDS))
                magnitude = st.slider("Magnitude (training standard deviations)", 0.0, 10.0,
                                      float(CFG.faults.magnitudes.get(kind, 3.0)), 0.5, disabled=kind == "stuck")
                offset = st.slider("Started this many cycles ago", 1, 60, int(CFG.faults.default_start_offset))
                b1, b2 = st.columns(2)
                if b1.button("Inject fault"):
                    st.session_state.faults[engine_id] = {"sensor": sensor, "kind": kind, "magnitude": float(magnitude), "start_offset": int(offset)}
                    st.rerun()
                if b2.button("Clear fault", disabled=fault is None):
                    st.session_state.faults.pop(engine_id, None)
                    st.rerun()

with tab_agent:
    engine_q = st.session_state.get("current_engine")
    st.markdown(f"Asking about **{engine_q}** (change it in the Engine tab).")
    suggestions = ["Should we inspect this engine?", "Why is the system not making a call?",
                   "Which sensors are driving this, in plain words?", "What if the next check is 60 cycles away?"]
    cols = st.columns(len(suggestions))
    clicked = None
    for c, q in zip(cols, suggestions):
        if c.button(q):
            clicked = q
    with st.form("ask_form", clear_on_submit=True):
        typed = st.text_input("Your question")
        submitted = st.form_submit_button("Ask")
    question = clicked or (typed if submitted and typed.strip() else None)
    if question and engine_q:
        with st.spinner("Muse Spark is gathering evidence ..." if agents_on else "Building the briefing ..."):
            res = orch.ask(question, engine_q, faults=st.session_state.faults,
                           horizon=H, use_agents=agents_on, use_reviewer=agents_on and reviewer_on)
        st.session_state.chat.append(res)
    for res in reversed(st.session_state.chat[-6:]):
        with st.container(border=True):
            st.markdown(f"**Q ({res['engine_id']}):** {res['question']}")
            render_answer(res, by_muse=agents_on)
    if st.session_state.chat:
        st.caption(f"Muse Spark usage since the app started: {orch.client.usage}")

with tab_eval:
    mpath = path_for(CFG, "reports_dir", "metrics.json")
    if not mpath.exists():
        st.info("Run `python -m ehagent.evaluate` to produce the evaluation report.")
    else:
        metrics = json.loads(mpath.read_text())
        st.caption(f"Model {metrics.get('model')} | generated {metrics.get('generated_at')}")
        test = metrics.get("test", {})
        st.markdown("#### Accuracy and calibration on the official test engines")
        stretch(st.dataframe, pd.DataFrame([{
            "subset": s,
            "RMSE (capped truth)": round(m["rmse_capped_truth"], 2),
            "RMSE (uncapped truth)": round(m["rmse_uncapped_truth"], 2),
            "NASA score (capped)": round(m["nasa_score_capped_truth"], 1),
            "RMSE true RUL <= 30": None if m["rmse_true_rul_le_30"] is None else round(m["rmse_true_rul_le_30"], 2),
            "90% band coverage": round(m["coverage"], 3),
            "mean band width": round(m["mean_band_width"], 1),
            "CRPS": round(m["crps_calibrated"], 2),
        } for s, m in test.items()]), hide_index=True)
        pcmp = metrics.get("policy_comparison")
        if pcmp:
            st.markdown("#### Maintenance policies replayed on held-out engines")
            stretch(st.dataframe, pd.DataFrame(pcmp["summary"]).T)
        for name in ("policy_comparison.png", "risk_coverage.png", "coverage_by_rul.png"):
            p = path_for(CFG, "reports_dir", name)
            if p.exists():
                st.image(str(p))
        for key, title in (("ood", "Unfamiliar engines (cross-subset test)"), ("fault_injection", "Sensor fault test"),
                           ("attribution", "Explanation faithfulness"), ("ablation", "Single-task vs multitask")):
            if metrics.get(key):
                with st.expander(title):
                    st.json(metrics[key])
