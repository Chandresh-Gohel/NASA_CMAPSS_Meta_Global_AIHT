"""Step 13 addition exit check: uploaded engine files run the whole pipeline.

Offline (default): CPU, <2 min, no API key.
  python scripts/check_upload.py [--live]

With --live (needs MODEL_API_KEY, cached): one live ask on an upload, then a
cached rerun. A single-engine call only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_examples(service):
    fleet = service.fleet()
    out: dict[str, str] = {}
    for row in fleet:
        out.setdefault(row["action"], row["engine_id"])
        if len(out) == 3:
            break
    assert set(out) == {"INSPECT", "ABSTAIN", "NO_INSPECTION"}, f"need all 3 actions, got {out}"
    subs = {service.subset_of(e) for e in out.values()}
    if not (subs & {"FD002", "FD004"}):
        for row in fleet:
            if service.subset_of(row["engine_id"]) in ("FD002", "FD004"):
                out[f"EXTRA_{row['action']}"] = row["engine_id"]
                break
    return out


def check_round_trip(service, examples: dict[str, str]) -> list[str]:
    from ehagent.upload import parse_engine_file

    ids: list[str] = []
    for action, eid in examples.items():
        subset = service.subset_of(eid)
        raw = service.export_raw(eid)
        df = parse_engine_file(raw.encode("utf-8"), service.cfg)
        got = service.add_upload(df, subset)
        assert len(got) == 1, f"{eid}: expected 1 upload, got {got}"
        uid = got[0]
        ids.append(uid)
        ref = service.analyze(eid)
        new = service.analyze(uid)
        assert new["action"] == ref["action"], f"{eid}: {new['action']} != {ref['action']}"
        assert new["reasons"] == ref["reasons"], f"{eid}: {new['reasons']} != {ref['reasons']}"
        for k in ("rul_mean", "rul_lo", "rul_hi"):
            assert abs(float(new[k]) - float(ref[k])) <= 0.1, f"{eid}: {k} {new[k]} != {ref[k]}"
        assert abs(float(new["p_fail_within_horizon"]) - float(ref["p_fail_within_horizon"])) <= 0.001, (
            f"{eid}: p {new['p_fail_within_horizon']} != {ref['p_fail_within_horizon']}")
        log(f"  round trip {action} {eid} -> {uid}: action={new['action']} reasons={new['reasons']}")
    return ids


def check_ids(service, examples: dict[str, str]) -> None:
    from ehagent.upload import parse_engine_file

    eid = next(iter(examples.values()))
    subset = service.subset_of(eid)
    raw = service.export_raw(eid)
    before = service.fleet()
    n_up = len(service.uploads)
    again = service.add_upload(parse_engine_file(raw.encode("utf-8"), service.cfg), subset)
    assert len(service.uploads) == n_up, "repeat upload added engines"
    other_subset = "FD002" if subset != "FD002" else "FD001"
    other = service.add_upload(parse_engine_file(raw.encode("utf-8"), service.cfg), other_subset)
    assert other[0] != again[0], f"subset change kept the ID: {other[0]}"
    assert service.fleet() == before, "fleet() changed after uploading"
    log(f"  ids: repeat same ({again[0]}), other subset differs ({other[0]}), fleet unchanged ({len(before)} rows)")


def check_wrong_type(service) -> str:
    from ehagent.upload import mismatch_warning, parse_engine_file, suggest_subset

    eids = service.engine_ids("FD004", "test")
    assert eids, "no FD004 test engines"
    eid = eids[0]
    df = parse_engine_file(service.export_raw(eid).encode("utf-8"), service.cfg)
    suggested, _ = suggest_subset(df, service.datasets)
    assert suggested in ("FD002", "FD004"), f"suggest_subset gave {suggested}"
    warn = mismatch_warning(df, "FD001", service.datasets)
    assert warn, "mismatch_warning did not fire for FD004-as-FD001"
    uid = service.add_upload(df, "FD001")[0]
    a = service.analyze(uid)
    log(f"  wrong type {eid} as FD001 -> {uid}: suggest={suggested} warned=True action={a['action']} reasons={a['reasons']}")
    assert a["action"] == "ABSTAIN", f"FD004-as-FD001 gave {a['action']} (no threshold changes allowed)"
    return uid


def check_no_truth(service, uid: str) -> None:
    from ehagent.agent.tools import ToolExecutor

    traj = service.trajectory(uid, include_truth=True)
    assert "rul_true" not in traj.columns, "upload trajectory leaks rul_true"
    try:
        service.answer_key(uid)
    except KeyError:
        pass
    else:
        raise AssertionError("answer_key did not raise KeyError for an upload")
    ex = ToolExecutor(service)
    calls = [("get_engine_status", {"engine_id": uid}),
              ("explain_prediction", {"engine_id": uid, "top_k": 3}),
              ("check_sensors", {"engine_id": uid}),
              ("what_if_horizon", {"engine_id": uid, "horizon_cycles": 30})]
    for name, args in calls:
        res = ex.execute(name, args)
        assert "error" not in res, f"{name}: {res.get('error')}"
        json.dumps(res, allow_nan=False)
    log(f"  no truth {uid}: no rul_true, answer_key KeyError, 4 tools JSON-safe")


def check_bad_files(cfg) -> None:
    from ehagent.upload import UploadError, parse_engine_file

    def line(unit, cycle, op="42.0 0.84 100.0", sensors=" ".join(["0.0"] * 21)):
        return f"{unit} {cycle} {op} {sensors}"

    good_row = line(1, 1)
    too_many = int(cfg.upload.max_engines) + 1
    cases = {
        "column count": b"1 2 3\n",
        "non-numeric": (line(2, 1) + "\n" + line(2, 2).replace("42.0", "abc", 1) + "\n").encode(),
        "decreasing cycles": ("\n".join([line(1, c) for c in list(range(1, 20)) + [10]]) + "\n").encode(),
        "10 cycles": ("\n".join(line(1, c) for c in range(1, 11)) + "\n").encode(),
        "too many engines": ("\n".join(line(u, c) for u in range(1, too_many + 1) for c in range(1, 16)) + "\n").encode(),
        "empty file": b"",
    }
    assert good_row
    for name, blob in cases.items():
        try:
            parse_engine_file(blob, cfg)
        except UploadError as exc:
            assert str(exc).strip(), f"{name}: empty message"
            log(f"  bad file {name}: UploadError ({exc})")
        except Exception as exc:  # noqa: BLE001 - the point is that nothing else raises
            raise AssertionError(f"{name}: raised {type(exc).__name__}, not UploadError") from exc
        else:
            raise AssertionError(f"{name}: parsed without error")


def check_agents_offline(cfg, service, uid: str) -> None:
    from ehagent.agent.guard import check_report
    from ehagent.agent.orchestrator import AgentOrchestrator
    from ehagent.agent.templates import template_report
    from ehagent.agent.tools import ToolExecutor

    # The disk cache is shared with live runs, so disable it here: this check must
    # deterministically exercise the template path, never replay a cached briefing.
    orch = AgentOrchestrator(cfg, service)
    orch.client.use_cache = False
    res = orch.ask("Should we inspect this engine?", uid)
    assert not res.get("error"), res.get("error")
    assert res.get("report"), "no report"
    assert res["used_fallback"] is True, "expected the offline template"
    assert res["report"]["recommendation"] == res["authoritative_action"]
    ex = ToolExecutor(service)
    status = ex.execute("get_engine_status", {"engine_id": uid})
    action = status["action"]
    ex.execute("explain_prediction", {"engine_id": uid, "top_k": 3})
    if action == "ABSTAIN" or status.get("sensor_flags"):
        ex.execute("check_sensors", {"engine_id": uid})
    ev = {}
    for t in ex.trace:
        if "error" not in (t.get("result") or {}):
            ev[t["tool"]] = t["result"]
    ev["authoritative_action"] = action
    ev["thresholds"] = {"p_inspect": float(cfg.decision.p_inspect), "p_clear": float(cfg.decision.p_clear),
                        "rul_cap": float(cfg.data.rul_cap),
                        "interval_coverage": round(1 - float(cfg.calibration.alpha), 2)}
    violations = check_report(template_report(ev, action, cfg), action, ev, cfg)
    assert not violations, f"template failed guard: {violations}"
    log(f"  agents offline {uid}: fallback=True recommendation={res['report']['recommendation']} guard clean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    # Offline checks always run with the API off, even under --live (which flips the
    # flag back for its own orchestrator only).
    os.environ["EHA_OFFLINE"] = "1"

    from ehagent.config import load_config
    from ehagent.service import EngineService

    cfg = load_config()
    failures: list[str] = []
    service = EngineService(cfg)
    log(f"service ready: {service.predictor.tag} uploads kept={service.keep_last}")

    examples = pick_examples(service)
    log(f"examples: {examples}")
    try:
        check_round_trip(service, examples)
        check_ids(service, examples)
        uid = check_wrong_type(service)
        check_no_truth(service, uid)
        check_bad_files(cfg)
        check_agents_offline(cfg, service, uid)
    except (AssertionError, KeyError, ValueError) as exc:
        failures.append(f"{type(exc).__name__}: {exc}")

    if args.live:
        key = os.environ.get(str(cfg.agent.api_key_env), "") or os.environ.get("META_API_KEY", "")
        if not key:
            print("MODEL_API_KEY is not set. Set it (outside .env edits here) and rerun --live.")
            sys.exit(2)
        from ehagent.agent.orchestrator import AgentOrchestrator

        os.environ["EHA_OFFLINE"] = "0"
        orch = AgentOrchestrator(cfg, service)
        assert orch.client.available, "no API client"
        res = orch.ask("Should we inspect this engine?", uid)
        print(f"  live: used_fallback={res['used_fallback']} review={res.get('review', {}).get('verdict')}")
        print(f"  live: first-draft violations={res.get('violations')}")
        print(f"  live: usage={orch.client.usage}")
        assert res["used_fallback"] is False, "live ask fell back to the template"
        before = dict(orch.client.usage)
        res2 = orch.ask("Should we inspect this engine?", uid)
        after = dict(orch.client.usage)
        assert after["cache_hits"] > before["cache_hits"], "rerun did not hit the cache"
        assert after["calls"] == before["calls"], "rerun made new API calls"
        print("  live rerun: from cache (no new calls)")

    if failures:
        print("upload check FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("upload check OK")


if __name__ == "__main__":
    main()
