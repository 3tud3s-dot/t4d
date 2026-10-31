"""CPU-only, episode-wise dataset construction. Default invocation is a plan.

--execute is the only mode that queries the network, downloads, converts or
cleans files. Execute on Linux through Slurm; never on the login node. Source
files are cleaned only after full V1 validation. Failed and ambiguous inputs
are retained with an explicit status, never silently classified as bad data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.error

try:
    from .catalog import Hub, REPOS, OFFICIAL_ENDPOINT, normalize_endpoint, build_catalog, safe_name, save_json, parse_scene_ids
    from .download import Downloader, SpaceBlocked, bounded_path, check_space, available_bytes
except ImportError:
    from catalog import Hub, REPOS, OFFICIAL_ENDPOINT, normalize_endpoint, build_catalog, safe_name, save_json, parse_scene_ids
    from download import Downloader, SpaceBlocked, bounded_path, check_space, available_bytes

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCHEMA = "t4d-dataset-build-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity():
    return {p.name: digest(p) for p in sorted(HERE.iterdir()) if p.suffix in (".py", ".sbatch")}


def load_config(path, project=PROJECT):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema", "source_root", "output_root", "runs_root", "scenes", "download", "space",
                "cleanup_verified_sources", "scene_rules"}
    keys = set(config)
    if "scene_ids" in config:
        expected = expected - {"scenes"} | {"scene_ids"}
    if keys not in (expected, expected | {"episodes"}) or config["schema"] != SCHEMA:
        raise ValueError("Invalid build configuration schema")
    roots = {k: bounded_path(project, config[k]) for k in ("source_root", "output_root", "runs_root")}
    for key, root in roots.items():
        for other_key, other in roots.items():
            if key != other_key and (root.is_relative_to(other) or other.is_relative_to(root)):
                raise ValueError("Source, output and run directories must be separate")
    if "scene_ids" in config:
        parse_scene_ids(config["scene_ids"])
    scenes = config.get("scenes", [])
    if not isinstance(scenes, list) or len(scenes) != len(set(scenes)):
        raise ValueError("Scene list must be unique")
    for scene in scenes:
        safe_name(scene)
    selections = config.get("episodes", {})
    if not isinstance(selections, dict):
        raise ValueError("Episode selections must be a dictionary")
    for scene, episodes in selections.items():
        safe_name(scene)
        if scenes and scene not in scenes:
            raise ValueError("Episode selection references an unselected scene")
        if "scene_ids" in config and (not re.match(r"^\d{3}-", scene)
                or int(scene[:3]) not in parse_scene_ids(config["scene_ids"])):
            raise ValueError("Episode selection references an unselected scene ID")
        if (not isinstance(episodes, list) or not episodes or any(type(n) is not int or n < 0 for n in episodes)
                or len(episodes) != len(set(episodes))):
            raise ValueError("Select distinct nonnegative Episode IDs")
    d, s = config["download"], config["space"]
    if set(d) not in ({"proxy", "attempts"}, {"endpoint", "proxy", "attempts"}) or type(d["attempts"]) is not int or not 1 <= d["attempts"] <= 10:
        raise ValueError("Invalid bounded download retry configuration")
    normalize_endpoint(d.get("endpoint", OFFICIAL_ENDPOINT))
    if d["proxy"] is not None and not isinstance(d["proxy"], str):
        raise ValueError("Invalid proxy")
    if isinstance(d["proxy"], str) and "@" in d["proxy"]:
        raise ValueError("Do not store proxy credentials in configuration")
    if set(s) != {"quota_limit_gib", "reserve_gib", "output_to_splat_tar_multiplier"}:
        raise ValueError("Invalid space configuration")
    for name, value in s.items():
        import math
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Space settings must be finite and positive")
    if s["output_to_splat_tar_multiplier"] < 1:
        raise ValueError("Do not assume compressed output savings for quota admission")
    if type(config["cleanup_verified_sources"]) is not bool:
        raise ValueError("Cleanup flag must be boolean")
    return config, roots


def select_episodes(catalog, selections):
    """Explicit smoke subsets are independent of dataset completeness filtering."""
    import copy
    result = copy.deepcopy(catalog)
    scenes = {s["scene"]: s for s in result["scenes"]}
    for scene, episodes in selections.items():
        if scene not in scenes:
            raise ValueError("Selected scene absent from pinned catalog")
        available = {e["episode"] for e in scenes[scene]["episodes"]}
        if scenes[scene].get("audit_status") == "request_failed":
            continue  # The episode list is unknown, not proven absent.
        if set(episodes) - available:
            raise ValueError(f"Requested Episode absent from pinned catalog: {scene}")
        scenes[scene]["episodes"] = [e for e in scenes[scene]["episodes"] if e["episode"] in episodes]
    return result


def verify_source_snapshot(project=PROJECT):
    sha = os.environ.get("T4D_SOURCE_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Actual builds require T4D_SOURCE_SHA from the reviewed local commit")
    path = bounded_path(project, ".source_version.json")
    version = json.loads(path.read_text(encoding="utf-8"))
    if version.get("commit") != sha:
        raise ValueError("Server source version differs from requested commit")
    files = version.get("files", {})
    required = {"scripts/rebuild_dataset/" + p.name for p in HERE.iterdir()
                if p.suffix in (".py", ".sbatch", ".json")}
    if not required <= files.keys():
        raise ValueError("Source manifest omits executable code or configuration")
    for relative, expected in files.items():
        if digest(bounded_path(project, relative)) != expected:
            raise ValueError(f"Server source file differs from pinned commit: {relative}")
    return sha


@contextmanager
def run_lock(path):
    import fcntl
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_command(args, log):
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\nCommand: " + json.dumps(list(map(str, args))) + "\n")
        stream.flush()
        result = subprocess.run(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    if result.returncode:
        raise RuntimeError(f"Conversion/validation failed (exit {result.returncode}); see {log}")


def converter_command(scene, episode, roots, rules, *, validate=False, structure_only=False):
    command = [sys.executable, "-B", str(HERE / ("validate.py" if validate else "rebuild.py")),
               "--scene", scene, "--episodes", str(episode), "--source-root", str(roots["source_root"]),
               "--data-root" if validate else "--output-root", str(roots["output_root"]),
               "--scene-rules", str(rules)]
    if structure_only:
        if not validate:
            raise ValueError("Structure-only is only a validation option")
        command.append("--structure-only")
    return command


def prepare_episode_rules(scene, episode, roots, run, configured_path, shared):
    try:
        from . import rebuild
    except ImportError:
        import rebuild
    configured = json.loads(Path(configured_path).read_text(encoding="utf-8"))
    medians = {stream: [Path(r["path"]).name for r in shared
                       if r["path"].startswith(f"raw/{scene}/brics-odroid_{stream}/")
                       and re.fullmatch(r"median_\d+\.npy", Path(r["path"]).name)]
               for stream in rebuild.STREAMS}
    effective, audit = rebuild.resolve_episode_rules(roots["source_root"], scene, episode, configured,
                                                    available_medians=medians)
    path = bounded_path(run, f"checks/{safe_name(scene)}-episode_{episode}-rules.json")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != effective:
            raise ValueError("Previously resolved Episode rules changed; preserving existing evidence")
    else:
        save_json(path, effective)
    return path, audit


def process_episode(scene, entry, shared, downloader, roots, run, config, state, rules, supported, approved_medians=None):
    try:
        from . import rebuild
    except ImportError:
        import rebuild
    key = f"{scene}/episode_{entry['episode']}"
    item = state["episodes"].setdefault(key, {"status": "pending"})
    log = run / "logs" / f"{scene}-episode_{entry['episode']}.log"
    target = bounded_path(roots["output_root"], key)
    records = entry["files"]
    if item["status"] in ("rejected_local", "rejected_cleaned"):
        if item["status"] == "rejected_local" and config["cleanup_verified_sources"]:
            downloader.discard(records, reason=item["reason"])
            item["status"] = "rejected_cleaned"
            save_json(run / "state.json", state)
        return
    if item["status"] in ("verified", "cleaned"):
        if "rules_file" in item:
            rules = bounded_path(run, item["rules_file"])
            if digest(rules) != item["rules_sha256"]:
                raise ValueError("Resolved Episode rules changed after validation")
        if digest(target / "metadata.json") != item["metadata_sha256"]:
            raise ValueError("Published V1 metadata changed after successful validation")
        run_command(converter_command(scene, entry["episode"], roots, rules, validate=True, structure_only=True), log)
        if config["cleanup_verified_sources"]:
            downloader.cleanup(records, validation_passed=True)
            item["status"] = "cleaned"
            save_json(run / "state.json", state)
        return
    if entry.get("audit_status") == "request_failed":
        item.update(status="network_audit_failed", reason=entry.get("request_errors", []))
        save_json(run / "state.json", state)
        return
    if not entry["download_ready"]:
        item.update(status="unavailable_sources", reason=entry["reasons"])
        save_json(run / "state.json", state)
        return
    space = config["space"]
    splat_bytes = sum(f["size"] for f in records if f["path"].endswith("/splatfacto.tar"))
    small = [r for r in records if r["path"].endswith(("/metadata.json", "/robot/robot.npy"))]
    # Ambiguous identity need not retain huge GS/camera archives: inspect the
    # pinned metadata/robot payloads first, without classifying the scene as bad.
    check_space(available_bytes(roots["source_root"], space["quota_limit_gib"]),
                downloader.missing_bytes(small), 0, int(space["reserve_gib"] * 2**30))
    item.update(status="checking_inputs", reason=None)
    save_json(run / "state.json", state)
    for record in small:
        downloader.fetch(record)
    try:
        rules, local_audit = prepare_episode_rules(scene, entry["episode"], roots, run, rules, shared)
    except rebuild.RulesPending as exc:
        item.update(status="needs_rules", reason=str(exc))
        save_json(run / "state.json", state)
        return
    except rebuild.DataRejected as exc:
        item.update(status="rejected_local", reason=str(exc))
        save_json(run / "state.json", state)  # Rejection evidence precedes deletion.
        if config["cleanup_verified_sources"]:
            downloader.discard(records, reason=str(exc))
            item["status"] = "rejected_cleaned"
            save_json(run / "state.json", state)
        return
    item.update(rules_file=rules.relative_to(run).as_posix(), rules_sha256=digest(rules), local_audit=local_audit)
    check_space(available_bytes(roots["source_root"], space["quota_limit_gib"]),
                downloader.missing_bytes(shared + records),
                int(splat_bytes * space["output_to_splat_tar_multiplier"]), int(space["reserve_gib"] * 2**30))
    item["status"] = "downloading"
    save_json(run / "state.json", state)
    small_paths = {r["path"] for r in small}
    for record in shared + records:
        if record["path"] not in small_paths:
            downloader.fetch(record)
    item["status"] = "downloaded"
    save_json(run / "state.json", state)
    # An existing uncommitted/partial output is never overwritten automatically.
    if target.exists():
        item.update(status="validating_existing")
    else:
        item.update(status="converting")
    save_json(run / "state.json", state)
    run_command(converter_command(scene, entry["episode"], roots, rules), log)
    run_command(converter_command(scene, entry["episode"], roots, rules, validate=True), log)
    item.update(status="verified", metadata_sha256=digest(target / "metadata.json"), full_source_validation=True)
    save_json(run / "state.json", state)
    if config["cleanup_verified_sources"]:
        downloader.cleanup(records, validation_passed=True)
        item["status"] = "cleaned"
        save_json(run / "state.json", state)


def execute(config, roots, run_id):
    if os.name != "posix" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Actual builds require a Linux Slurm CPU job")
    verify_source_snapshot()
    for command in ("aria2c", "quota"):
        if not shutil.which(command):
            raise RuntimeError(f"Missing existing tool: {command}")
    for module in ("numpy", "plyfile"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f"Missing existing dependency: {module}")
    run = bounded_path(roots["runs_root"], run_id)
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "processed"):
        bounded_path(roots["source_root"], name).mkdir(exist_ok=True)
    run.mkdir(exist_ok=True)
    with run_lock(run / "run.lock"):
        with run_lock(roots["source_root"] / ".build.lock"):
            return execute_locked(config, roots, run)


def execute_locked(config, roots, run):
    identity = code_identity()
    protocol = {"schema": SCHEMA, "config": config, "code_sha256": identity}
    snapshot = run / "config.json"
    if snapshot.exists():
        if json.loads(snapshot.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Resume rejected: source code or configuration changed; preserve old run")
    else:
        save_json(snapshot, protocol)
    # Bind this source staging area to one run. No automatic takeover after failure.
    marker = roots["source_root"] / ".build_owner.json"
    owner = {"run": str(run.resolve()), "schema": SCHEMA}
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != owner:
        raise ValueError("Source staging area belongs to another run")
    save_json(marker, owner)
    for folder in ("logs", "checks", "catalog_cache"):
        (run / folder).mkdir(exist_ok=True)
    rules = run / "scene_rules.json"
    if rules.exists():
        if json.loads(rules.read_text(encoding="utf-8")) != config["scene_rules"]:
            raise ValueError("Configured scene-rule snapshot changed")
    else:
        save_json(rules, config["scene_rules"])
    # These are precisely the same mapping tables used by both child CLIs.
    try:
        from . import rebuild
    except ImportError:
        import rebuild
    rebuild.load_scene_rules(rules)
    hub = Hub(endpoint=config["download"].get("endpoint", OFFICIAL_ENDPOINT), proxy=config["download"]["proxy"])
    print("Network policy:", json.dumps(hub.network_info), flush=True)
    pins = run / "revisions.json"
    if pins.exists():
        revisions = json.loads(pins.read_text(encoding="utf-8"))
    else:
        try:
            revisions = {k: hub.pin(repo) for k, repo in REPOS.items()}
        except (OSError, urllib.error.URLError) as exc:
            previous = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else {"episodes": {}}
            previous.update(status="network_audit_failed", reason=type(exc).__name__ + ": " + str(exc))
            save_json(run / "state.json", previous)
            raise
        save_json(pins, revisions)
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
    manifest = bounded_path(run, provenance.get("manifest_file", "manifest.json")) if provenance else run / "manifest.json"
    if provenance and (not manifest.exists() or provenance["manifest_sha256"] != digest(manifest)):
        raise ValueError("Pinned manifest changed")
    if manifest.exists():
        catalog = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        catalog = None
    refresh = catalog is not None and any(s.get("audit_status") == "request_failed" for s in catalog["scenes"])
    if catalog is None or refresh:
        if refresh:
            # Immutable generations plus one atomic provenance pointer. A crash
            # never replaces a published successful manifest with half a write.
            number = 2
            while (run / f"manifest-{number:04d}.json").exists():
                number += 1
            manifest = run / f"manifest-{number:04d}.json"
        try:
            kwargs = {"scene_ids": config["scene_ids"]} if "scene_ids" in config else {}
            catalog = select_episodes(build_catalog(hub, revisions, config.get("scenes", []),
                                                    run / "catalog_cache", **kwargs), config.get("episodes", {}))
        except (OSError, urllib.error.URLError) as exc:
            previous = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else {"episodes": {}}
            previous.update(status="network_audit_failed", reason=type(exc).__name__ + ": " + str(exc))
            save_json(run / "state.json", previous)
            raise
        save_json(manifest, catalog)
    if catalog["revisions"] != revisions:
        raise ValueError("Manifest revision mismatch")
    if provenance is None:
        provenance = {"code_sha256": identity, "manifest_sha256": digest(manifest),
                  "manifest_file": manifest.name,
                  "source_commit": os.environ.get("T4D_SOURCE_SHA"),
                  "network": hub.network_info,
                  "config_sha256": digest(snapshot), "python": sys.version,
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                  "started_utc": datetime.now(timezone.utc).isoformat()}
    elif refresh:
        provenance.setdefault("audit_history", []).append({"manifest_file": provenance.get("manifest_file", "manifest.json"),
                                                          "sha256": provenance["manifest_sha256"]})
        provenance.update(manifest_file=manifest.name, manifest_sha256=digest(manifest))
    save_json(provenance_path, provenance)
    save_json(run / "network_audit.json", {
        "revisions": revisions, "selection_errors": catalog.get("selection_errors", []),
        "scenes": [{k: s.get(k) for k in ("scene", "audit_status", "request_errors", "reasons", "streams")}
                   | {"episodes": [{k: e.get(k) for k in ("episode", "audit_status", "reasons", "request_errors")}
                                    for e in s["episodes"]]} for s in catalog["scenes"]]})
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"episodes": {}, "attempts": []}
    state["attempts"].append({"job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat()})
    state["status"] = "running"
    save_json(state_path, state)
    downloader = Downloader(roots["source_root"], run / "source_ledger.json", run / "logs" / "download.log", config["download"])
    blocked = False
    state.setdefault("scene_status", {})
    state.setdefault("completed_scenes", [])
    for scene in catalog["scenes"]:
        if scene.get("audit_status") == "request_failed":
            state["scene_status"][scene["scene"]] = {"status": "network_audit_failed", "reason": scene.get("request_errors", [])}
            save_json(state_path, state)
            blocked = True
            continue
        for entry in scene["episodes"]:
            key = f"{scene['scene']}/episode_{entry['episode']}"
            if code_identity() != identity:
                raise RuntimeError("Running source changed; stopping before next Episode")
            try:
                process_episode(scene["scene"], entry, scene["raw_files"], downloader, roots, run,
                                config, state, rules, rebuild.SCENE_BINDINGS, rebuild.SCENE_MEDIANS)
            except SpaceBlocked as exc:
                state["status"] = "space_blocked"
                state["reason"] = str(exc)
                save_json(state_path, state)
                return 2
            except Exception as exc:
                state["episodes"].setdefault(key, {}).update(status="failed_retained", reason=f"{type(exc).__name__}: {exc}")
                save_json(run / "checks" / f"{scene['scene']}-episode_{entry['episode']}.json", state["episodes"][key])
                save_json(state_path, state)
                blocked = True
            status = state["episodes"][key]["status"]
            print(f"{key}: {status}", flush=True)
            blocked |= status not in ("verified", "cleaned", "unavailable_sources", "rejected_local", "rejected_cleaned")
        keys = [f"{scene['scene']}/episode_{e['episode']}" for e in scene["episodes"]]
        statuses = [state["episodes"].get(k, {}).get("status") for k in keys]
        successful = bool(keys) and all(s in ("verified", "cleaned") for s in statuses)
        terminal = bool(keys) and all(s in ("verified", "cleaned", "unavailable_sources", "rejected_local", "rejected_cleaned") for s in statuses)
        if terminal:
            if config["cleanup_verified_sources"]:
                if any(s in ("verified", "cleaned") for s in statuses):
                    downloader.cleanup(scene["raw_files"], validation_passed=True)
                else:
                    downloader.discard(scene["raw_files"], reason="All selected Episodes were unavailable or explicitly rejected")
        state["scene_status"][scene["scene"]] = {"status": "completed" if successful else "needs_attention" if not terminal and keys else "completed_with_exclusions",
                                                "episodes": len(keys), "validated": sum(s in ("verified", "cleaned") for s in statuses)}
        if successful and scene["scene"] not in state["completed_scenes"]:
            state["completed_scenes"].append(scene["scene"])
            print(f"SCENE_COMPLETE {scene['scene']} completed_scenes={len(state['completed_scenes'])}", flush=True)
        save_json(state_path, state)
    from collections import Counter
    state["summary"] = dict(Counter(v["status"] for v in state["episodes"].values()))
    state["scenes_without_processed_episodes"] = [s["scene"] for s in catalog["scenes"] if not s["episodes"]]
    excluded = bool(state["summary"].get("unavailable_sources") or state["summary"].get("rejected_local")
                    or state["summary"].get("rejected_cleaned") or state["scenes_without_processed_episodes"]
                    or catalog.get("selection_errors"))
    state["status"] = "needs_attention" if blocked else ("completed_with_exclusions" if excluded else "completed")
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(state_path, state)
    return 2 if blocked else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE / "build_config.json")
    p.add_argument("--run-id", required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    try:
        safe_name(args.run_id)
        config, roots = load_config(args.config)
        if not args.execute:
            print(json.dumps({"mode": "plan_only", "run": str(roots["runs_root"] / args.run_id),
                              "paths": {k: str(v) for k, v in roots.items()}, "scenes": config.get("scenes", []),
                              "scene_ids": parse_scene_ids(config["scene_ids"]) if "scene_ids" in config else None,
                              "explicit_episode_selection": config.get("episodes", {}),
                              "endpoint": config["download"].get("endpoint", OFFICIAL_ENDPOINT),
                              "proxy_mode": "direct" if config["download"]["proxy"] is None else "explicit_proxy",
                              "automatic_proxy_fallback": False,
                              "episode_count_filter": False, "multiple_median_filter": False,
                              "network_queries": False, "downloads": False, "conversion": False, "cleanup": False}, indent=2))
            return 0
        return execute(config, roots, args.run_id)
    except Exception as exc:
        print(f"Build stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
