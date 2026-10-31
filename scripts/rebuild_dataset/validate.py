#!/usr/bin/env python3
"""Read-only validator for a reconstructed, audited Deform360 scene.

Dependencies: Python >= 3.10, numpy; plyfile for full core source verification,
h5py for --depth-only (which never reads core tactile/controller/GS arrays).
Examples (not executed automatically):
  python scripts/rebuild_dataset/validate.py --scene 001-rope
  python scripts/rebuild_dataset/validate.py --scene 008-pink-cloth
  python scripts/rebuild_dataset/validate.py --scene 051-cube
  python scripts/rebuild_dataset/validate.py --scene 008-pink-cloth --episodes 0 --structure-only
  python scripts/rebuild_dataset/validate.py --scene 001-rope --depth-only

--depth-only checks the separate touch4d-depth-v1 sibling of --data-root;
override it with --depth-root. It binds to core metadata/timestamps without
recomputing or hashing tactile, controller or GS files.

By default verify every output hash, schema, timeline, mask, controller geometry,
and every source hash; recompute tactile and GS fields from original files.
--structure-only skips original-file access, but still checks output hashes,
causal alignment from retained timestamps, controller reconstruction and GS data.
Neither mode writes files, modifies the dataset, downloads nor extracts archives.
Use --source-root to relocate original data without editing stored metadata.
Default full-scene validation includes source episode IDs, so a missing rebuilt
episode fails. Structure-only mode checks all discovered output episodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

sys.dont_write_bytecode = True
if __package__:
    from . import rebuild as _rebuild
    from .rebuild import (GS_FIELDS, HAND_ORDER, POINT_IDS, PROJECT, SCENE_BINDINGS, SCENE_MEDIANS,
                          SUPPORTED_SCRIPT_VERSIONS, STREAMS, VERSION, SourceReader,
                          binding_evidence, discover_episodes, episode_layout,
                          find_raw_sources, observation_timeline,
                          controller_arrays, gs_arrays, hand_slots, hash_file,
                          load_actions, check_actions, load_median, load_raw, load_robot, parameters, parse_episodes,
                          require, safe_relative, select_raw, tactile_arrays, timestamp_array)
else:
    import rebuild as _rebuild
    from rebuild import (GS_FIELDS, HAND_ORDER, POINT_IDS, PROJECT, SCENE_BINDINGS, SCENE_MEDIANS,
                         SUPPORTED_SCRIPT_VERSIONS, STREAMS, VERSION, SourceReader,
                         binding_evidence, discover_episodes, episode_layout,
                         find_raw_sources, observation_timeline,
                         controller_arrays, gs_arrays, hand_slots, hash_file,
                         load_actions, check_actions, load_median, load_raw, load_robot, parameters, parse_episodes,
                         require, safe_relative, select_raw, tactile_arrays, timestamp_array)


def read_npz(path, required_keys):
    require(path.is_file() and not path.is_symlink(), f"Missing or linked NPZ: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), f"Duplicate NPZ members: {path}")
        require(set(archive.files) == set(required_keys), f"Wrong NPZ fields: {path}: {archive.files}")
        arrays = {key: archive[key] for key in archive.files}
    require(all(array.dtype.kind in "biuf" for array in arrays.values()), f"Non-numeric NPZ: {path}")
    return arrays


def field(array, shape, dtype, name, *, finite=False):
    require(array.shape == shape, f"{name}: expected shape {shape}, got {array.shape}")
    require(array.dtype == np.dtype(dtype), f"{name}: expected dtype {dtype}, got {array.dtype}")
    if finite:
        require(np.isfinite(array).all(), f"Non-finite {name}")


def equal(actual, expected, name):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype,
            f"{name}: recomputed shape/dtype mismatch")
    require(np.array_equal(actual, expected, equal_nan=True), f"{name}: values differ from source/recomputation")


def check_source_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "member", "bytes", "sha256"},
            "Malformed source reference")
    safe_relative(ref["path"])
    if ref["member"] is not None:
        safe_relative(ref["member"])
    require(isinstance(ref["bytes"], int) and ref["bytes"] >= 0, "Invalid source byte count")
    require(isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
            "Invalid source SHA-256")


def check_sources_schema(refs, scene, episode, frames, prefix, cameras, camera):
    expected = {"observation", "robot", "episode_metadata"}
    expected.update(f"tactile_{c}_{kind}" for c in range(4) for kind in ("raw", "timestamps", "median"))
    expected.update(f"gs_{f}" for f in range(frames))
    expected.update(f"camera_timestamps_{name}" for name in cameras)
    require(set(refs) == expected, "Source manifest is incomplete or has unexpected fields")
    for ref in refs.values():
        check_source_ref(ref)
    safe_relative(prefix)
    match = re.fullmatch(re.escape(f"processed/{scene}/episode_") + r"(\d+)", prefix)
    require(match is not None and int(match[1]) == episode, "Wrong source episode directory")
    for key, relative in [("robot", f"{prefix}/robot/robot.npy"),
                          ("episode_metadata", f"{prefix}/metadata.json")]:
        require((refs[key]["path"], refs[key]["member"]) == (relative, None), f"Unexpected source for {key}")
    observation = f"{camera}/aligned_timestamps.txt"
    require((refs["observation"]["path"], refs["observation"]["member"]) in (
        (f"{prefix}/{observation}", None), (f"{prefix}/cameras.tar", observation)),
        "Observation timeline must come from the agreed aligned camera timestamps")
    require(refs["observation"] == refs[f"camera_timestamps_{camera}"], "Primary camera provenance mismatch")
    for name in cameras:
        relative = f"{name}/aligned_timestamps.txt"
        ref = refs[f"camera_timestamps_{name}"]
        require((ref["path"], ref["member"]) in ((f"{prefix}/{relative}", None),
                (f"{prefix}/cameras.tar", relative)), "Wrong camera timestamp source")
    for channel in range(4):
        sensor = "brics-odroid_" + STREAMS[channel]
        parent = f"raw/{scene}/{sensor}"
        for kind in ("raw", "timestamps", "median"):
            ref = refs[f"tactile_{channel}_{kind}"]
            path = Path(ref["path"])
            require(ref["member"] is None and path.parent.as_posix() == parent,
                    "Tactile source has wrong scene/sensor directory")
            pattern = r"median_\d+\.npy" if kind == "median" else (
                re.escape(sensor) + (r"_\d+\.npy" if kind == "raw" else r"_\d+\.txt"))
            require(re.fullmatch(pattern, path.name) is not None, "Unexpected tactile source name")
            if kind == "median" and scene in SCENE_MEDIANS:
                require(path.name == SCENE_MEDIANS[scene][STREAMS[channel]],
                        f"Wrong reviewed baseline for {scene}/{sensor}")
        require(Path(refs[f"tactile_{channel}_raw"]["path"]).stem ==
                Path(refs[f"tactile_{channel}_timestamps"]["path"]).stem, "Raw/timestamp stems differ")
    for f in range(frames):
        ref = refs[f"gs_{f}"]
        # Match numeric frame IDs; original PLY names may contain leading zeros.
        name = ref["member"] or ref["path"]
        match = re.fullmatch(r"splat_(\d+)\.ply", Path(name).name)
        require(match is not None and int(match[1]) == f, f"GS source frame mismatch: {f}")
        if ref["member"] is None:
            require(str(Path(ref["path"]).parent).replace("\\", "/") == f"{prefix}/splatfacto", "Wrong GS source directory")
        else:
            require(ref["path"] == f"{prefix}/splatfacto.tar"
                    and str(Path(ref["member"]).parent).replace("\\", "/") == "splatfacto", "Wrong GS archive member")


def checked_payload(reader, refs, key):
    ref = refs[key]
    payload = reader.read(key, ref["path"], ref["member"])
    require(reader.refs[key] == ref, f"Source hash/size differs: {key}")
    return payload


def validate_episode(directory, *, expected_scene=None, source_root=None, verify_sources=True):
    directory = Path(directory).resolve()
    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    require(metadata["schema_version"] == VERSION and metadata["script_version"] in SUPPORTED_SCRIPT_VERSIONS,
            "Unsupported schema/script version")
    scene = metadata["scene"]
    require(scene in SCENE_BINDINGS, "Scene has no reviewed bindings")
    require(expected_scene is None or scene == expected_scene, "Requested scene differs from metadata")
    require(directory.parent.name == scene, "Output directory scene differs from metadata")
    episode = metadata["episode"]
    require(type(episode) is int and episode >= 0, "Invalid episode ID")
    require(directory.name in (f"episode_{episode}", f".episode_{episode}.incomplete"),
            "Directory name disagrees with metadata episode ID")
    frames = metadata["frames"]
    require(type(frames) is int and frames > 0, "Invalid frame count")
    slots = hand_slots(scene, episode)
    require(metadata["binding_basis"] == binding_evidence(scene, episode), "Binding provenance mismatch")
    require(metadata["stream_order"] == list(STREAMS) and metadata["hand_order"] == list(HAND_ORDER),
            "Channel/hand order mismatch")
    require(metadata["robot_slot_hands"] == slots, "Wrong episode hand-slot mapping")
    require(metadata["physical_hand_to_robot_slot"] == {h: slots.index(h) if h in slots else None for h in HAND_ORDER},
            "Incorrect hand-to-slot metadata")
    require(metadata["controller_point_ids"] == POINT_IDS, "Incorrect controller point ordering")
    require(metadata["parameters"] == parameters(), "Processing parameters differ from this protocol version")
    camera, cameras = metadata["observation_camera"], metadata["observation_cameras"]
    require(isinstance(cameras, list) and cameras and all(isinstance(c, str) and
            re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in cameras), "Invalid observation cameras")
    require(cameras == sorted(set(cameras)) and camera == cameras[0], "Invalid primary camera/list order")
    prefix = metadata["source_episode_directory"]
    for name in ("rebuild.py", "validate.py"):
        require(re.fullmatch(r"[0-9a-f]{64}", metadata["scripts_sha256"][name]), "Missing generator provenance")
    files = {"tactile.npz", "controller.npz", "timestamps.npz"}
    files.update(f"gs/frame_{f:06d}.npz" for f in range(frames))
    require(set(metadata["output_files"]) == files, "Output manifest does not cover the complete episode")
    actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    require(actual_files == files | {"metadata.json"}, "Missing or unexpected episode files")
    require(not any(p.is_symlink() for p in directory.rglob("*")), "Output episode must not contain symbolic links")
    for name in sorted(files):
        require(hash_file(directory / name) == metadata["output_files"][name], f"Output SHA-256 mismatch: {name}")
    refs = metadata["source_files"]
    check_sources_schema(refs, scene, episode, frames, prefix, cameras, camera)

    tactile = read_npz(directory / "tactile.npz", {"tactile", "valid_mask"})
    field(tactile["tactile"], (frames, 4, 12, 32), np.float32, "tactile", finite=True)
    field(tactile["valid_mask"], (frames, 4, 12, 32), bool, "valid_mask")
    require(np.all(tactile["tactile"] >= 0), "Negative reconstructed tactile response")
    require(np.all(tactile["tactile"][~tactile["valid_mask"]] == 0), "Invalid tactile must be zero-filled")
    require(not tactile["valid_mask"][..., 31].any(), "Tactile column 31 must be invalid")
    timestamp_keys = {"observation_timestamp_us", "tactile_raw_index", "tactile_age_us"}
    timestamp_keys.update(f"raw_tactile_timestamp_us_{c}" for c in range(4))
    timestamps = read_npz(directory / "timestamps.npz", timestamp_keys)
    obs = timestamps["observation_timestamp_us"]
    field(obs, (frames,), np.int64, "observation_timestamp_us")
    require(np.all(obs >= 0) and np.all(np.diff(obs) > 0), "Observation time must be strictly increasing Unix microseconds")
    for key in ("tactile_raw_index", "tactile_age_us"):
        field(timestamps[key], (frames, 4), np.int64, key)
    for channel in range(4):
        raw_ts = timestamps[f"raw_tactile_timestamp_us_{channel}"]
        require(raw_ts.ndim == 1 and raw_ts.dtype == np.int64 and np.all(raw_ts >= 0)
                and np.all(np.diff(raw_ts) >= 0), f"Invalid raw timeline: {channel}")
        index, age, time_valid = select_raw(raw_ts, obs)
        equal(timestamps["tactile_raw_index"][:, channel], index, f"raw_index channel {channel}")
        equal(timestamps["tactile_age_us"][:, channel], age, f"age channel {channel}")
        require(not tactile["valid_mask"][~time_valid, channel].any(), "Future/missing/stale tactile accepted")
        # Further pixel validity and response values require the original raw/median.

    controller_keys = {"controller_points", "controller_valid_mask", "T_worlds", "openings", "opening_clipped"}
    has_actions = metadata["script_version"] in ("1.3.0", "1.4.0")
    if has_actions:
        controller_keys.add("actions")
    controller = read_npz(directory / "controller.npz", controller_keys)
    field(controller["controller_points"], (frames, 2, 30, 3), np.float32, "controller_points", finite=True)
    field(controller["controller_valid_mask"], (frames, 2, 30), bool, "controller_valid_mask")
    field(controller["opening_clipped"], (frames, 2), bool, "opening_clipped")
    t, openings = controller["T_worlds"], controller["openings"]
    require(t.dtype.kind == "f" and t.dtype.itemsize in (4, 8) and t.shape == (frames, len(slots), 4, 4), "Bad T_worlds")
    require(openings.dtype.kind == "f" and openings.dtype.itemsize in (4, 8)
            and openings.shape == (frames, len(slots)), "Bad openings")
    expected_dtypes = {"T_worlds": str(t.dtype), "openings": str(openings.dtype)}
    if has_actions:
        actions = controller["actions"]
        require(actions.shape == (frames, len(slots), 5, 3)
                and actions.dtype.kind == "f" and actions.dtype.itemsize in (4, 8), "Bad actions")
        check_actions(actions, t, openings)
        expected_dtypes["actions"] = str(actions.dtype)
        allowed_shapes = [[frames, len(slots), 5, 3]]
        if len(slots) == 1:
            allowed_shapes.append([frames, 5, 3])
        require(metadata["source_robot_shapes"].get("actions") in allowed_shapes,
                "Original actions shape metadata mismatch")
    require(metadata["source_robot_dtypes"] == expected_dtypes,
            "Source dtype metadata mismatch")
    points, mask, clipped = controller_arrays(t, openings, slots)
    equal(controller["controller_points"], points, "controller_points recomputation")
    equal(controller["controller_valid_mask"], mask, "controller_valid_mask recomputation")
    equal(controller["opening_clipped"], clipped, "opening_clipped recomputation")
    require(np.all(controller["controller_points"][~mask] == 0), "Invalid controller points must be zero-filled")
    for hand_index, hand in enumerate(HAND_ORDER):
        if hand not in slots:
            require(not mask[:, hand_index].any() and not clipped[:, hand_index].any(),
                    f"Absent {hand} hand marked valid/clipped")

    counts = []
    reader = SourceReader(source_root or metadata["source_root"]) if verify_sources else None
    try:
        if reader is not None:
            actual_prefix, camera_refs, gs_refs = episode_layout(reader, scene, episode)
            actual_camera, actual_obs = observation_timeline(reader, camera_refs)
            require(actual_prefix == prefix and actual_camera == camera and sorted(camera_refs) == cameras,
                    "Discovered episode/cameras differ from provenance")
            equal(obs, actual_obs, "all-camera timeline agreement")
            require(sorted(gs_refs) == list(range(frames)), "Source GS frame coverage differs")
            for name in cameras:
                key = f"camera_timestamps_{name}"
                require(reader.refs[key] == refs[key], f"Camera source hash/size differs: {name}")
            discovered_raw = find_raw_sources(reader.root, scene, obs)
            for channel, paths in enumerate(discovered_raw):
                for kind, path in zip(("raw", "timestamps", "median"), paths):
                    ref = refs[f"tactile_{channel}_{kind}"]
                    require((ref["path"], ref["member"]) == (path, None), "Discovered raw/median pairing differs")
            for frame, ref in gs_refs.items():
                saved = refs[f"gs_{frame}"]
                require((saved["path"], saved["member"]) == ref, "Discovered GS source differs")
            source_obs = timestamp_array(checked_payload(reader, refs, "observation"), strict=True)
            equal(obs, source_obs, "observation timeline source")
            source_meta = json.loads(checked_payload(reader, refs, "episode_metadata").decode("utf-8"))
            require(set(cameras) == set(source_meta.get("cameras", [])), "Cameras differ from original metadata")
            robot_payload = checked_payload(reader, refs, "robot")
            source_t, source_o, shapes = load_robot(robot_payload, frames, slots)
            if has_actions:
                source_actions, shapes["actions"] = load_actions(robot_payload, frames, slots, source_t, source_o)
                equal(actions, source_actions, "original actions")
                require(actions.dtype == source_actions.dtype, "Original actions dtype mismatch")
            equal(t, source_t, "original T_worlds")
            equal(openings, source_o, "original openings")
            require(metadata["source_robot_shapes"] == shapes, "Original shape metadata mismatch")
            for channel in range(4):
                raw = load_raw(checked_payload(reader, refs, f"tactile_{channel}_raw"))
                median = load_median(checked_payload(reader, refs, f"tactile_{channel}_median"))
                raw_ts = timestamp_array(checked_payload(reader, refs, f"tactile_{channel}_timestamps"), strict=False, allow_empty=True)
                equal(timestamps[f"raw_tactile_timestamp_us_{channel}"], raw_ts, f"raw timestamp source {channel}")
                values, accepted, _, _ = tactile_arrays(raw, median, raw_ts, obs)
                equal(tactile["tactile"][:, channel], values, f"raw-minus-median clipping {channel}")
                equal(tactile["valid_mask"][:, channel], accepted, f"pixel/time validity {channel}")
        for frame in range(frames):
            gs = read_npz(directory / "gs" / f"frame_{frame:06d}.npz", set(GS_FIELDS) | {"frame_id", "timestamp_us"})
            require(gs["xyz"].ndim == 2, "GS xyz must have rank 2")
            count = len(gs["xyz"])
            require(count > 0, f"Empty GS frame {frame}")
            counts.append(count)
            for name, fields in GS_FIELDS.items():
                field(gs[name], (count, len(fields)), np.float32, f"GS {frame} {name}", finite=True)
            for name, expected in (("frame_id", frame), ("timestamp_us", obs[frame])):
                field(gs[name], (), np.int64, f"GS {name}")
                require(gs[name].item() == expected, f"GS {name} mismatch at frame {frame}")
            norm = np.linalg.norm(gs["quaternions"].astype(np.float64), axis=1)
            require(np.all(np.abs(norm - 1) <= 1e-6), f"GS quaternion not normalized: {frame}")
            if reader is not None:
                original = gs_arrays(checked_payload(reader, refs, f"gs_{frame}"), frame, obs[frame])
                for name in gs:
                    equal(gs[name], original[name], f"GS source {frame}/{name}")
    finally:
        if reader is not None:
            reader.stack.close()
    require(metadata["gs_point_counts"] == counts, "GS point counts disagree with metadata")
    expected_summary = {"tactile_valid_elements": int(tactile["valid_mask"].sum()),
                        "controller_valid_points": int(mask.sum()),
                        "opening_clipped_hand_frames": int(clipped.sum())}
    require(metadata["summary"] == expected_summary, "Summary disagrees with arrays")
    return {"scene": scene, "episode": episode, "status": "PASS", "frames": frames,
            "contains_actions": has_actions,
            "gs_count_min": min(counts), "gs_count_max": max(counts), **expected_summary,
            "source_verification": "full" if verify_sources else "skipped (structure-only)",
            "source_files_verified": len(refs) if verify_sources else 0}


def check_depth_sources_schema(metadata):
    scene, episode, cameras = metadata["scene"], metadata["episode"], metadata["cameras"]
    prefix = safe_relative(metadata["source_episode_directory"])
    match = re.fullmatch(re.escape(f"processed/{scene}/episode_") + r"(\d+)", prefix)
    require(match is not None and int(match[1]) == episode, "Wrong depth source episode directory")
    refs = metadata["source_files"]
    expected = {"episode_metadata", "extrinsics", *(f"camera_timestamps_{c}" for c in cameras),
                *(f"depth_{c}" for c in cameras)}
    require(isinstance(refs, dict) and set(refs) == expected, "Incomplete or unexpected depth source manifest")
    for ref in refs.values():
        check_source_ref(ref)
    for key, path in (("episode_metadata", f"{prefix}/metadata.json"),
                      ("extrinsics", f"raw/{scene}/calibration_refined/extrinsics.npy")):
        require((refs[key]["path"], refs[key]["member"]) == (path, None), f"Wrong depth source: {key}")
    for camera in cameras:
        for key, filename in ((f"camera_timestamps_{camera}", "aligned_timestamps.txt"),
                              (f"depth_{camera}", "rendered_depth.h5")):
            relative = f"{camera}/{filename}"
            ref = refs[key]
            require((ref["path"], ref["member"]) in
                    ((f"{prefix}/{relative}", None), (f"{prefix}/cameras.tar", relative)),
                    f"Wrong depth source path: {key}")
        require(refs[f"depth_{camera}"]["sha256"] == metadata["output_files"][f"{camera}/rendered_depth.h5"],
                f"Depth must preserve exact source H5 bytes: {camera}")


def validate_depth_episode(directory, *, core_directory, expected_scene=None, expected_episode=None,
                           source_root=None, verify_sources=True):
    """Validate only the isolated extension and its small binding to existing core data."""
    _rebuild.require_h5py()
    directory, core_directory = map(_rebuild.checked_path, (directory, core_directory))
    require(not directory.is_relative_to(core_directory) and not core_directory.is_relative_to(directory),
            "Depth and core episode directories must be disjoint")
    metadata = _rebuild.read_depth_metadata(directory, core_directory,
                    expected_scene=expected_scene, expected_episode=expected_episode)
    check_depth_sources_schema(metadata)
    scene, episode, cameras, frames = (metadata[k] for k in ("scene", "episode", "cameras", "frames"))
    for relative, digest in metadata["output_files"].items():
        path = directory / relative
        require(hash_file(path) == digest, f"Depth output SHA-256 mismatch: {relative}")
        if relative.endswith(".h5"):
            require(path.stat().st_size == metadata["source_files"][f"depth_{Path(relative).parent.name}"]["bytes"],
                    f"Depth source/output byte count mismatch: {relative}")
            _rebuild.check_depth_h5(path, frames, metadata["height"], metadata["width"])
    if verify_sources:
        source_root = _rebuild.checked_path(source_root or metadata["source_root"])
        core, obs, binding = _rebuild.read_core_depth_binding(core_directory, expected_scene=scene,
                                                             expected_episode=episode)
        require(binding == metadata["core_binding"], "Core changed during depth verification")
        require(core["source_episode_directory"] == metadata["source_episode_directory"], "Core source directory changed")
        with _rebuild.DepthSourceReader(source_root) as reader:
            plan = _rebuild.depth_source_layout(reader, core, obs, source_refs=metadata["source_files"])
            require((plan["height"], plan["width"]) == (metadata["height"], metadata["width"]),
                    "Depth dimensions differ from source metadata")
            with np.load(directory / "camera_parameters.npz", allow_pickle=False) as params:
                equal(params["intrinsics"], plan["intrinsics"], "Depth source intrinsics")
                equal(params["camera_to_world"], plan["camera_to_world"], "Depth source camera_to_world")
            for camera in cameras:
                reader.stream(f"depth_{camera}", *plan["depth_refs"][camera])
            require(reader.refs == metadata["source_files"], "Depth source hashes/paths changed")
    # A core overwrite during validation must not leave a stale extension marked valid.
    _, _, binding = _rebuild.read_core_depth_binding(core_directory, expected_scene=scene, expected_episode=episode)
    require(binding == metadata["core_binding"], "Core changed during depth validation")
    return {"scene": scene, "episode": episode, "status": "PASS", "frames": frames,
            "cameras": len(cameras), "height": metadata["height"], "width": metadata["width"],
            "depth_only": True, "source_verification": "full" if verify_sources else "skipped (structure-only)",
            "source_files_verified": len(metadata["source_files"]) if verify_sources else 0}


def validate_depth_main(args):
    core_root = _rebuild.checked_path(args.data_root)
    depth_root = _rebuild.checked_path(args.depth_root or (core_root.parent / _rebuild.DEPTH_VERSION))
    require(not depth_root.is_relative_to(core_root) and not core_root.is_relative_to(depth_root),
            "Depth root must be disjoint from core root")
    available = discover_episodes(core_root / args.scene)
    episodes = args.episodes if args.episodes is not None else list(available)
    require(set(episodes) <= set(available), "Missing core episodes for depth validation")
    failed = False
    for episode in episodes:
        target = depth_root / args.scene / f"episode_{episode}"
        try:
            require(not target.with_name(f".episode_{episode}.incomplete").exists(), "Unpublished depth partial exists")
            result = validate_depth_episode(target, core_directory=available[episode], expected_scene=args.scene,
                                           expected_episode=episode, source_root=args.source_root,
                                           verify_sources=not args.structure_only)
            backup = target.with_name(f".episode_{episode}.previous")
            if backup.exists() or backup.is_symlink():
                print(f"WARNING: published depth validated; retained backup needs inspection: {backup}",
                      file=sys.stderr, flush=True)
        except (ValueError, OSError, KeyError, TypeError, ImportError, tarfile.TarError) as exc:
            failed = True
            result = {"scene": args.scene, "episode": episode, "depth_only": True, "status": "FAIL", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, choices=sorted(SCENE_BINDINGS), help="Full audited scene name")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "datasets/touch4d-v1")
    parser.add_argument("--depth-only", action="store_true", help="Validate only the isolated depth extension")
    parser.add_argument("--depth-root", type=Path,
                        help="Depth extension root; default: data-root sibling touch4d-depth-v1")
    parser.add_argument("--source-root", type=Path, help="Override the original data root stored in metadata")
    parser.add_argument("--episodes", type=parse_episodes, metavar="0-9 or 0,5",
                        help="Optional subset; default: all scene episodes")
    parser.add_argument("--structure-only", action="store_true", help="Skip original-file reads and source-value comparisons")
    args = parser.parse_args(argv)
    require(args.depth_only or args.depth_root is None, "--depth-root requires --depth-only")
    if args.depth_only:
        return validate_depth_main(args)
    scene_dir = args.data_root.resolve() / args.scene
    available = discover_episodes(scene_dir)
    source_root = args.source_root.resolve() if args.source_root is not None else None
    episodes = args.episodes if args.episodes is not None else list(available)
    if args.episodes is None:
        # Never silently ignore retained, unpublished partial episodes.
        partial_ids = {int(m[1]) for p in scene_dir.iterdir()
                       if (m := re.fullmatch(r"\.episode_(\d+)\.(?:incomplete|previous)", p.name))}
        episodes = sorted(set(episodes) | partial_ids)
        if not args.structure_only:
            if source_root is None:
                roots = set()
                for path in available.values():
                    with (path / "metadata.json").open(encoding="utf-8") as stream:
                        roots.add(Path(json.load(stream)["source_root"]).resolve())
                require(len(roots) == 1, "Mixed source roots; specify --source-root explicitly")
                source_root = roots.pop()
            source_episodes = discover_episodes(source_root / "processed" / args.scene)
            episodes = sorted(set(episodes) | set(source_episodes))
    failed = False
    for episode in episodes:
        path = available.get(episode, scene_dir / f"episode_{episode}")
        try:
            require(not (scene_dir / f".episode_{episode}.incomplete").exists(), "Unpublished partial output exists")
            result = validate_episode(path, expected_scene=args.scene, source_root=source_root,
                                      verify_sources=not args.structure_only)
            backup = scene_dir / f".episode_{episode}.previous"
            if backup.exists() or backup.is_symlink():
                print(f"WARNING: validated published output; retained backup needs inspection: {backup}",
                      file=sys.stderr, flush=True)
        except (ValueError, OSError, KeyError, TypeError, ImportError, tarfile.TarError) as exc:
            failed = True
            result = {"scene": args.scene, "episode": episode, "status": "FAIL", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return int(failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, ImportError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
