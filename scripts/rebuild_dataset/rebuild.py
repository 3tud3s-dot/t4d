#!/usr/bin/env python3
"""Reconstruct an audited Deform360 scene according to DATASET_PROTOCOL.md.

Dependencies: Python >= 3.10, numpy, plyfile; h5py for depth-only mode.
Run reconstruction on the server.
Example (not executed automatically):
  python scripts/rebuild_dataset/rebuild.py --scene 001-rope
  python scripts/rebuild_dataset/rebuild.py --scene 008-pink-cloth
  python scripts/rebuild_dataset/rebuild.py --scene 051-cube
  python scripts/rebuild_dataset/rebuild.py --scene 001-rope --depth-only

Defaults resolve relative to this project's root, not the current directory:
  --source-root datasets/deform360 (contains raw/ and processed/)
  --output-root datasets/touch4d-v1

Original files are read-only. Core reconstruction reads archive members in memory;
no whole-archive extraction or download is performed.
--depth-only streams only rendered_depth.h5 members into a separate sibling
touch4d-depth-v1 root, preserving their bytes; it never rebuilds core arrays.
Use --depth-root /path/to/depth-extension to override that root. Depth is the
published camera-Z expected-depth convention, not an independently measured sensor modality.
Existing episodes are skipped only after full source validation. --overwrite
rebuilds selected episodes, validates replacements fully, then replaces old outputs.
An interrupted/failed episode stays at .episode_N.incomplete for inspection;
only a validated episode is renamed to episode_N. No training features added.
Published robot.npy and extrinsics.npy use pickled dictionaries; source-root must be trusted.
All output NPZ files contain ordinary numeric/bool arrays, never pickle objects.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
VERSION = "touch4d-v1"
SCRIPT_VERSION = "1.4.0"
DEPTH_VERSION = "touch4d-depth-v1"
DEPTH_SCRIPT_VERSION = "1.0.0"
SUPPORTED_SCRIPT_VERSIONS = ("1.1.0", "1.2.0", "1.3.0", SCRIPT_VERSION)
# Reviewed physical identities, not an inference from episode number or R.
# New scenes/episodes require a reviewed entry before reconstruction.
SCENE_BINDINGS = {
    "001-rope": {
        0: ("right",), 1: ("right",), 2: ("right",), 3: ("right",), 4: ("right",),
        5: ("right", "left"), 6: ("right", "left"), 7: ("right", "left"),
        8: ("right", "left"), 9: ("right", "left"),
    },
    "008-pink-cloth": {
        0: ("right",), 1: ("right",), 2: ("right",), 3: ("right",), 4: ("right",),
        5: ("right", "left"), 6: ("right", "left"), 7: ("right", "left"),
        8: ("right", "left"), 9: ("right", "left"),
    },
    "051-cube": {
        0: ("right",), 1: ("right",), 2: ("right",), 3: ("right",), 4: ("right",),
        5: ("right", "left"), 6: ("right", "left"), 7: ("right", "left"),
        8: ("right", "left"), 9: ("right", "left"),
    },
}
CUBE_REVIEW_FRAMES = {
    0: [86, 127, 168], 1: [72, 108, 145], 2: [82, 122, 163],
    3: [93, 131, 169], 4: [98, 126, 154], 5: [61, 123, 185],
    6: [149, 194, 240], 7: [96, 120, 145], 8: [54, 109, 164],
    9: [58, 116, 174],
}
# Explicit audited choices, not a general earliest/latest-median heuristic.
# These four baselines reproduce all 40 published synced_tactile arrays for
# 051-cube under the release processing. Some thresholded arrays are ambiguous.
# Reconstruction still uses continuous positive values and causal alignment.
SCENE_MEDIANS = {
    "051-cube": {
        "tactilel_left": "median_1769464491882420.npy",
        "tactilel_right": "median_1769464492448159.npy",
        "tactiler_left": "median_1769464491240896.npy",
        "tactiler_right": "median_1769464491904483.npy",
    },
}
PINK_REVIEW_FRAMES = {
    0: [113, 163, 213], 1: [113, 238, 364], 2: [103, 185, 268],
    3: [97, 264, 432], 4: [101, 243, 385], 5: [88, 212, 336],
    6: [133, 241, 350], 7: [93, 210, 327], 8: [107, 236, 366],
    9: [130, 207, 284],
}
STREAMS = ("tactilel_left", "tactilel_right", "tactiler_left", "tactiler_right")
HAND_ORDER = ("left", "right")
MAX_AGE_US = 75000  # Provisional, as specified in the protocol.
RIGID_TOL = 1e-6
OPENING_LIMITS = (0.04, 0.112)
GRID_ROWS = (0, 6, 11)
GRID_COLS = (0, 8, 16, 23, 31)
POINT_IDS = [f"{side}_r{r:02d}_c{c:02d}"
             for side in HAND_ORDER for r in GRID_ROWS for c in GRID_COLS]
GS_FIELDS = {
    "xyz": ("x", "y", "z"),
    "log_scales": ("scale_0", "scale_1", "scale_2"),
    "quaternions": ("rot_0", "rot_1", "rot_2", "rot_3"),
    "opacity_logits": ("opacity",),
    "sh_dc": tuple(f"f_dc_{i}" for i in range(3)),
    "sh_rest": tuple(f"f_rest_{i}" for i in range(45)),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_episodes(value):
    try:
        items = []
        for part in value.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                require(start <= end, "Reversed episode range")
                items.extend(range(start, end + 1))
            else:
                items.append(int(part))
        require(items and all(i >= 0 for i in items), "Episode IDs must be nonnegative")
        require(len(set(items)) == len(items), "Duplicate episode selection")
        return sorted(items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def hand_slots(scene, episode):
    require(scene in SCENE_BINDINGS, f"Scene has no reviewed hand bindings: {scene}")
    require(episode in SCENE_BINDINGS[scene], f"Unreviewed hand binding: {scene}/episode_{episode}")
    return list(SCENE_BINDINGS[scene][episode])


def binding_evidence(scene, episode):
    hand_slots(scene, episode)
    if scene == "051-cube":
        return {"camera": "brics-odroid-001_cam0",
                "method": "episode-specific RGB and projected robot geometry review; no inferred identity",
                "review_frames": CUBE_REVIEW_FRAMES[episode],
                "review_record": "051-cube representative-frame review"}
    return {"camera": "brics-odroid-001_cam0",
            "method": "episode-specific RGB and projected robot geometry review; no inferred identity",
            "review_frames": PINK_REVIEW_FRAMES[episode] if scene == "008-pink-cloth" else [],
            "review_record": "008 representative-frame review" if scene == "008-pink-cloth"
                             else "001-rope previously confirmed episode bindings"}


def discover_episodes(directory):
    directory = Path(directory)
    require(directory.is_dir(), f"Scene directory missing: {directory}")
    found = {}
    for path in directory.iterdir():
        match = re.fullmatch(r"episode_(\d+)", path.name)
        if match and path.is_dir():
            episode = int(match[1])
            require(episode not in found, f"Duplicate padded/unpadded episode ID: {episode}")
            found[episode] = path
    require(found, f"No episode directories in {directory}")
    return dict(sorted(found.items()))


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and ".." not in p.parts
            and "\\" not in name and ":" not in name, f"Unsafe relative path: {name}")
    return p.as_posix()


class SourceReader:
    """Read loose files or exact regular members; never extract an archive."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.stack = ExitStack()
        self.archives = {}
        self.refs = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def archive(self, relative):
        relative = safe_relative(relative)
        if relative not in self.archives:
            path = self.root / relative
            stat = path.stat()
            tar = self.stack.enter_context(tarfile.open(path, "r:*"))
            members = {}
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = safe_relative(member.name)
                require(name not in members, f"Duplicate archive member: {relative}!{name}")
                members[name] = member
            self.archives[relative] = (tar, members, (stat.st_size, stat.st_mtime_ns))
        return self.archives[relative]

    def read(self, key, relative, member=None):
        relative = safe_relative(relative)
        path = self.root / relative
        before = path.stat()
        if member is None:
            payload = path.read_bytes()
        else:
            member = safe_relative(member)
            tar, members, signature = self.archive(relative)
            require(signature == (before.st_size, before.st_mtime_ns), f"Source changed: {path}")
            require(member in members, f"Missing member: {relative}!{member}")
            with tar.extractfile(members[member]) as stream:
                payload = stream.read()
            require(len(payload) == members[member].size, f"Truncated member: {member}")
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Source changed while reading: {path}")
        ref = {"path": relative, "member": member, "bytes": len(payload),
               "sha256": hashlib.sha256(payload).hexdigest()}
        if key in self.refs:
            require(self.refs[key] == ref, f"Source changed: {key}")
        self.refs[key] = ref
        return payload


def timestamp_array(payload, *, strict, allow_empty=False):
    stamps = []
    for line in payload.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"frame_(\d+)_(\d+)(?:\s+512)?", line.strip())
        require(match is not None, f"Malformed timestamp row: {line[:100]}")
        timestamp, index = map(int, match.groups())
        require(index == len(stamps), "Timestamp source frame indices must start at 0 and be contiguous")
        require(0 <= timestamp <= np.iinfo(np.int64).max, "Timestamp outside int64 range")
        stamps.append(timestamp)
    require(allow_empty or stamps, "Empty observation timeline")
    ts = np.asarray(stamps, dtype=np.int64)
    require(np.all(np.diff(ts) > 0 if strict else np.diff(ts) >= 0), "Non-monotonic timestamps")
    return ts


def load_raw(payload):
    require(not payload.startswith(b"\x93NUMPY"), "Expected headerless raw float32, not a NumPy container")
    require(len(payload) % (16 * 32 * 4) == 0, "Raw tactile byte count is not a multiple of 2048")
    return np.frombuffer(payload, dtype="<f4").reshape(-1, 16, 32)


def load_median(payload):
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    require(isinstance(value, np.ndarray) and value.shape == (16, 32)
            and value.dtype.kind == "f", "Median must be a floating [16,32] array")
    return value


def load_robot(payload, frames, slots):
    # The trusted published format is a NumPy scalar containing a dictionary.
    value = np.load(io.BytesIO(payload), allow_pickle=True)
    require(isinstance(value, np.ndarray) and value.shape == (), "Unsupported robot.npy container")
    data = value.item()
    require(isinstance(data, dict) and {"T_worlds", "openings"} <= data.keys(), "Missing robot fields")
    t, o = np.asarray(data["T_worlds"]), np.asarray(data["openings"])
    original_shapes = {"T_worlds": list(t.shape), "openings": list(o.shape)}
    if t.ndim == 3:
        require(len(slots) == 1 and t.shape == (frames, 4, 4)
                and o.shape == (frames,), "Malformed single-hand arrays")
        t, o = t[:, None], o[:, None]
    require(t.shape == (frames, len(slots), 4, 4) and o.shape == (frames, len(slots)),
            "Robot shape/frame count does not match observations and audited hand binding")
    require(t.dtype.kind == "f" and o.dtype.kind == "f"
            and t.dtype.itemsize in (4, 8) and o.dtype.itemsize in (4, 8), "Unsupported robot dtype")
    if "bimanual" in data:
        require(bool(data["bimanual"]) == (len(slots) == 2), "bimanual disagrees with audited slots")
    return t, o, original_shapes


def load_actions(payload, frames, slots, t, openings):
    data = np.load(io.BytesIO(payload), allow_pickle=True).item()
    require(isinstance(data, dict) and "actions" in data, "Missing original actions")
    actions = np.asarray(data["actions"])
    shape = list(actions.shape)
    if actions.ndim == 3:
        require(len(slots) == 1 and actions.shape == (frames, 5, 3), "Malformed single-hand actions")
        actions = actions[:, None]
    require(actions.shape == (frames, len(slots), 5, 3)
            and actions.dtype.kind == "f" and actions.dtype.itemsize in (4, 8), "Bad original actions")
    check_actions(actions, t, openings)
    return actions, shape


def check_actions(actions, t, openings):
    require(np.array_equal(actions[..., 0, :], t[..., :3, 3], equal_nan=True), "Actions translation mismatch")
    require(np.array_equal(actions[..., 1:4, :], t[..., :3, :3], equal_nan=True), "Actions rotation mismatch")
    require(np.array_equal(actions[..., 4, 0], openings, equal_nan=True), "Actions opening mismatch")
    require(np.all(actions[..., 4, 1:] == 0), "Actions last row must be [opening, 0, 0]")


def select_raw(raw_ts, obs):
    index = (np.searchsorted(raw_ts, obs, side="right") - 1).astype(np.int64)
    age = np.full(len(obs), -1, dtype=np.int64)
    has = index >= 0
    age[has] = obs[has] - raw_ts[index[has]]
    valid = has & (age >= 0) & (age <= MAX_AGE_US)
    return index, age, valid


def tactile_arrays(raw, median, raw_ts, obs):
    require(len(raw) == len(raw_ts), "Raw tactile data/timestamp count mismatch")
    index, age, time_valid = select_raw(raw_ts, obs)
    values = np.zeros((len(obs), 12, 32), dtype=np.float32)
    valid = np.zeros(values.shape, dtype=bool)
    if len(raw):
        selected = raw[np.maximum(index, 0), :12].astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = selected - median[:12].astype(np.float64)
            positive = np.maximum(delta, 0).astype(np.float32)
        valid = (time_valid[:, None, None] & np.isfinite(selected)
                 & np.isfinite(median[:12])[None] & np.isfinite(delta) & np.isfinite(positive))
        valid[:, :, 31] = False
        values = np.where(valid, positive, 0).astype(np.float32)
    return values, valid, index, age


def pose_validity(t, openings):
    finite = np.isfinite(t).all(axis=(-1, -2)) & np.isfinite(openings)
    safe = np.where(finite[..., None, None], t.astype(np.float64), np.eye(4))
    r = safe[..., :3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        orth = np.max(np.abs(np.swapaxes(r, -1, -2) @ r - np.eye(3)), axis=(-1, -2))
        det = np.abs(np.linalg.det(r) - 1)
        bottom = np.max(np.abs(safe[..., 3, :] - [0, 0, 0, 1]), axis=-1)
    return finite & (orth <= RIGID_TOL) & (det <= RIGID_TOL) & (bottom <= RIGID_TOL)


def controller_arrays(t, openings, slots):
    frames = len(t)
    points = np.zeros((frames, 2, 30, 3), dtype=np.float32)
    mask = np.zeros((frames, 2, 30), dtype=bool)
    clipped = np.zeros((frames, 2), dtype=bool)
    good = pose_validity(t, openings)
    for slot, hand in enumerate(slots):
        target = HAND_ORDER.index(hand)
        opening = openings[:, slot].astype(np.float64)
        clipped[:, target] = np.isfinite(opening) & ((opening < .04) | (opening > .112))
        q = .038 - .033 * (np.clip(np.where(np.isfinite(opening), opening, .04), .04, .112) - .04) / .072
        pose = np.where(good[:, slot, None, None], t[:, slot].astype(np.float64), np.eye(4))
        point = 0
        for sign, base in [(1, [-.04246242, .0835, .0097]), (-1, [.04246242, .0835, .0107])]:
            for row in GRID_ROWS:
                for col in GRID_COLS:
                    finger = np.array([sign * .007, -.056 - .002 * col, (.025 / 12) * (11.5 - row)])
                    root = np.broadcast_to(np.asarray(base) + finger * [1, -1, -1], (frames, 3)).copy()
                    root[:, 0] += sign * q
                    with np.errstate(over="ignore", invalid="ignore"):
                        world = ((pose[:, :3, :3] @ root[:, :, None])[:, :, 0] + pose[:, :3, 3]).astype(np.float32)
                    point_good = good[:, slot] & np.isfinite(world).all(axis=1)
                    points[:, target, point] = np.where(point_good[:, None], world, 0)
                    mask[:, target, point] = point_good
                    point += 1
    return points, mask, clipped


def gs_arrays(payload, frame, timestamp):
    from plyfile import PlyData

    ply = PlyData.read(io.BytesIO(payload))
    require("vertex" in ply, "PLY has no vertex element")
    vertex = ply["vertex"].data
    require(len(vertex) > 0, "Empty GS frame")
    arrays = {}
    for name, fields in GS_FIELDS.items():
        require(set(fields) <= set(vertex.dtype.names or ()), f"Missing PLY fields: {name}")
        for field in fields:
            require(vertex[field].dtype.kind == "f" and vertex[field].dtype.itemsize == 4,
                    f"Expected float32 PLY field: {field}")
        arrays[name] = np.column_stack([vertex[field] for field in fields]).astype(np.float32)
        require(np.isfinite(arrays[name]).all(), f"Non-finite GS field: {name}, frame {frame}")
    q = arrays["quaternions"].astype(np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    require(np.all(np.isfinite(norm) & (norm > 0)), f"Invalid quaternion norm, frame {frame}")
    arrays["quaternions"] = (q / norm).astype(np.float32)
    arrays["frame_id"] = np.asarray(frame, dtype=np.int64)
    arrays["timestamp_us"] = np.asarray(timestamp, dtype=np.int64)
    return arrays


def episode_layout(reader, scene, episode):
    directories = discover_episodes(reader.root / "processed" / scene)
    require(episode in directories, f"Missing source episode: {scene}/{episode}")
    prefix = directories[episode].relative_to(reader.root).as_posix()
    meta = json.loads(reader.read("episode_metadata", f"{prefix}/metadata.json").decode("utf-8"))
    cameras = meta.get("cameras")
    require(isinstance(cameras, list) and cameras and all(
        isinstance(c, str) and re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in cameras),
        "Invalid camera list in episode metadata")
    require(len(cameras) == len(set(cameras)), "Duplicate camera names")
    camera_refs = {}
    for camera in sorted(cameras):
        name = f"{camera}/aligned_timestamps.txt"
        if (reader.root / prefix / name).is_file():
            camera_refs[camera] = (f"{prefix}/{name}", None)
        else:
            archive = f"{prefix}/cameras.tar"
            _, members, _ = reader.archive(archive)
            require(name in members, f"Missing {archive}!{name}")
            camera_refs[camera] = (archive, name)
    loose = reader.root / prefix / "splatfacto"
    if loose.is_dir():
        gs_refs = [(p.name, (f"{prefix}/splatfacto/{p.name}", None)) for p in loose.glob("*.ply")]
    else:
        archive = f"{prefix}/splatfacto.tar"
        _, members, _ = reader.archive(archive)
        gs_refs = [(PurePosixPath(name).name, (archive, name)) for name in members
                   if name.startswith("splatfacto/") and name.endswith(".ply")]
    by_frame = {}
    for name, ref in gs_refs:
        match = re.fullmatch(r"splat_(\d+)\.ply", name)
        require(match is not None, f"Unexpected GS filename: {name}")
        frame = int(match[1])
        require(frame not in by_frame, f"Duplicate GS frame {frame}")
        by_frame[frame] = ref
    require(by_frame, f"No GS frames for {scene}/episode_{episode}")
    return prefix, camera_refs, by_frame


def observation_timeline(reader, camera_refs):
    # All published cameras must agree; never use metadata's research subwindow.
    camera = sorted(camera_refs)[0]
    obs = None
    for name, ref in sorted(camera_refs.items()):
        ts = timestamp_array(reader.read(f"camera_timestamps_{name}", *ref), strict=True)
        if obs is None:
            obs = ts
        else:
            require(np.array_equal(obs, ts), f"Aligned camera timelines disagree: {name}")
    primary = timestamp_array(reader.read("observation", *camera_refs[camera]), strict=True)
    require(np.array_equal(primary, obs), "Observation source changed while reading")
    return camera, obs


def find_raw_sources(source_root, scene, obs):
    """Pair by exact stem and unique time-interval overlap, never sorted ordinal.

    A partial download may omit unrelated episodes without renumbering anything.
    Multiple medians require an explicit reviewed scene/sensor choice.
    Missing/ambiguous selected recordings or missing chosen baselines fail.
    """
    selected = []
    for sensor_name in STREAMS:
        sensor = "brics-odroid_" + sensor_name
        directory = Path(source_root) / "raw" / scene / sensor
        require(directory.is_dir(), f"Missing tactile sensor directory: {directory}")
        if scene in SCENE_MEDIANS:
            require(sensor_name in SCENE_MEDIANS[scene], f"Missing reviewed baseline for {scene}/{sensor}")
            median = directory / SCENE_MEDIANS[scene][sensor_name]
        else:
            medians = sorted(directory.glob("median_*.npy"))
            require(len(medians) == 1, f"Expected one unambiguous baseline in {directory}, found {len(medians)}")
            median = medians[0]
        require(median.is_file() and re.fullmatch(r"median_\d+\.npy", median.name),
                f"Missing or unexpected baseline: {median}")
        data = {p.stem: p for p in directory.glob("*.npy") if not p.name.startswith("median_")}
        times = {p.stem: p for p in directory.glob("*.txt")}
        require(data and data.keys() == times.keys(), f"Unpaired raw .npy/.txt files in {directory}")
        candidates = []
        for stem in sorted(data):
            require(re.fullmatch(re.escape(sensor) + r"_\d+", stem) is not None,
                    f"Unexpected raw recording name: {stem}")
            ts = timestamp_array(times[stem].read_bytes(), strict=False)
            if ts[0] <= obs[-1] and ts[-1] >= obs[0]:
                require(data[stem].stat().st_size == len(ts) * 16 * 32 * 4,
                        f"Raw byte/timestamp count mismatch: {data[stem]}")
                candidates.append((data[stem], times[stem], median))
        require(len(candidates) == 1,
                f"{scene}/{sensor}: expected one raw recording overlapping observation "
                f"[{obs[0]}, {obs[-1]}], found {len(candidates)}; cannot guess pairing")
        selected.append(tuple(p.relative_to(source_root).as_posix() for p in candidates[0]))
    return selected


def output_paths(source_root, output_root, scene, episode, *, allow_existing=False):
    target = output_root / scene / f"episode_{episode}"
    partial = target.with_name(f".episode_{episode}.incomplete")
    require(not (output_root / scene).is_symlink(), "Output scene directory must not be a symlink")
    backup = target.with_name(f".episode_{episode}.previous")
    for path in (target, partial, backup):
        resolved = path.resolve()
        require(resolved.is_relative_to(output_root.resolve()), f"Output escapes destination: {path}")
        for source in (source_root / "raw", source_root / "processed"):
            require(not resolved.is_relative_to(source.resolve()), f"Output is inside source data: {path}")
        require(not path.is_symlink(), f"Refusing linked output: {path}")
        if path == target and allow_existing:
            require(not path.exists() or path.is_dir(), f"Output is not a directory: {path}")
        elif path == backup and allow_existing and target.is_dir():
            # A published target can be validated/skipped despite cleanup leftovers.
            # A backup without a target still requires recovery, never regeneration.
            require(not path.exists() or path.is_dir(), f"Backup is not a directory: {path}")
        else:
            require(not path.exists(), f"Refusing existing/partial/backup output: {path}")
    return target, partial


def preflight(source_root, output_root, scene, episodes):
    require(source_root.is_dir(), f"Source root missing: {source_root}")
    for source in (source_root / "raw", source_root / "processed"):
        require(source.is_dir(), f"Source directory missing: {source}")
    plans = {}
    seen_recordings = set()
    for ep in episodes:
        slots = hand_slots(scene, ep)
        output_paths(source_root, output_root, scene, ep, allow_existing=True)
        with SourceReader(source_root) as reader:
            prefix, camera_refs, gs_refs = episode_layout(reader, scene, ep)
            camera, obs = observation_timeline(reader, camera_refs)
            require(sorted(gs_refs) == list(range(len(obs))), f"GS gaps/extra frames in episode {ep}")
            robot_payload = reader.read("robot", f"{prefix}/robot/robot.npy")
            t, o, _ = load_robot(robot_payload, len(obs), slots)
            load_actions(robot_payload, len(obs), slots, t, o)
            tactile_sources = find_raw_sources(source_root, scene, obs)
            for channel, paths in enumerate(tactile_sources):
                raw_path, time_path, median_path = paths
                require(raw_path not in seen_recordings, f"One raw recording matched multiple episodes: {raw_path}")
                seen_recordings.add(raw_path)
                raw = load_raw(reader.read(f"tactile_{channel}_raw", raw_path))
                ts = timestamp_array(reader.read(f"tactile_{channel}_timestamps", time_path), strict=False)
                require(len(raw) == len(ts), f"Raw/time count mismatch: {raw_path}")
                load_median(reader.read(f"tactile_{channel}_median", median_path))
            plans[ep] = {"prefix": prefix, "camera": camera, "camera_refs": camera_refs,
                         "gs_refs": gs_refs, "obs": obs, "tactile_sources": tactile_sources,
                         "source_refs": dict(reader.refs)}
    return plans


def write_npz(path, arrays):
    require(all(isinstance(v, np.ndarray) and v.dtype.kind in "biuf" for v in arrays.values()),
            "Output NPZ must contain numeric/bool arrays only")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def require_h5py():
    """Depth is optional; ordinary reconstruction never imports h5py."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Depth mode requires h5py in the server environment; install it before --depth-only") from exc
    return h5py


def checked_path(path):
    """Reject linked components before resolving paths (including missing targets)."""
    path = Path(path).absolute()
    for component in (path, *path.parents):
        require(not component.is_symlink(), f"Refusing symbolic-link path: {component}")
        require(not getattr(component, "is_junction", lambda: False)(), f"Refusing junction: {component}")
    return path.resolve()


def depth_roots(source_root, core_root, depth_root):
    source_root, core_root, depth_root = map(checked_path, (source_root, core_root, depth_root))
    require(source_root.is_dir() and core_root.is_dir(), "Depth mode requires existing source and core roots")
    for protected in (source_root, core_root):
        require(not depth_root.is_relative_to(protected) and not protected.is_relative_to(depth_root),
                f"Depth root must be disjoint from core/source data: {depth_root}, {protected}")
    require(not depth_root.exists() or depth_root.is_dir(), f"Depth root is not a directory: {depth_root}")
    return source_root, core_root, depth_root


def read_core_depth_binding(directory, *, expected_scene=None, expected_episode=None):
    directory = checked_path(directory)
    require(directory.is_dir(), f"Core episode must already exist: {directory}")
    meta_path = checked_path(directory / "metadata.json")
    time_path = checked_path(directory / "timestamps.npz")
    require(meta_path.is_file() and time_path.is_file(), f"Missing core metadata/timestamps: {directory}")
    metadata_hash, timestamp_hash = hash_file(meta_path), hash_file(time_path)
    with meta_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    scene, episode, frames = metadata["scene"], metadata["episode"], metadata["frames"]
    require(metadata["schema_version"] == VERSION and scene in SCENE_BINDINGS, "Unsupported core schema/scene")
    require(type(episode) is int and episode in SCENE_BINDINGS[scene], "Unreviewed core episode")
    require(type(frames) is int and frames > 0, "Invalid core frame count")
    require(expected_scene is None or expected_scene == scene, "Core scene mismatch")
    require(expected_episode is None or expected_episode == episode, "Core episode mismatch")
    require(directory.name == f"episode_{episode}" and directory.parent.name == scene,
            "Core directory does not match scene/episode")
    cameras = metadata["observation_cameras"]
    require(isinstance(cameras, list) and cameras and all(isinstance(c, str) and
            re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in cameras) and cameras == sorted(set(cameras)),
            "Invalid core camera list")
    require(metadata["output_files"].get("timestamps.npz") == timestamp_hash, "Core timestamp SHA-256 mismatch")
    with np.load(time_path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "Duplicate core timestamp fields")
        obs = archive["observation_timestamp_us"]
    require(obs.shape == (frames,) and obs.dtype == np.int64 and np.all(obs >= 0)
            and np.all(np.diff(obs) > 0), "Invalid core observation timeline")
    require(hash_file(meta_path) == metadata_hash and hash_file(time_path) == timestamp_hash,
            "Core binding changed while reading")
    return metadata, obs, {"metadata_sha256": metadata_hash, "timestamps_sha256": timestamp_hash}


def check_camera_arrays(intrinsics, camera_to_world, cameras):
    require(intrinsics.shape == (len(cameras), 3, 3) and intrinsics.dtype == np.float64,
            "intrinsics must be float64 [V,3,3]")
    require(camera_to_world.shape == (len(cameras), 4, 4) and camera_to_world.dtype == np.float64,
            "camera_to_world must be float64 [V,4,4]")
    require(np.isfinite(intrinsics).all() and np.isfinite(camera_to_world).all(), "Non-finite calibration")
    require(np.all(intrinsics[:, 0, 0] > 0) and np.all(intrinsics[:, 1, 1] > 0)
            and np.allclose(intrinsics[:, 2, :], [0, 0, 1], atol=1e-8, rtol=0)
            and np.all(np.abs(np.linalg.det(intrinsics)) > 1e-12), "Invalid camera intrinsics")
    rotation = camera_to_world[:, :3, :3]
    require(np.allclose(camera_to_world[:, 3, :], [0, 0, 0, 1], atol=RIGID_TOL, rtol=0)
            and np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=RIGID_TOL, rtol=0)
            and np.allclose(np.linalg.det(rotation), 1, atol=RIGID_TOL, rtol=0),
            "Extrinsics must be right-handed rigid camera-to-world transforms")


def read_depth_metadata(directory, core_directory, *, expected_scene=None, expected_episode=None):
    """Lightweight loader check: binding, manifest, calibration, timeline; no depth hashes/reads."""
    directory = checked_path(directory)
    core, obs, binding = read_core_depth_binding(core_directory, expected_scene=expected_scene,
                                                expected_episode=expected_episode)
    with checked_path(directory / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    scene, episode, frames, cameras = core["scene"], core["episode"], core["frames"], core["observation_cameras"]
    require(metadata["schema_version"] == DEPTH_VERSION and metadata["script_version"] == DEPTH_SCRIPT_VERSION,
            "Unsupported depth schema/script version")
    require(type(metadata["episode"]) is int and type(metadata["frames"]) is int,
            "Depth episode/frame count must be integers")
    require(metadata["scene"] == scene and metadata["episode"] == episode and metadata["frames"] == frames,
            "Depth/core scene, episode or frames mismatch")
    require(directory.parent.name == scene and directory.name in
            (f"episode_{episode}", f".episode_{episode}.incomplete"), "Depth directory scene/episode mismatch")
    require(metadata["cameras"] == cameras and metadata["core_binding"] == binding,
            "Stale depth extension: core binding or camera list changed")
    require(metadata["depth_unit"] == "mm" and type(metadata["invalid_value"]) is int
            and metadata["invalid_value"] == 0
            and metadata["depth_kind"] == "camera_z_expected"
            and metadata["extrinsics_convention"] == "camera_to_world"
            and metadata["coordinate_system"] == "native_world_metres",
            "Unsupported depth/calibration convention")
    require(all(type(metadata[key]) is int and metadata[key] > 0 for key in ("height", "width")),
            "Invalid depth image dimensions")
    expected_files = {"camera_parameters.npz", *(f"{c}/rendered_depth.h5" for c in cameras)}
    files = metadata["output_files"]
    require(isinstance(files, dict) and set(files) == expected_files and all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in files.values()),
        "Incomplete or invalid depth output manifest")
    actual_files = set()
    for path in directory.rglob("*"):
        checked_path(path)
        if path.is_file():
            actual_files.add(path.relative_to(directory).as_posix())
    require(actual_files == expected_files | {"metadata.json"}, "Missing or unexpected depth files")
    parameters_path = directory / "camera_parameters.npz"
    require(hash_file(parameters_path) == files["camera_parameters.npz"], "Camera parameter SHA-256 mismatch")
    with np.load(parameters_path, allow_pickle=False) as archive:
        require(len(archive.files) == 3 and set(archive.files) ==
                {"intrinsics", "camera_to_world", "observation_timestamp_us"}, "Bad camera parameter fields")
        intrinsics, extrinsics, timestamps = (archive[k] for k in
                                             ("intrinsics", "camera_to_world", "observation_timestamp_us"))
    check_camera_arrays(intrinsics, extrinsics, cameras)
    require(timestamps.dtype == np.int64 and np.array_equal(timestamps, obs), "Depth/core timeline mismatch")
    return metadata


class DepthSourceReader(SourceReader):
    """Strict archive access plus bounded-memory, byte-preserving depth copying."""

    def archive(self, relative):
        relative = safe_relative(relative)
        checked_path(self.root / relative)
        if relative not in self.archives:
            path = self.root / relative
            before = path.stat()
            tar = self.stack.enter_context(tarfile.open(path, "r:*"))
            members, seen = {}, set()
            for member in tar.getmembers():
                name = safe_relative(member.name)
                require(name not in seen, f"Duplicate archive member: {relative}!{name}")
                seen.add(name)
                require(member.isfile() or member.isdir(), f"Linked/special archive member: {relative}!{name}")
                if member.isfile():
                    members[name] = member
            self.archives[relative] = (tar, members, (before.st_size, before.st_mtime_ns))
        return self.archives[relative]

    def read(self, key, relative, member=None):
        checked_path(self.root / safe_relative(relative))
        return super().read(key, relative, member)

    def stream(self, key, relative, member=None, *, destination=None):
        relative = safe_relative(relative)
        path = checked_path(self.root / relative)
        before = path.stat()
        with ExitStack() as stack:
            if member is None:
                source = stack.enter_context(path.open("rb"))
                size = before.st_size
            else:
                member = safe_relative(member)
                tar, members, signature = self.archive(relative)
                require(signature == (before.st_size, before.st_mtime_ns), f"Source changed: {path}")
                require(member in members, f"Missing depth archive member: {member}")
                source = stack.enter_context(tar.extractfile(members[member]))
                size = members[member].size
            target = stack.enter_context(Path(destination).open("xb")) if destination is not None else None
            digest, count = hashlib.sha256(), 0
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
                count += len(block)
                if target is not None:
                    target.write(block)
        after = path.stat()
        require(count == size and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Source changed/truncated while streaming: {path}")
        ref = {"path": relative, "member": member, "bytes": count, "sha256": digest.hexdigest()}
        require(key not in self.refs or self.refs[key] == ref, f"Source changed: {key}")
        self.refs[key] = ref
        return ref


def depth_camera_source(reader, key, prefix, relative, recorded_refs):
    """Keep a recorded source location; discover only sources not yet recorded."""
    loose, archive = f"{prefix}/{relative}", f"{prefix}/cameras.tar"
    if key in recorded_refs:
        ref = recorded_refs[key]
        require(isinstance(ref, dict) and set(ref) == {"path", "member", "bytes", "sha256"},
                f"Malformed recorded depth source: {key}")
        require((ref["path"], ref["member"]) in ((loose, None), (archive, relative)),
                f"Wrong recorded depth source path: {key}")
        require(type(ref["bytes"]) is int and ref["bytes"] >= 0
                and isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
                f"Invalid recorded depth source hash/size: {key}")
        path, member = ref["path"], ref["member"]
    else:
        path, member = (loose, None) if checked_path(reader.root / loose).is_file() else (archive, relative)
    # Missing or changed recorded sources fail; an unrecorded copy cannot replace them.
    source = checked_path(reader.root / path)
    require(source.is_file(), f"Missing depth source: {path}")
    if member is not None:
        _, members, _ = reader.archive(path)
        require(member in members, f"Missing depth archive member: {path}!{member}")
    return path, member


def depth_source_layout(reader, core, obs, *, source_refs=None):
    """New depth uses core timeline provenance; existing depth uses its own manifest."""
    scene, episode, cameras = core["scene"], core["episode"], core["observation_cameras"]
    core_refs = core.get("source_files", {})
    require(isinstance(core_refs, dict), "Malformed core source manifest")
    if source_refs is not None:
        expected = {"episode_metadata", "extrinsics", *(f"camera_timestamps_{c}" for c in cameras),
                    *(f"depth_{c}" for c in cameras)}
        require(isinstance(source_refs, dict) and set(source_refs) == expected,
                "Incomplete or unexpected recorded depth source manifest")
    prefix = safe_relative(core["source_episode_directory"])
    match = re.fullmatch(re.escape(f"processed/{scene}/episode_") + r"(\d+)", prefix)
    require(match is not None and int(match[1]) == episode, "Wrong core source episode directory")
    metadata = json.loads(reader.read("episode_metadata", f"{prefix}/metadata.json").decode("utf-8"))
    source_cameras = metadata["cameras"]
    require(isinstance(source_cameras, list) and sorted(source_cameras) == cameras, "Source/core cameras mismatch")
    wh = metadata["WH"]
    require(isinstance(wh, list) and len(wh) == 2 and all(type(x) is int and x > 0 for x in wh),
            "Invalid source WH")
    all_intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
    require(all_intrinsics.shape == (len(cameras), 3, 3), "Bad source intrinsics dimensions")
    intrinsics = all_intrinsics[[source_cameras.index(c) for c in cameras]]
    payload = reader.read("extrinsics", f"raw/{scene}/calibration_refined/extrinsics.npy")
    # Like the published robot.npy, the trusted original calibration is a pickled dictionary.
    value = np.load(io.BytesIO(payload), allow_pickle=True)
    require(isinstance(value, np.ndarray) and value.shape == (), "Unsupported extrinsics container")
    extrinsics = value.item()
    require(isinstance(extrinsics, dict) and all(c in extrinsics for c in cameras), "Missing camera extrinsics")
    camera_to_world = np.asarray([extrinsics[c] for c in cameras], dtype=np.float64)
    check_camera_arrays(intrinsics, camera_to_world, cameras)
    depth_refs = {}
    for camera in cameras:
        for filename in ("aligned_timestamps.txt", "rendered_depth.h5"):
            relative = f"{camera}/{filename}"
            key = f"camera_timestamps_{camera}" if filename == "aligned_timestamps.txt" else f"depth_{camera}"
            recorded = source_refs if source_refs is not None else core_refs
            ref = depth_camera_source(reader, key, prefix, relative, recorded)
            if filename == "aligned_timestamps.txt":
                timeline = timestamp_array(reader.read(key, *ref), strict=True)
                require(np.array_equal(timeline, obs), f"Full published depth/core timeline differs: {camera}")
            else:
                depth_refs[camera] = ref
    for key in ("episode_metadata", *(f"camera_timestamps_{c}" for c in cameras)):
        if key in core_refs:
            require(reader.refs[key] == core_refs[key], f"Depth source differs from core provenance: {key}")
    if source_refs is not None:
        for key, ref in reader.refs.items():
            require(ref == source_refs[key], f"Depth source hashes/paths changed: {key}")
    return {"width": wh[0], "height": wh[1], "intrinsics": intrinsics,
            "camera_to_world": camera_to_world, "depth_refs": depth_refs}


def check_depth_h5(path, frames, height, width, *, read_pixels=True):
    h5py = require_h5py()
    with h5py.File(checked_path(path), "r") as archive:
        require(set(archive.keys()) == {"data"}, f"Expected one H5 data array: {path}")
        require(isinstance(archive.get("data", getlink=True), h5py.HardLink), f"External/soft H5 data link: {path}")
        data = archive["data"]
        require(isinstance(data, h5py.Dataset) and data.dtype == np.uint16
                and data.shape == (frames, height, width), f"Expected uint16 [F,H,W] depth: {path}")
        require(not data.is_virtual and not data.external, f"External/virtual H5 storage: {path}")
        if read_pixels:
            # Released videos are frame-chunked: one ~1.8 MB frame at a time avoids
            # repeatedly decompressing that chunk when it exceeds the HDF5 cache.
            for frame in range(frames):
                data[frame]


def depth_output_paths(depth_root, scene, episode, *, allow_existing=True):
    require(scene in SCENE_BINDINGS and type(episode) is int and episode in SCENE_BINDINGS[scene],
            "Unreviewed depth scene/episode")
    target = Path(depth_root) / scene / f"episode_{episode}"
    partial, backup = (target.with_name(f".episode_{episode}.{suffix}") for suffix in ("incomplete", "previous"))
    for path in (target, partial, backup):
        checked_path(path)
        require(not path.exists() or path.is_dir(), f"Depth output is not a directory: {path}")
    if target.exists():
        for path in target.rglob("*"):
            checked_path(path)
    require(not partial.exists(), f"Inspect existing unfinished depth output: {partial}")
    require(allow_existing or not target.exists(), f"Depth output already exists: {target}")
    require(not backup.exists() or target.exists(), f"Recover depth backup before rebuilding: {backup}")
    return target, partial, backup


def _validators():
    if __package__:
        from . import validate
    else:
        import validate
    return validate


def rebuild_depth_episode(source_root, core_root, depth_root, scene, episode, *, overwrite=False):
    """Copy only the depth extension. Never writes or fully revalidates core arrays."""
    require_h5py()
    source_root, core_root, depth_root = depth_roots(source_root, core_root, depth_root)
    core_directory = core_root / scene / f"episode_{episode}"
    core, obs, binding = read_core_depth_binding(core_directory, expected_scene=scene, expected_episode=episode)
    target, partial, backup = depth_output_paths(depth_root, scene, episode)
    if target.exists() and not overwrite:
        result = _validators().validate_depth_episode(target, core_directory=core_directory,
                    expected_scene=scene, expected_episode=episode, source_root=source_root, verify_sources=True)
        print(f"Skipped (full depth validation passed): {target}", flush=True)
        return {**result, "skipped": True}
    require(not backup.exists(), f"Inspect previous depth backup before overwriting: {backup}")
    old_identity = (target.stat().st_dev, target.stat().st_ino) if target.exists() else None
    with DepthSourceReader(source_root) as reader:
        plan = depth_source_layout(reader, core, obs)
        partial.mkdir(parents=True, exist_ok=False)
        try:
            output_files = {}
            for camera in core["observation_cameras"]:
                (partial / camera).mkdir()
                destination = partial / camera / "rendered_depth.h5"
                ref = reader.stream(f"depth_{camera}", *plan["depth_refs"][camera], destination=destination)
                check_depth_h5(destination, len(obs), plan["height"], plan["width"], read_pixels=False)
                output_files[f"{camera}/rendered_depth.h5"] = ref["sha256"]
            params = partial / "camera_parameters.npz"
            write_npz(params, {"intrinsics": plan["intrinsics"], "camera_to_world": plan["camera_to_world"],
                               "observation_timestamp_us": obs})
            output_files["camera_parameters.npz"] = hash_file(params)
            metadata = {"schema_version": DEPTH_VERSION, "script_version": DEPTH_SCRIPT_VERSION,
                "scene": scene, "episode": episode, "frames": len(obs), "cameras": core["observation_cameras"],
                "height": plan["height"], "width": plan["width"], "depth_unit": "mm", "invalid_value": 0,
                "depth_kind": "camera_z_expected", "extrinsics_convention": "camera_to_world",
                "coordinate_system": "native_world_metres", "core_binding": binding,
                "source_root": str(source_root), "source_episode_directory": core["source_episode_directory"],
                "source_files": reader.refs, "output_files": output_files,
                "created_utc": datetime.now(timezone.utc).isoformat()}
            with (partial / "metadata.json").open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            result = _validators().validate_depth_episode(partial, core_directory=core_directory,
                        expected_scene=scene, expected_episode=episode, source_root=source_root, verify_sources=True)
            checked_path(target)
            if old_identity is None:
                require(not target.exists(), f"Depth output appeared during copying: {target}")
                partial.rename(target)
            else:
                require(target.is_dir() and (target.stat().st_dev, target.stat().st_ino) == old_identity,
                        f"Depth output changed during copying: {target}")
                require(not backup.exists(), f"Depth backup appeared during copying: {backup}")
                target.rename(backup)
                try:
                    partial.rename(target)
                except Exception:
                    backup.rename(target)
                    raise
        except Exception:
            print(f"Depth supplement failed; core data untouched. Inspect {partial}, {target}, {backup}", file=sys.stderr)
            raise
    print(f"Completed depth only: {target}", flush=True)
    if old_identity is not None:
        try:
            shutil.rmtree(backup)
        except Exception as exc:
            print(f"WARNING: depth publication succeeded; old backup cleanup failed at {backup}: {exc}. "
                  "Validation and default skipping remain available.", file=sys.stderr, flush=True)
    return {**result, "skipped": False}


def run_depth_only(args):
    require_h5py()
    source_root, core_root, depth_root = depth_roots(args.source_root, args.output_root,
        args.depth_root if args.depth_root is not None else args.output_root.parent / DEPTH_VERSION)
    available = discover_episodes(core_root / args.scene)
    episodes = args.episodes if args.episodes is not None else list(available)
    require(set(episodes) <= set(available), "Depth mode cannot create missing core episodes; rebuild core separately")
    import fcntl
    scene_dir = depth_root / args.scene
    with ExitStack() as stack:
        # Core reconstruction locks this same directory exclusively. Hold a shared
        # lock through publication so a concurrently replaced core cannot bind stale depth.
        core_descriptor = os.open(core_root / args.scene, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, core_descriptor)
        try:
            fcntl.flock(core_descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"A core reconstruction is using {core_root / args.scene}") from exc
        # Preflight every selected episode before any mutation. No robot/GS/tactile reads.
        for episode in episodes:
            core, obs, _ = read_core_depth_binding(available[episode], expected_scene=args.scene, expected_episode=episode)
            target, _, _ = depth_output_paths(depth_root, args.scene, episode)
            source_refs = None
            if target.exists() and not args.overwrite:
                metadata = read_depth_metadata(target, available[episode], expected_scene=args.scene,
                                               expected_episode=episode)
                _validators().check_depth_sources_schema(metadata)
                source_refs = metadata["source_files"]
            with DepthSourceReader(source_root) as reader:
                depth_source_layout(reader, core, obs, source_refs=source_refs)
        scene_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(scene_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, descriptor)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another depth supplement is using {scene_dir}") from exc
        for episode in episodes:
            rebuild_depth_episode(source_root, core_root, depth_root, args.scene, episode, overwrite=args.overwrite)
    return 0


def parameters():
    return {"max_age_us": MAX_AGE_US, "max_age_status": "provisional", "rigid_atol": RIGID_TOL,
            "rigid_rtol": 0, "opening_clip_m": list(OPENING_LIMITS), "surface_offset_m": .007,
            "crop_rows": [0, 12], "invalid_columns": [31], "row_flip": False,
            "tactile_formula": "maximum(raw - median, 0)", "tactile_subtraction_dtype": "float64",
            "tactile_output_dtype": "float32", "contact_threshold": None,
            "duplicate_timestamps": "last raw record wins", "sampling": "latest past, no interpolation",
            "gs_coordinate_transform": "identity; native world metres", "quaternion_order": "WXYZ",
            "sh_rest_order": "f_rest_0..44 unchanged; channel/basis layout pending",
            "gs_correspondence": "no fixed cross-frame identity"}


def rebuild_episode(source_root, output_root, scene, episode, plan, *, overwrite=False):
    target, partial = output_paths(source_root, output_root, scene, episode, allow_existing=overwrite)
    backup = target.with_name(f".episode_{episode}.previous")
    require(not backup.exists() and not backup.is_symlink(),
            f"Inspect/remove the previous backup before another overwrite: {backup}")
    old_identity = (target.stat().st_dev, target.stat().st_ino) if target.exists() else None
    partial.mkdir(parents=True, exist_ok=False)
    (partial / "gs").mkdir()
    try:
        with SourceReader(source_root) as reader:
            prefix, camera_refs, gs_refs = episode_layout(reader, scene, episode)
            camera, obs = observation_timeline(reader, camera_refs)
            require(prefix == plan["prefix"] and camera_refs == plan["camera_refs"]
                    and gs_refs == plan["gs_refs"] and np.array_equal(obs, plan["obs"]),
                    "Episode layout/timeline changed after preflight")
            tactile_sources = find_raw_sources(source_root, scene, obs)
            require(tactile_sources == plan["tactile_sources"], "Raw discovery changed after preflight")
            frames = len(obs)
            require(sorted(gs_refs) == list(range(frames)), "GS must cover the full observation timeline")
            source_meta = json.loads(reader.read("episode_metadata", f"{prefix}/metadata.json").decode("utf-8"))
            require(camera in source_meta.get("cameras", []), "Observation camera absent in source metadata")
            slots = hand_slots(scene, episode)
            robot_payload = reader.read("robot", f"{prefix}/robot/robot.npy")
            t, o, original_shapes = load_robot(robot_payload, frames, slots)
            actions, original_shapes["actions"] = load_actions(robot_payload, frames, slots, t, o)
            points, controller_mask, clipped = controller_arrays(t, o, slots)
            write_npz(partial / "controller.npz", {"controller_points": points,
                "controller_valid_mask": controller_mask, "T_worlds": t, "openings": o,
                "opening_clipped": clipped, "actions": actions})
            tactile = np.zeros((frames, 4, 12, 32), dtype=np.float32)
            valid = np.zeros(tactile.shape, dtype=bool)
            index = np.full((frames, 4), -1, dtype=np.int64)
            age = np.full_like(index, -1)
            timestamps = {"observation_timestamp_us": obs}
            for channel in range(4):
                raw_path, time_path, median_path = tactile_sources[channel]
                raw = load_raw(reader.read(f"tactile_{channel}_raw", raw_path))
                raw_ts = timestamp_array(reader.read(f"tactile_{channel}_timestamps", time_path), strict=False, allow_empty=True)
                median = load_median(reader.read(f"tactile_{channel}_median", median_path))
                values, accepted, ix, ages = tactile_arrays(raw, median, raw_ts, obs)
                tactile[:, channel], valid[:, channel] = values, accepted
                index[:, channel], age[:, channel] = ix, ages
                timestamps[f"raw_tactile_timestamp_us_{channel}"] = raw_ts
            timestamps.update(tactile_raw_index=index, tactile_age_us=age)
            write_npz(partial / "tactile.npz", {"tactile": tactile, "valid_mask": valid})
            write_npz(partial / "timestamps.npz", timestamps)
            counts = []
            for frame in range(frames):
                gs = gs_arrays(reader.read(f"gs_{frame}", *gs_refs[frame]), frame, obs[frame])
                counts.append(len(gs["xyz"]))
                write_npz(partial / "gs" / f"frame_{frame:06d}.npz", gs)
                if frame % 100 == 0:
                    print(f"episode {episode}: GS {frame + 1}/{frames}", flush=True)
            for key, ref in plan["source_refs"].items():
                require(reader.refs.get(key) == ref, f"Source changed after preflight: {key}")
            output_files = sorted(partial.rglob("*.npz"))
            protocol = PROJECT / "docs" / "DATASET_PROTOCOL.md"
            metadata = {"schema_version": VERSION, "script_version": SCRIPT_VERSION,
                "scene": scene, "episode": episode, "frames": frames,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "observation_camera": camera, "observation_cameras": sorted(camera_refs),
                "source_episode_directory": prefix, "source_root": str(source_root),
                "stream_order": list(STREAMS), "hand_order": list(HAND_ORDER),
                "robot_slot_hands": slots,
                "physical_hand_to_robot_slot": {h: slots.index(h) if h in slots else None for h in HAND_ORDER},
                "binding_basis": binding_evidence(scene, episode),
                "controller_point_ids": POINT_IDS, "source_robot_shapes": original_shapes,
                "source_robot_dtypes": {"T_worlds": str(t.dtype), "openings": str(o.dtype), "actions": str(actions.dtype)},
                "parameters": parameters(), "source_files": reader.refs,
                "gs_point_counts": counts,
                "protocol_sha256": hash_file(protocol) if protocol.is_file() else None,
                "scripts_sha256": {name: hash_file(Path(__file__).with_name(name))
                                   for name in ("rebuild.py", "validate.py")},
                "output_files": {p.relative_to(partial).as_posix(): hash_file(p) for p in output_files},
                "summary": {"tactile_valid_elements": int(valid.sum()),
                            "controller_valid_points": int(controller_mask.sum()),
                            "opening_clipped_hand_frames": int(clipped.sum())}}
            with (partial / "metadata.json").open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        # Replacements must pass full source verification before touching old output.
        _validators().validate_episode(partial, expected_scene=scene, source_root=source_root, verify_sources=overwrite)
        require(not target.is_symlink(), f"Output became a symlink: {target}")
        if old_identity is None:
            require(not target.exists(), f"Output appeared during rebuild: {target}")
            partial.rename(target)
        else:
            require(target.is_dir() and (target.stat().st_dev, target.stat().st_ino) == old_identity,
                    f"Output changed during rebuild: {target}")
            backup = target.with_name(f".episode_{episode}.previous")
            require(not backup.exists() and not backup.is_symlink(), f"Backup already exists: {backup}")
            target.rename(backup)
            try:
                partial.rename(target)
            except Exception:
                backup.rename(target)
                raise
    except Exception:
        print(f"Episode {episode} failed; inspect {partial}, {target}, and "
              f"{target.with_name(f'.episode_{episode}.previous')} before retrying", file=sys.stderr)
        raise
    # Publication is committed. Cleanup must not change reconstruction success.
    print(f"Completed: {target}", flush=True)
    if old_identity is not None:
        try:
            shutil.rmtree(backup)
        except Exception as exc:
            print(f"WARNING: reconstruction succeeded at {target}, but old backup cleanup "
                  f"failed at {backup}: {exc}. Validation and default skipping remain available; "
                  "inspect the backup before another overwrite.", file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, choices=sorted(SCENE_BINDINGS), help="Full audited scene name")
    parser.add_argument("--source-root", type=Path, default=PROJECT / "datasets/deform360")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "datasets/touch4d-v1")
    parser.add_argument("--depth-only", action="store_true",
                        help="Only supplement depth and camera metadata for existing core episodes")
    parser.add_argument("--depth-root", type=Path,
                        help="Separate depth extension root; default: output-root sibling touch4d-depth-v1")
    parser.add_argument("--episodes", type=parse_episodes, metavar="0-9 or 0,5",
                        help="Optional subset; default: every discovered source episode")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild selected episodes; replace existing outputs only after full validation")
    args = parser.parse_args(argv)
    require(args.depth_only or args.depth_root is None, "--depth-root requires --depth-only")
    require(os.name != "nt", "Reconstruction outputs belong on the server; do not run on local Windows")
    if args.depth_only:
        return run_depth_only(args)
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    # Resolve dependencies before creating any directories or partial outputs.
    from plyfile import PlyData
    validate_episode = _validators().validate_episode
    available = discover_episodes(source_root / "processed" / args.scene)
    episodes = args.episodes if args.episodes is not None else list(available)
    require(set(episodes) <= set(available), "Requested episodes are absent from the source scene")
    # Lock the scene directory itself: no lock file and no concurrent skip/replace races.
    import fcntl
    scene_dir = output_root / args.scene
    for episode in episodes:
        output_paths(source_root, output_root, args.scene, episode, allow_existing=True)
    scene_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock_fd = os.open(scene_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, lock_fd)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another rebuild is using this scene: {scene_dir}") from exc
        pending, skipped = [], []
        for episode in episodes:
            target, _ = output_paths(source_root, output_root, args.scene, episode, allow_existing=True)
            if target.exists() and not args.overwrite:
                print(f"Full validation before skipping: {target}", flush=True)
                try:
                    result = validate_episode(target, expected_scene=args.scene, source_root=source_root,
                                              verify_sources=True)
                except Exception as exc:
                    raise ValueError(f"Existing episode failed full validation: {target}; "
                                     f"inspect the error or explicitly use --overwrite: {exc}") from exc
                require(result["contains_actions"],
                        f"Existing episode passed full validation but needs an upgrade: {target}; "
                        "legacy output lacks actions. Use --overwrite to upgrade.")
                skipped.append(episode)
                print(f"Skipped (full validation passed): {target}", flush=True)
            else:
                backup = target.with_name(f".episode_{episode}.previous")
                require(not backup.exists() and not backup.is_symlink(),
                        f"Inspect/remove the previous backup before another overwrite: {backup}")
                pending.append(episode)
        # Include skipped episodes in pairing checks so a recording cannot be reused.
        plans = preflight(source_root, output_root, args.scene, episodes) if pending else {}
        print(f"Source: {source_root}\nOutput: {scene_dir}\nRebuild: {pending}\nSkipped: {skipped}", flush=True)
        for episode in pending:
            rebuild_episode(source_root, output_root, args.scene, episode, plans[episode],
                            overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    # Also suppress sibling-module bytecode when the validator imports helpers.
    sys.dont_write_bytecode = True
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, ImportError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
