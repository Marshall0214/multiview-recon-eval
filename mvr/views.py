"""Initial view sets and baseline selection strategies (README §3, §4.2)."""

import math

import numpy as np
import torch


def directions(scene, ids):
    """Unit vectors from object centre to camera centres, [len(ids), 3] numpy."""
    c = scene.centers[ids].cpu().numpy()
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def elev_azim_deg(d):
    return np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1))), np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360


def fps(dirs, k, start):
    """Farthest point sampling on the sphere (angular distance), starting at index `start`."""
    picks = [start]
    dmin = np.arccos(np.clip(dirs @ dirs[start], -1, 1))
    while len(picks) < k:
        j = int(np.argmax(dmin))
        picks.append(j)
        dmin = np.minimum(dmin, np.arccos(np.clip(dirs @ dirs[j], -1, 1)))
    return picks


def initial_views(scene, kind, k=10, seed=0):
    """I-uniform: FPS over all candidates. I-biased: FPS restricted to elev > 30 deg, azimuth in [0, 180)."""
    cand = scene.ids("cand")
    d = directions(scene, cand)
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        pool = np.arange(len(cand))
    elif kind == "biased":
        e, a = elev_azim_deg(d)
        pool = np.where((e > 30) & (a < 180))[0]
    else:
        raise ValueError(kind)
    sub = fps(d[pool], k, start=int(rng.integers(len(pool))))
    return [cand[pool[i]] for i in sub]


def select_random(avail, n, rng):
    return list(rng.choice(avail, size=n, replace=False))


def select_fps(scene, train_ids, avail, n):
    """Baseline: candidates farthest (angularly) from all current training views."""
    d_av = directions(scene, avail)
    dmin = np.min(np.arccos(np.clip(d_av @ directions(scene, train_ids).T, -1, 1)), axis=1)
    picks = []
    for _ in range(n):
        j = int(np.argmax(dmin))
        picks.append(avail[j])
        dmin = np.minimum(dmin, np.arccos(np.clip(d_av @ d_av[j], -1, 1)))
    return picks


def select_by_score(scene, avail, scores, n, nms_deg=15.0):
    """Greedy top-n by score with angular non-maximum suppression."""
    d = directions(scene, avail)
    order = np.argsort(-np.asarray(scores))
    picks = []
    for j in order:
        if all(math.degrees(math.acos(np.clip(d[j] @ d[k], -1, 1))) >= nms_deg for k in picks):
            picks.append(j)
        if len(picks) == n:
            break
    for j in order:  # NMS too strict: fill up with the next best
        if len(picks) == n:
            break
        if j not in picks:
            picks.append(j)
    return [avail[j] for j in picks]
