"""Shape simulator, step 1: one S. pasteurii lineage per thumbnail, as if zoomed on it, rendered like a brightfield
microscope image. Dataset for the encoder / clustering of the PhenoME notebook (not done here).

Growth curve (user, 2026-11-12): thumbnail i of every phase is the SAME lineage i, seen at a later time. A lineage
starts as one cell, grows and divides, and its daughters stay attached: Lag = one cell, Expo = a growing / dividing
cell or a fresh doublet, End_Expo and Stationary = chains only. Each thumbnail has its ground truth in labels.csv:
it is never used for training, only to check afterwards whether the clusters find the phases.

Lineage model (time A in generations of the first cell):
  - growth: every cell lengthens as L_birth * 2^(t / 1 generation) and divides when it reaches its division length
    DIVISION_RATIO * BIRTH_LENGTH * (1 + noise) (a "sizer": lengths stay bounded). Daughters get a share SPLIT +- noise.
  - septum: in the last part of the cycle (from SEPTUM_START) the septum forms at the split point, progress p 0 -> 1.
  - after division the two daughters stay attached; the junction opens as a hinge to its own angle within HINGE_TIME.
  - CURVED_FRACTION of the rods are curved (user: 1 in 6): drawn for every new cell (arc or angle).
  - the whole lineage is simulated once; a phase is only a time A drawn in its window, so the identity is kept.
Shape model (user, 2026-11-12): a chain of circles along a centerline (heading = integral of the curvature + kinks).
Inside a cell r = R (half width); at a pole r = distance to the pole (union = exact capsule). At a septum a shallow
waist r = min(R, sqrt((R (1 - NOTCH_DEPTH p))^2 + u^2)), u = distance to the septum, and a thin cut across the rod:
thickness * (1 - p exp(-d^2 / 2 SEPTUM_UM^2)), d = axial distance to the septum (fine separation, user 2026-11-12).
Rendering: Beer-Lambert absorption, transmission = exp(-ABSORPTION * thickness), thickness = largest sphere chord
(2 sqrt(r^2 - d^2)), SUPERSAMPLE x finer grid then averaged, Gaussian PSF (+ random defocus), uneven illumination,
Gaussian noise. Fixed intensity scale and fixed pixel size (no per-object zoom): size and contrast stay comparable.

All parameters are in the PARAMETERS block below. The lengths, windows and probabilities are illustrative choices
(S. pasteurii 1.3-4.0 x 0.5-1.2 um, ~3 um observed), not measurements.

Usage (from the project root, .venv):
  python Video_Process_Script/Shape_Simulator/shape_sim.py                      # napari preview, 50 lineages
  python Video_Process_Script/Shape_Simulator/shape_sim.py --export --n 1000 --name v2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# ======================================================================================================== PARAMETERS
# --- Thumbnail and output
THUMB_PX = 224                 # thumbnail side in pixels (multiple of 14 = DINOv2 patch)
UM_PER_PX = 0.06               # pixel size (x100 estimate, 50 px for a 3 um cell): 224 px = 13.4 um
MARGIN_UM = 0.4                # free border kept around the object; a lineage that does not fit is redrawn
MAX_TRIES = 50                 # redraws of a whole lineage before giving up
SEED = 0                       # same seed = same dataset
OUTPUT_DIR = "Processing_Results/Shape_Simulator"   # export goes to OUTPUT_DIR/<--name>/
PREVIEW_N = 50                 # lineages shown by the napari preview (default of --n)

# --- Phases: time windows in generations of the first cell (born at A = 0, divides at A ~ 1).
# after_division = True: the window starts no earlier than the first division + AFTER_DIVISION_DELAY (chains only)
PHASES = {
    "Lag": {"window": (0.0, 0.5), "after_division": False},
    "Expo": {"window": (0.55, 1.25), "after_division": False},
    "End_Expo": {"window": (1.3, 2.0), "after_division": True},
    "Stationary": {"window": (2.05, 2.6), "after_division": True},
}
AFTER_DIVISION_DELAY = 0.05    # generations

# --- Cell size and growth
BIRTH_LENGTH = 2.0             # um, length of the first newborn cell
BIRTH_LENGTH_CV = 0.05         # relative std of the first cell's birth length
DIVISION_RATIO = 2.0           # a cell divides at DIVISION_RATIO * BIRTH_LENGTH (2 -> 2-4 um, mean ~3 um)
DIVISION_LENGTH_CV = 0.05      # relative std of the division length
MIN_GROWTH_RATIO = 1.05        # a cell grows at least by this factor before dividing (guard for short cycles)
SPLIT = 0.5                    # share of the mother's length given to the first daughter
SPLIT_STD = 0.025              # std of that share (asymmetric divisions)
WIDTH_UM = 1.0                 # mean cell width (published 0.5-1.2, measured ~1.1), per lineage, inherited
WIDTH_STD_UM = 0.08
MIN_WIDTH_UM = 0.5

# --- Septum and chain junctions
SEPTUM_START = 0.7             # fraction of the cell cycle when the septum starts to form
NOTCH_DEPTH = 0.3              # waist at a finished septum: half width R (1 - NOTCH_DEPTH)
SEPTUM_UM = 0.04               # half thickness (std) of the septum cut: the visible separation line
HINGE_TIME = 0.3               # generations for a junction to open to its angle after division
JOINT_ANGLE_STD_DEG = 25.0     # std of the final angle of a junction

# --- Curved rods
CURVED_FRACTION = 1 / 6        # 1 rod in 6 is curved (user)
CURVED_TYPES = {"arc": 0.5, "angle": 0.5}   # probabilities of the curved shapes
ARC_ANGLE_DEG = (30.0, 150.0)  # total bend of an arc-shaped cell (uniform range, random sign)
KINK_ANGLE_DEG = (30.0, 110.0) # bend of an angle-shaped cell (uniform range, random sign)
KINK_POSITION = (0.3, 0.7)     # position of the kink along the cell (fraction of its length)

# --- Circle chain
CIRCLE_STEP_UM = 0.05          # spacing of the circle centers on the centerline (edge ripple ~ step^2 / 8R)

# --- Microscope rendering
SUPERSAMPLE = 2                # the thickness is computed on a SUPERSAMPLE x finer grid, then averaged
ABSORPTION = 0.5               # 1/um: a 1 um thick rod transmits exp(-0.5) = 61 %
PSF_SIGMA_UM = 0.12            # in-focus optical blur (Gaussian approximation of the PSF)
DEFOCUS_UM = (0.0, 0.15)       # extra blur, uniform range, per thumbnail
BACKGROUND = 0.8               # mean illumination level (0-1, saved as 0-255)
BACKGROUND_STD = 0.03          # per thumbnail
ILLUM_GRADIENT = 0.04          # max relative illumination slope across the thumbnail
NOISE_STD = 0.015              # Gaussian camera noise
# ==================================================================================================================


def _pick(rng, probs):
    keys = list(probs)
    p = np.array([probs[k] for k in keys], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _new_cell(rng, birth, length):
    """A cell of the lineage tree: birth time, birth length, division time, curvature, junction of its division."""
    curve = {"type": "straight"}
    if rng.uniform() < CURVED_FRACTION:
        kind = _pick(rng, CURVED_TYPES)
        if kind == "arc":
            curve = {"type": "arc", "bend_deg": rng.uniform(*ARC_ANGLE_DEG) * rng.choice([-1, 1])}
        else:
            curve = {"type": "angle", "kink_deg": rng.uniform(*KINK_ANGLE_DEG) * rng.choice([-1, 1]),
                     "kink_at": rng.uniform(*KINK_POSITION)}
    div_length = DIVISION_RATIO * BIRTH_LENGTH * (1 + DIVISION_LENGTH_CV * rng.standard_normal())
    division = birth + np.log2(max(div_length / length, MIN_GROWTH_RATIO))
    return {"birth": birth, "length": length, "division": division, "curve": curve,
            "joint_deg": rng.normal(0, JOINT_ANGLE_STD_DEG), "split": SPLIT + SPLIT_STD * rng.standard_normal(),
            "children": None}


def grow_lineage(rng, until):
    """Lineage tree simulated once up to time `until` (generations); the daughters are kept in chain order."""
    root = _new_cell(rng, 0.0, BIRTH_LENGTH * (1 + BIRTH_LENGTH_CV * rng.standard_normal()))
    todo = [root]
    while todo:
        c = todo.pop()
        if c["division"] < until:
            L = c["length"] * 2 ** (c["division"] - c["birth"])
            c["children"] = [_new_cell(rng, c["division"], L * c["split"]),
                             _new_cell(rng, c["division"], L * (1 - c["split"]))]
            todo += c["children"]
    return root


def snapshot(cell, A):
    """Rods and junctions of the lineage at time A. A rod = a visible segment; a cell with a septum in progress is
    two rods joined by a junction of progress p < 1."""
    if cell["children"] is not None and A >= cell["division"]:
        left, right = (snapshot(ch, A) for ch in cell["children"])
        opened = min(1.0, (A - cell["division"]) / HINGE_TIME)
        joint = {"progress": 1.0, "angle_deg": cell["joint_deg"] * opened}
        return {"rods": left["rods"] + right["rods"], "junctions": left["junctions"] + [joint] + right["junctions"],
                "n_cells": left["n_cells"] + right["n_cells"], "septa": left["septa"] + right["septa"]}
    age = (A - cell["birth"]) / (cell["division"] - cell["birth"])
    L = cell["length"] * 2 ** (A - cell["birth"])
    p = float(np.clip((age - SEPTUM_START) / (1 - SEPTUM_START), 0, 1))
    curve = cell["curve"]
    if p == 0:
        return {"rods": [{"length": L, "curve": curve}], "junctions": [], "n_cells": 1, "septa": []}
    # septum forming: two halves sharing the mother's curvature
    # (an arc bends by half on each side, a kink stays where it was, in the half that holds it)
    sp = cell["split"]
    halves = [{"length": L * sp, "curve": {"type": "straight"}}, {"length": L * (1 - sp), "curve": {"type": "straight"}}]
    if curve["type"] == "arc":
        for h, frac in zip(halves, (sp, 1 - sp)):
            h["curve"] = {"type": "arc", "bend_deg": curve["bend_deg"] * frac}
    elif curve["type"] == "angle":
        k = curve["kink_at"]
        halves[0 if k < sp else 1]["curve"] = {"type": "angle", "kink_deg": curve["kink_deg"],
                                               "kink_at": k / sp if k < sp else (k - sp) / (1 - sp)}
    return {"rods": halves, "junctions": [{"progress": p, "angle_deg": 0.0}], "n_cells": 1, "septa": [p]}


def build_circles(snap, R, orientation_deg):
    """Circle chain of a snapshot: centers (N, 2) um, radii, arc length s, unit tangents, septa (s, p), extent."""
    rods, junctions = snap["rods"], snap["junctions"]
    total = sum(r["length"] for r in rods)
    s = np.linspace(0, total, max(2, int(np.ceil(total / CIRCLE_STEP_UM)) + 1))
    ends = np.cumsum([r["length"] for r in rods])
    starts = ends - [r["length"] for r in rods]

    r = np.minimum(R, np.minimum(s, total - s))  # outer poles
    for b, jn in zip(ends[:-1], junctions):
        r = np.minimum(r, np.sqrt((R * (1 - NOTCH_DEPTH * jn["progress"])) ** 2 + (s - b) ** 2))

    kappa = np.zeros_like(s)
    theta = np.zeros_like(s)
    for a, b, rod in zip(starts, ends, rods):
        c = rod["curve"]
        inside = (s >= a) & (s < b)
        if c["type"] == "arc":
            kappa[inside] = np.deg2rad(c["bend_deg"]) / rod["length"]
        elif c["type"] == "angle":
            theta += np.deg2rad(c["kink_deg"]) * (s >= a + c["kink_at"] * rod["length"])
    theta += np.concatenate([[0], np.cumsum((kappa[1:] + kappa[:-1]) / 2 * np.diff(s))])
    for b, jn in zip(ends[:-1], junctions):
        theta += np.deg2rad(jn["angle_deg"]) * (s >= b)
    theta += np.deg2rad(orientation_deg)

    step = np.diff(s, prepend=0)
    tangent = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    xy = np.cumsum(tangent * step[:, None], axis=0)
    keep = r > 0
    xy, r, s, tangent = xy[keep], r[keep], s[keep], tangent[keep]
    lo, hi = (xy - r[:, None]).min(0), (xy + r[:, None]).max(0)
    septa = [(b, jn["progress"]) for b, jn in zip(ends[:-1], junctions)]
    return xy - (lo + hi) / 2, r, s, tangent, septa, hi - lo


def render(xy, r, s, tangent, septa, rng):
    """Brightfield-like image (float, 0-1) and the optical parameters drawn for it."""
    n, ss = THUMB_PX, SUPERSAMPLE
    px = UM_PER_PX / ss
    m = n * ss
    thick = np.zeros((m, m))
    axial = np.zeros((m, m))  # arc length of each pixel, from the circle that gives its thickness
    for (cx, cy), rad, si, (tx, ty) in zip(xy, r, s, tangent):
        i0, i1 = max(int(np.floor((cy - rad) / px + m / 2)), 0), min(int(np.ceil((cy + rad) / px + m / 2)) + 1, m)
        j0, j1 = max(int(np.floor((cx - rad) / px + m / 2)), 0), min(int(np.ceil((cx + rad) / px + m / 2)) + 1, m)
        yy = (np.arange(i0, i1) + 0.5 - m / 2) * px - cy
        xx = (np.arange(j0, j1) + 0.5 - m / 2) * px - cx
        chord = 2 * np.sqrt(np.clip(rad ** 2 - yy[:, None] ** 2 - xx[None, :] ** 2, 0, None))
        win = thick[i0:i1, j0:j1]
        better = chord > win
        win[better] = chord[better]
        axial[i0:i1, j0:j1][better] = (si + xx[None, :] * tx + yy[:, None] * ty)[better]
    for sj, p in septa:
        thick *= 1 - p * np.exp(-(axial - sj) ** 2 / (2 * SEPTUM_UM ** 2))
    trans = np.exp(-ABSORPTION * thick).reshape(n, ss, n, ss).mean(axis=(1, 3))

    defocus = rng.uniform(*DEFOCUS_UM)
    blur_um = np.hypot(PSF_SIGMA_UM, defocus)
    trans = gaussian_filter(trans, blur_um / UM_PER_PX, mode="nearest")
    bg = rng.normal(BACKGROUND, BACKGROUND_STD)
    ang = rng.uniform(0, 2 * np.pi)
    g = np.linspace(-1, 1, n)
    illum = bg * (1 + ILLUM_GRADIENT * rng.uniform(0, 1) * (np.cos(ang) * g[None, :] + np.sin(ang) * g[:, None]))
    img = illum * trans + NOISE_STD * rng.standard_normal((n, n))
    return np.clip(img, 0, 1), {"blur_um": blur_um, "defocus_um": defocus, "background": bg}


def _state(snap):
    if snap["n_cells"] > 1:
        return "chain"
    return "dividing" if snap["septa"] else "single"


def simulate_lineage(i):
    """Lineage i in every phase: {phase: (image, (xy, r) in um, label row)}. The lineage is redrawn as a whole if
    one of its phases does not fit in the thumbnail, so that the identity is kept across phases."""
    rng = np.random.default_rng([SEED, i])
    half = THUMB_PX * UM_PER_PX / 2 - MARGIN_UM
    last = max(ph["window"][1] for ph in PHASES.values())
    for _ in range(MAX_TRIES):
        root = grow_lineage(rng, last)
        R = max(MIN_WIDTH_UM, rng.normal(WIDTH_UM, WIDTH_STD_UM)) / 2
        orientation = rng.uniform(0, 360)
        times, snaps, geoms = {}, {}, {}
        for phase, ph in PHASES.items():
            lo, hi = ph["window"]
            if ph["after_division"]:
                lo = max(lo, root["division"] + AFTER_DIVISION_DELAY)
            times[phase] = rng.uniform(lo, max(lo, hi))
            snaps[phase] = snapshot(root, times[phase])
            geoms[phase] = build_circles(snaps[phase], R, orientation)
        if all(np.all(g[5] / 2 <= half) for g in geoms.values()):
            break
    else:
        raise RuntimeError(f"lineage {i}: does not fit in the thumbnail after {MAX_TRIES} tries")

    out = {}
    for k, phase in enumerate(PHASES):
        xy, r, s, tangent, septa, _ = geoms[phase]
        snap = snaps[phase]
        img, optics = render(xy, r, s, tangent, septa, np.random.default_rng([SEED, i, k]))
        rods = snap["rods"]
        row = {"lineage": i, "phase": phase, "time_gen": round(times[phase], 3), "state": _state(snap),
               "n_cells": snap["n_cells"], "n_septa_forming": len(snap["septa"]),
               "septum_progress": ";".join(f"{p:.2f}" for p in snap["septa"]),
               "total_length_um": round(sum(rd["length"] for rd in rods), 3),
               "rod_lengths_um": ";".join(f"{rd['length']:.3f}" for rd in rods),
               "rod_shapes": ";".join(rd["curve"]["type"] for rd in rods),
               "joint_angles_deg": ";".join(f"{j['angle_deg']:.1f}" for j in snap["junctions"]
                                            if j["progress"] == 1),
               "width_um": round(2 * R, 3), "orientation_deg": round(orientation, 1),
               **{k2: round(v, 4) for k2, v in optics.items()}}
        out[phase] = (img, (xy, r), row)
    return out


def export(n, name):
    from PIL import Image

    out = Path(OUTPUT_DIR) / name
    if out.exists():
        raise FileExistsError(f"{out} already exists: pick another --name")
    for phase in PHASES:
        (out / phase).mkdir(parents=True)
    rows = []
    for i in range(n):
        for phase, (img, _, row) in simulate_lineage(i).items():
            fname = f"{phase}/{phase}_{i:05d}.png"
            Image.fromarray((img * 255).round().astype(np.uint8)).save(out / fname)
            rows.append({"file": fname, **row})
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)
    params = {k: v for k, v in globals().items() if k.isupper()}
    (out / "parameters.txt").write_text("\n".join(f"{k} = {v!r}" for k, v in params.items()) + "\n")
    print(f"{n} lineages x {len(PHASES)} phases -> {out.resolve()}")


def preview(n):
    """napari: one stack per phase (slider = lineage, the same in every phase), circle chain as points, label on top."""
    import napari

    viewer = napari.Viewer(title="Shape simulator")
    lineages = [simulate_lineage(i) for i in range(n)]
    rows = {phase: [lin[phase][2] for lin in lineages] for phase in PHASES}
    c = THUMB_PX / 2
    for phase in PHASES:
        pts, sizes = [], []
        for i, lin in enumerate(lineages):
            xy, r = lin[phase][1]
            pts.append(np.column_stack([np.full(len(r), i), xy[:, ::-1] / UM_PER_PX + c - 0.5]))
            sizes.append(2 * r / UM_PER_PX)
        viewer.add_image(np.stack([lin[phase][0] for lin in lineages]), name=phase, colormap="gray",
                         contrast_limits=(0, 1), visible=False)
        viewer.add_points(np.concatenate(pts), size=np.concatenate(sizes), name=f"{phase} circles",
                          face_color="transparent", border_color="orange", border_width=0.02, visible=False)
    viewer.layers["Expo"].visible = True

    def show_label(*_):
        visible = [l.name for l in viewer.layers if l.visible and l.name in rows]
        if not visible:
            return
        row = rows[visible[-1]][int(viewer.dims.current_step[0])]
        viewer.text_overlay.text = "\n".join(f"{k}: {v}" for k, v in row.items())

    viewer.text_overlay.visible = True
    viewer.text_overlay.font_size = 9
    viewer.dims.events.current_step.connect(show_label)
    for layer in viewer.layers:
        layer.events.visible.connect(show_label)
    show_label()
    napari.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="write PNG thumbnails + labels.csv instead of previewing")
    ap.add_argument("--n", type=int, default=PREVIEW_N, help="number of lineages (= thumbnails per phase)")
    ap.add_argument("--name", default="v2", help="run folder in OUTPUT_DIR (export)")
    a = ap.parse_args()
    if a.export:
        export(a.n, a.name)
    else:
        preview(a.n)
