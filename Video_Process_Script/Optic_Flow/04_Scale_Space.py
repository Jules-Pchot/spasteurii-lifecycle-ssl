"""Optic flow track, step 4: scale-space decomposition of one reference frame, to tell in-focus from out-of-focus.

Input: frame - background of step 2 (rolling masked mean, % of brightness, signed: dark < 0), video VIDEO,
segment SEGMENT, frame FRAME of the segment (local index). Expo3: segment 1 frame 273 (global 688). 0003
(11:18-13:00 of 0003.mp4, not stabilized, 2026-11-10): a second video to check that the rolling mean of step 2
does not erase most out-of-focus cells (it does on Expo3).
Observation behind the step: after step 2 an in-focus bacterium is dark inside with a bright halo on its edges,
an out-of-focus one has a dark membrane (edge) and a brighter inside.

Decomposition (per pixel, no object, no threshold on the image):
  - masked Gaussian blur G_s = G(I * M) / G(M), M = pixels with data (the stabilization border never leaks in)
  - bands B_k = G_sk - G_sk+1 (difference of two consecutive scales = what is added between them), divided by
    (r - 1), r the ratio between consecutive sigmas, so that the two grids below are on the same scale
    (DoG(s, r s) ~ (r - 1) s^2 LoG). Band k is placed at the geometric mean of its two sigmas.
    Sign: B < 0 = darker than its surroundings at that scale, B > 0 = brighter.
  - two grids compared: octave (1, 2, ..., 32 -> 5 bands) and half octave (1, 1.41, ..., 32 -> 10 bands).
    First run up to 8 (2026-11-10); extended to 32 (user, 2026-11-11) for clumps and large blurred cells, whose
    peak fell on the last band
Per grid, maps from the curve B(sigma) of every pixel:
  - sigma_peak: band of the largest |B|, and sigma_peak_interp refined by a parabola on log2 sigma
  - amplitude: signed B at the peak (dark or bright structure)
  - sigma_sign_change: first scale where B changes sign, both sides being significant (expected cue for an
    out-of-focus cell: bright inside at fine scale, dark object at coarse scale)
  - significance: |B at peak| / noise std of that band (MAD over the frame). The sigma maps are NaN where it is
    below SIGNIFICANCE_K (background noise has a peak too, at a random scale)
Sharpening (chosen: band reweighting, 2026-11-10): I' = w0 (I - G_1) + sum_k w_k B_k (raw) + G_32, on the half
octave grid; all weights 1 give I back exactly. The weights are napari sliders, to raise the contrast between
the two classes once the curves show in which bands they differ.
Heightmap: a second napari viewer shows I' as a surface, altitude = -I' (dark bacteria are hills), colored by
the half-octave sigma_peak_interp.
Filters are isotropic (2026-11-10): the anisotropic / Hessian version is kept for later, as a comparison.
LoG and Hessian (added 2026-11-11, user, in test), on the LOG_GRID sigmas, Gaussian derivatives of the frame
(pixels without data set to 0 = background level), scale-normalized by sigma^2:
  - LoG = sigma^2 (Hxx + Hyy): blobs = local extrema in (sigma, y, x) (3 x 3 x 3), |LoG| >= SIGNIFICANCE_K * noise
    std of that scale. LoG > 0 = dark blob (opposite sign to B), radius ~ sqrt(2) sigma. Object-level decision
    (one point per structure) instead of per pixel
  - Hessian eigenvalues of sigma^2 H, lambda1 = largest |.|, lambda2 = the other one; per pixel the sigma of
    the largest sigma^2 |lambda1| (significance computed after, at that sigma). Maps: sigma, lambda1 (> 0 = dark), lambda2, anisotropy 1 - |lambda2| / |lambda1|
    (0 = round, 1 = rod / ridge), long-axis orientation (degrees from the x axis, clockwise on screen since y
    points down), significance |lambda1| / noise std of that scale

Outputs (external drive, never overwritten): per grid, the bands stack and the maps stack (TIFF, names in the
metadata). If they exist they are loaded, not recomputed.
napari: shift + click on a pixel adds its curve B(sigma) to the dock panel (both grids), with the histograms of
sigma_peak for dark and bright peaks below.

Run with: python 04_Scale_Space.py
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = Path("/Volumes/Extreme SSD/TIFFIle/Optic_Flow")
VIDEOS = {  # same as step 2; stabilized = None when step 1 was skipped (every pixel has data)
    "Expo3": {"stabilized": DATA_DIR / "Expo3_1_stabilized.tif", "segments": [(0, 386), (415, 962)],
              "segment": 1, "frame": 273},  # global 688
    "0003": {"stabilized": None, "segments": [(0, 291)], "segment": 0, "frame": 207},  # chosen by the user
}
VIDEO = "0003"
STABILIZED_TIFF = VIDEOS[VIDEO]["stabilized"]
SEGMENTS = VIDEOS[VIDEO]["segments"]
SEGMENT = VIDEOS[VIDEO]["segment"]  # index in SEGMENTS
FRAME = VIDEOS[VIDEO]["frame"]      # reference frame, local index in the segment
GRIDS = {"octave": [2 ** k for k in range(6)],             # 1, 2, 4, 8, 16, 32
         "half_octave": [2 ** (k / 2) for k in range(11)]}  # 1, 1.41, 2, ..., 22.6, 32
SHARPEN_GRID = "half_octave"       # grid whose bands get the sharpening weights
SIGNIFICANCE_K = 3.0               # sigma maps kept where |B at peak| > this * band noise std
HEIGHTMAP_STEP = 2                 # heightmap drawn on every 2nd pixel (1080p surface is heavy)
HEIGHT_SCALE = 4.0                 # heightmap altitude, px per % of brightness
MAX_CURVES = 8                     # clicked pixels kept in the curve panel

LOG_MAX = float(np.log2(max(max(s) for s in GRIDS.values())))  # display range of log2 sigma: 0 ... LOG_MAX
SIGMA_LIMITS = (-1, LOG_MAX)       # contrast limits of the sigma maps (-1 = not significant, black)

PANEL = "#262930"                  # napari dark theme background
CURVE_COLORS = ["#d95926", "#3fa9f5", "#7ac943", "#f5c13f", "#c86fd9", "#f0efec", "#2ec4b6", "#ff6b9a"]
MAP_NAMES = ["sigma_peak", "sigma_peak_interp", "amplitude", "sigma_sign_change", "significance"]
LOG_GRID = "half_octave"           # sigmas of the LoG blobs and of the Hessian (user, 2026-11-11)
HESSIAN_MAP_NAMES = ["sigma", "lambda1", "lambda2", "anisotropy", "orientation", "significance"]
AXIS_STEP = 6                      # Hessian long axes drawn every AXIS_STEP px
AXIS_MIN_ANISOTROPY = 0.5          # ... where the structure is elongated enough
BLOB_COLORS = {"dark": "#f5c13f", "bright": "#c86fd9"}  # not the class colors of the notebook


def global_frame() -> int:
    return SEGMENTS[SEGMENT][0] + FRAME


def output_paths(grid: str) -> dict:
    start, end = SEGMENTS[SEGMENT]
    sigmas = GRIDS[grid]  # the sigma range is in the name: a grid change never loads old files
    name = f"{VIDEO}_4_seg{SEGMENT}_{start}-{end}_frame{global_frame()}_{grid}_{sigmas[0]:g}-{sigmas[-1]:g}"
    return {"bands": DATA_DIR / f"{name}_bands.tif", "maps": DATA_DIR / f"{name}_maps.tif"}


def lazy(path: Path):
    """Stack read on demand (only the frames used are decoded)."""
    import zarr
    return zarr.open(tifffile.imread(path, aszarr=True), mode="r")


def noise_std(values: np.ndarray) -> float:
    """Robust std of the background noise (MAD * 1.4826, insensitive to the objects)."""
    sample = values.ravel()[::7]
    return float(1.4826 * np.median(np.abs(sample - np.median(sample))))


def masked_blur(image: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur over the pixels with data only (normalized convolution)."""
    weight = valid.astype(np.float32)
    num = ndimage.gaussian_filter(image * weight, sigma, mode="nearest")
    den = ndimage.gaussian_filter(weight, sigma, mode="nearest")
    return np.where(valid, num / np.maximum(den, 1e-3), 0).astype(np.float32)


def load_frame() -> tuple[np.ndarray, np.ndarray]:
    start, end = SEGMENTS[SEGMENT]
    diff = np.asarray(lazy(DATA_DIR / f"{VIDEO}_2_seg{SEGMENT}_{start}-{end}_diff_rolling.tif")[FRAME])
    if STABILIZED_TIFF is None:
        return diff.astype(np.float32), np.ones(diff.shape, dtype=bool)
    valid = np.asarray(lazy(STABILIZED_TIFF)[global_frame()]) > 0
    return diff.astype(np.float32), valid


def bands_of(blurs: dict, sigmas: list) -> tuple[np.ndarray, np.ndarray, float]:
    """Raw bands G_sk - G_sk+1 (K-1, H, W), band centers, sigma ratio r."""
    raw = np.stack([blurs[a] - blurs[b] for a, b in zip(sigmas[:-1], sigmas[1:])])
    centers = np.sqrt(np.array(sigmas[:-1]) * np.array(sigmas[1:]))
    return raw, centers, sigmas[1] / sigmas[0]


def scale_maps(bands: np.ndarray, centers: np.ndarray, ratio: float, valid: np.ndarray) -> np.ndarray:
    """Maps (MAP_NAMES order) from the curve B(sigma) of every pixel. bands are already divided by (r - 1)."""
    noise = np.array([noise_std(b[valid]) for b in bands])
    log_c, step = np.log2(centers), np.log2(ratio)
    strength = np.abs(bands)
    k = np.argmax(strength, axis=0)
    amplitude = np.take_along_axis(bands, k[None], axis=0)[0]
    significance = np.abs(amplitude) / noise[k]

    # parabola on |B| through the peak and its two neighbors (log2 sigma axis), only for interior peaks
    kk = np.clip(k, 1, len(bands) - 2)
    a, b, c = (np.take_along_axis(strength, (kk + d)[None], axis=0)[0] for d in (-1, 0, 1))
    curvature = a - 2 * b + c
    offset = np.where(curvature < 0, 0.5 * (a - c) / np.where(curvature < 0, curvature, -1), 0)
    offset = np.where((k == kk), np.clip(offset, -0.5, 0.5), 0)
    sigma_interp = 2 ** (log_c[k] + offset * step)

    # first sign change of B along sigma with a significant band on each side of it
    z = strength / noise[:, None, None]
    left = np.maximum.accumulate(z, axis=0)[:-1] >= SIGNIFICANCE_K
    right = np.maximum.accumulate(z[::-1], axis=0)[::-1][1:] >= SIGNIFICANCE_K
    change = (np.sign(bands[:-1]) * np.sign(bands[1:]) < 0) & left & right
    first = np.argmax(change, axis=0)
    b0 = np.take_along_axis(bands, first[None], axis=0)[0]
    b1 = np.take_along_axis(bands, (first + 1)[None], axis=0)[0]
    t = np.where(change.any(axis=0), b0 / np.where(b0 != b1, b0 - b1, 1), 0)
    sigma_change = np.where(change.any(axis=0), 2 ** (log_c[first] + t * step), np.nan)

    keep = valid & (significance >= SIGNIFICANCE_K)
    maps = np.stack([centers[k], sigma_interp, amplitude, sigma_change, significance]).astype(np.float32)
    maps[:4, ~keep] = np.nan
    maps[4, ~valid] = np.nan
    print(f"    band noise std (%): {', '.join(f'{n:.3f}' for n in noise)}; "
          f"{keep.sum() / valid.sum():.1%} of the pixels significant, "
          f"{np.isfinite(sigma_change).sum() / max(keep.sum(), 1):.1%} of them with a sign change")
    return maps


def compute(diff: np.ndarray, valid: np.ndarray) -> dict:
    """Bands and maps of every grid, loaded from disk when they exist, else computed and saved."""
    sigmas_all = sorted({s for sigmas in GRIDS.values() for s in sigmas})
    blurs = {s: masked_blur(diff, valid, s) for s in sigmas_all}
    results = {}
    for grid, sigmas in GRIDS.items():
        raw, centers, ratio = bands_of(blurs, sigmas)
        paths = output_paths(grid)
        print(f"  {grid}: sigmas {', '.join(f'{s:.2f}' for s in sigmas)}")
        if all(p.exists() for p in paths.values()):
            bands, maps = tifffile.imread(paths["bands"]), tifffile.imread(paths["maps"])
            print(f"    loaded: {paths['bands'].name}, {paths['maps'].name}")
        elif any(p.exists() for p in paths.values()):
            raise FileExistsError(f"not overwritten (delete all the outputs of this grid to recompute): {paths}")
        else:
            bands = (raw / (ratio - 1)).astype(np.float32)
            maps = scale_maps(bands, centers, ratio, valid)
            meta = {"step": "4 scale-space bands of frame - background", "segment": list(SEGMENTS[SEGMENT]),
                    "frame": global_frame(), "sigmas": list(sigmas), "band_centers": centers.tolist(),
                    "normalization": f"bands / (r - 1), r = {ratio:.4f}", "significance_k": SIGNIFICANCE_K}
            tifffile.imwrite(paths["bands"], bands, compression="zlib", metadata=meta)
            tifffile.imwrite(paths["maps"], maps, compression="zlib", metadata={**meta, "maps": MAP_NAMES})
            for p in paths.values():
                print(f"    saved: {p}")
        results[grid] = {"sigmas": sigmas, "centers": centers, "bands": bands, "raw": raw,
                         "maps": dict(zip(MAP_NAMES, maps))}
    results["fine"] = diff - blurs[GRIDS[SHARPEN_GRID][0]]      # finer than the first sigma
    results["coarse"] = blurs[GRIDS[SHARPEN_GRID][-1]]          # coarser than the last sigma
    return results


def sharpen(results: dict, weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """I' = w0 (I - G_first) + sum_k w_k B_k + G_last (all weights 1 -> I)."""
    raw = results[SHARPEN_GRID]["raw"]
    image = weights[0] * results["fine"] + np.tensordot(weights[1:], raw, axes=1) + results["coarse"]
    return np.where(valid, image, 0).astype(np.float32)


def log_sigma_display(sigma: np.ndarray) -> np.ndarray:
    """log2 sigma for display (0 = 1 px ... LOG_MAX), -1 where not significant (drawn black)."""
    return np.where(np.isfinite(sigma), np.log2(np.where(np.isfinite(sigma), sigma, 1)), -1).astype(np.float32)


def sigma_colormap():
    """turbo from log2 sigma 0 to LOG_MAX, black below (non-significant pixels), for SIGMA_LIMITS."""
    from matplotlib import colormaps
    from napari.utils import Colormap
    start = 1 / (LOG_MAX + 1)  # position of log2 sigma = 0 between the limits
    controls = np.concatenate([[0, start - 0.01], np.linspace(start, 1, 64)])
    colors = np.concatenate([[[0, 0, 0, 1], [0, 0, 0, 1]], colormaps["turbo"](np.linspace(0, 1, 64))])
    return Colormap(colors=colors, controls=controls, name="sigma_turbo")


def log_hessian_paths() -> dict:
    base = output_paths(LOG_GRID)["bands"].name.removesuffix("_bands.tif")
    return {"log": DATA_DIR / f"{base}_log.tif", "hessian": DATA_DIR / f"{base}_hessian_maps.tif",
            "blobs": DATA_DIR / f"{base}_log_blobs.csv"}


def hessian_at(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale-normalized Hessian sigma^2 (Hyy, Hxx, Hxy) by Gaussian derivatives."""
    s2 = sigma ** 2
    return tuple((s2 * ndimage.gaussian_filter(image, sigma, order=o, mode="nearest")).astype(np.float32)
                 for o in ((2, 0), (0, 2), (1, 1)))


def eigen(hyy: np.ndarray, hxx: np.ndarray, hxy: np.ndarray):
    """lambda1 (largest |.|), lambda2, long-axis angle (degrees from x, -90 ... 90) of a 2 x 2 symmetric matrix."""
    mean = (hxx + hyy) / 2
    radius = np.sqrt(((hxx - hyy) / 2) ** 2 + hxy ** 2)
    plus, minus = mean + radius, mean - radius
    plus_first = np.abs(plus) >= np.abs(minus)
    lambda1, lambda2 = np.where(plus_first, plus, minus), np.where(plus_first, minus, plus)
    theta = 0.5 * np.arctan2(2 * hxy, hxx - hyy)            # eigenvector of the larger eigenvalue (plus)
    across = np.where(plus_first, theta, theta + np.pi / 2)  # direction of the strongest curvature
    axis = np.degrees(across + np.pi / 2)                    # long axis = perpendicular to it
    return lambda1, lambda2, (axis + 90) % 180 - 90


def blobs_of(log: np.ndarray, sigmas: list, valid: np.ndarray):
    """Local extrema of the LoG stack in (sigma, y, x), significant at SIGNIFICANCE_K."""
    import pandas as pd
    noise = np.array([noise_std(l[valid]) for l in log])
    z = log / noise[:, None, None]
    rows = []
    for polarity, sign in (("dark", 1), ("bright", -1)):
        zz = sign * z
        peak = (zz == ndimage.maximum_filter(zz, size=3, mode="nearest")) & (zz >= SIGNIFICANCE_K) & valid[None]
        k, y, x = np.nonzero(peak)
        edge = np.where(k == 0, "first", np.where(k == len(sigmas) - 1, "last", ""))
        rows.append(pd.DataFrame({"y": y, "x": x, "sigma": np.array(sigmas)[k], "radius": np.sqrt(2) * np.array(sigmas)[k],
                                  "polarity": polarity, "log": log[k, y, x], "significance": zz[k, y, x],
                                  "grid_edge": edge}))
    blobs = pd.concat(rows, ignore_index=True).sort_values("significance", ascending=False, ignore_index=True)
    print(f"    LoG noise std: {', '.join(f'{n:.3f}' for n in noise)}; {len(blobs)} blobs "
          f"({(blobs['polarity'] == 'dark').sum()} dark, {(blobs['polarity'] == 'bright').sum()} bright)")
    return blobs


def compute_log_hessian(diff: np.ndarray, valid: np.ndarray) -> dict:
    """LoG stack, LoG blobs and Hessian maps on LOG_GRID, loaded from disk when they exist."""
    import pandas as pd
    sigmas = GRIDS[LOG_GRID]
    paths = log_hessian_paths()
    if all(p.exists() for p in paths.values()):
        log, maps = tifffile.imread(paths["log"]), tifffile.imread(paths["hessian"])
        blobs = pd.read_csv(paths["blobs"], keep_default_na=False)
        print(f"  LoG / Hessian loaded: {paths['log'].name}, {paths['hessian'].name}, {paths['blobs'].name}")
    elif any(p.exists() for p in paths.values()):
        raise FileExistsError(f"not overwritten (delete all the LoG / Hessian outputs to recompute): {paths}")
    else:
        image = np.where(valid, diff, 0).astype(np.float32)
        log = np.empty((len(sigmas),) + image.shape, np.float32)
        best = {n: np.zeros(image.shape, np.float32) for n in HESSIAN_MAP_NAMES}
        strongest = np.full(image.shape, -1, np.float32)
        for k, sigma in enumerate(sigmas):
            hyy, hxx, hxy = hessian_at(image, sigma)
            log[k] = hxx + hyy
            lambda1, lambda2, axis = eigen(hyy, hxx, hxy)
            z = np.abs(lambda1) / noise_std(lambda1[valid])
            better = np.abs(lambda1) > strongest  # scale selection on sigma^2 |lambda1|, as argmax |B| for the bands
            strongest = np.where(better, np.abs(lambda1), strongest)
            for name, value in (("sigma", sigma), ("lambda1", lambda1), ("lambda2", lambda2),
                                ("anisotropy", 1 - np.abs(lambda2) / np.maximum(np.abs(lambda1), 1e-9)),
                                ("orientation", axis), ("significance", z)):
                best[name] = np.where(better, value, best[name]).astype(np.float32)
            print(f"    sigma {sigma:.3g} done")
        maps = np.stack([best[n] for n in HESSIAN_MAP_NAMES])
        keep = valid & (best["significance"] >= SIGNIFICANCE_K)
        maps[:5, ~keep] = np.nan
        maps[5, ~valid] = np.nan
        blobs = blobs_of(log, sigmas, valid)
        meta = {"step": "4 LoG and Hessian of frame - background", "segment": list(SEGMENTS[SEGMENT]),
                "frame": global_frame(), "sigmas": list(sigmas), "normalization": "sigma^2",
                "significance_k": SIGNIFICANCE_K}
        tifffile.imwrite(paths["log"], log, compression="zlib", metadata=meta)
        tifffile.imwrite(paths["hessian"], maps, compression="zlib", metadata={**meta, "maps": HESSIAN_MAP_NAMES})
        blobs.to_csv(paths["blobs"], index=False)
        for p in paths.values():
            print(f"    saved: {p}")
        print(f"    Hessian: {keep.sum() / valid.sum():.1%} of the pixels significant")
    return {"sigmas": sigmas, "log": log, "maps": dict(zip(HESSIAN_MAP_NAMES, maps)), "blobs": blobs}


def axis_vectors(maps: dict) -> tuple[np.ndarray, np.ndarray]:
    """napari Vectors (N, 2, 2) of the Hessian long axis on a grid, centered on the pixel, length 2 sigma."""
    h, w = maps["sigma"].shape
    yy, xx = np.mgrid[0:h:AXIS_STEP, 0:w:AXIS_STEP]
    yy, xx = yy.ravel(), xx.ravel()
    keep = np.isfinite(maps["anisotropy"][yy, xx]) & (maps["anisotropy"][yy, xx] >= AXIS_MIN_ANISOTROPY)
    yy, xx = yy[keep], xx[keep]
    angle = np.radians(maps["orientation"][yy, xx])
    length = 2 * maps["sigma"][yy, xx]
    proj = np.column_stack([np.sin(angle), np.cos(angle)]) * length[:, None]
    vectors = np.stack([np.column_stack([yy, xx]) - proj / 2, proj], axis=1)
    dark = maps["lambda1"][yy, xx] > 0
    colors = np.where(dark[:, None], [BLOB_COLORS["dark"]], [BLOB_COLORS["bright"]])[:, 0]
    return vectors, colors


def black_below_colormap(name: str, limits: tuple, cmap: str):
    """cmap from 0 to limits[1], black below 0 (non-significant pixels are given a value < 0)."""
    from matplotlib import colormaps
    from napari.utils import Colormap
    start = -limits[0] / (limits[1] - limits[0])
    controls = np.concatenate([[0, start - 0.01], np.linspace(start, 1, 64)])
    colors = np.concatenate([[[0, 0, 0, 1], [0, 0, 0, 1]], colormaps[cmap](np.linspace(0, 1, 64))])
    return Colormap(colors=colors, controls=controls, name=name)


def heightmap(image: np.ndarray, values: np.ndarray, faces: np.ndarray | None = None):
    """Surface (vertices, faces, values) on every HEIGHTMAP_STEP-th pixel, altitude = -image."""
    s = HEIGHTMAP_STEP
    img, val = image[::s, ::s], values[::s, ::s]
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w] * s
    vertices = np.column_stack([-img.ravel() * HEIGHT_SCALE, yy.ravel(), xx.ravel()]).astype(np.float32)
    if faces is None:
        idx = np.arange(h * w).reshape(h, w)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        faces = np.concatenate([np.column_stack([a, b, c]), np.column_stack([b, d, c])])
    return vertices, faces, val.ravel()


def style(axes) -> None:
    axes.set_facecolor(PANEL)
    axes.tick_params(colors="#c3c2b7", labelsize=7)
    for spine in axes.spines.values():
        spine.set_color("#5a5d66")
    axes.grid(color="#3a3d45", linewidth=0.5)


def curves_widget(viewer, results: dict, points):
    """Dock panel: B(sigma) of the shift-clicked pixels (both grids), histograms of sigma_peak below."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

    figure = Figure(figsize=(5, 8), facecolor=PANEL)
    curves = figure.add_subplot(3, 1, 1)
    bins = np.geomspace(1, 2 ** LOG_MAX, 61)
    for row, grid in enumerate(GRIDS):
        axes = figure.add_subplot(3, 1, row + 2)
        style(axes)
        maps = results[grid]["maps"]
        sigma, amplitude = maps["sigma_peak_interp"], maps["amplitude"]
        for sign, color, label in ((-1, "#3fa9f5", "dark peak (B < 0)"), (1, "#d95926", "bright peak (B > 0)")):
            sel = np.isfinite(sigma) & (np.sign(amplitude) == sign)
            axes.hist(sigma[sel], bins=bins, histtype="step", color=color, linewidth=1.2, label=label)
        axes.set_xscale("log")
        axes.set_xticks(results[grid]["sigmas"], [f"{s:.3g}" for s in results[grid]["sigmas"]])
        axes.set_ylabel(f"pixels ({grid})", color="#c3c2b7", fontsize=8)
        axes.legend(facecolor=PANEL, labelcolor="#f0efec", edgecolor="#5a5d66", fontsize=7)
    axes.set_xlabel("sigma at the peak of |B| (px, interpolated)", color="#c3c2b7", fontsize=8)
    clicks = []

    def redraw() -> None:
        curves.clear()
        style(curves)
        curves.axhline(0, color="#c3c2b7", linewidth=0.8)
        for i, (y, x) in enumerate(clicks):
            color = CURVE_COLORS[i % len(CURVE_COLORS)]
            for grid, line, marker in (("half_octave", "-", "o"), ("octave", "--", "s")):
                r = results[grid]
                curves.plot(r["centers"], r["bands"][:, y, x], line, marker=marker, color=color, markersize=4,
                            linewidth=1.0, markerfacecolor=color if grid == "half_octave" else "none",
                            label=f"({y}, {x})" if grid == "half_octave" else None)
        curves.set_xscale("log")
        curves.set_xticks(GRIDS["half_octave"], [f"{s:.3g}" for s in GRIDS["half_octave"]])
        curves.set_xlabel("band center sigma (px)  -o- half octave, --s-- octave", color="#c3c2b7", fontsize=8)
        curves.set_ylabel("B / (r - 1)  (%)", color="#c3c2b7", fontsize=8)
        if clicks:
            curves.legend(facecolor=PANEL, labelcolor="#f0efec", edgecolor="#5a5d66", fontsize=7)
        else:
            curves.set_title("shift + click on a pixel", color="#c3c2b7", fontsize=8)
        figure.tight_layout()
        canvas.draw_idle()
        points.data = np.array(clicks, dtype=float).reshape(-1, 2)
        points.face_color = [CURVE_COLORS[i % len(CURVE_COLORS)] for i in range(len(clicks))] or "white"

    def on_click(viewer, event):
        if event.type != "mouse_press" or "Shift" not in event.modifiers:
            return
        y, x = (int(round(v)) for v in event.position[-2:])
        h, w = results["fine"].shape
        if 0 <= y < h and 0 <= x < w:
            clicks.append((y, x))
            del clicks[:-MAX_CURVES]
            redraw()

    def clear() -> None:
        clicks.clear()
        redraw()

    widget = QWidget()
    layout = QVBoxLayout(widget)
    canvas = FigureCanvasQTAgg(figure)
    button = QPushButton("clear curves")
    button.clicked.connect(clear)
    layout.addWidget(NavigationToolbar2QT(canvas, widget))
    layout.addWidget(canvas)
    layout.addWidget(button)
    viewer.mouse_drag_callbacks.append(on_click)
    redraw()
    return widget


def weights_widget(results: dict, on_change):
    """Dock panel: one slider per component of the sharpening (0 to 5, 1 = unchanged)."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QGridLayout, QLabel, QPushButton, QSlider, QWidget

    sigmas = GRIDS[SHARPEN_GRID]
    names = [f"fine (< {sigmas[0]:.3g} px)"] + [f"band {a:.3g}-{b:.3g} px" for a, b in zip(sigmas[:-1], sigmas[1:])]
    widget = QWidget()
    layout = QGridLayout(widget)
    sliders, values = [], []
    for row, name in enumerate(names):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 50)
        slider.setValue(10)
        slider.setTracking(False)  # recompute on release only (the heightmap is heavy)
        value = QLabel("1.0")
        slider.sliderMoved.connect(lambda v, label=value: label.setText(f"{v / 10:.1f}"))
        slider.valueChanged.connect(lambda v, label=value: (label.setText(f"{v / 10:.1f}"), emit()))
        layout.addWidget(QLabel(name), row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value, row, 2)
        sliders.append(slider)

    def emit() -> None:
        on_change(np.array([s.value() / 10 for s in sliders]))

    def set_weights(weights) -> None:
        """Set every slider at once (one recomputation), e.g. from the analysis notebook."""
        for row, (s, w) in enumerate(zip(sliders, weights)):
            s.blockSignals(True)
            s.setValue(int(round(w * 10)))
            s.blockSignals(False)
            layout.itemAtPosition(row, 2).widget().setText(f"{s.value() / 10:.1f}")
        emit()

    def reset() -> None:
        for s in sliders:
            s.blockSignals(True)
            s.setValue(10)
            s.blockSignals(False)
        for row in range(len(sliders)):
            layout.itemAtPosition(row, 2).widget().setText("1.0")
        emit()

    button = QPushButton("reset (all 1 = original frame)")
    button.clicked.connect(reset)
    layout.addWidget(button, len(names), 0, 1, 3)
    widget.set_weights, widget.names = set_weights, names
    return widget


def build_viewers() -> dict:
    """Compute (or load) the frame, open both napari viewers; returns their handles (used by the notebook)."""
    import napari  # imported here so the functions can be used without a GUI

    print(f"segment {SEGMENT}, frame {FRAME} (global {global_frame()})")
    diff, valid = load_frame()
    results = compute(diff, valid)
    noise = noise_std(diff[valid])
    span = 4 * noise
    weights = np.ones(len(GRIDS[SHARPEN_GRID]))
    check = np.abs(sharpen(results, weights, valid) - diff)[valid].max()
    print(f"  noise std {noise:.2f} %; reconstruction with all weights 1: max error {check:.1e} %")

    cmap = sigma_colormap()
    viewer = napari.Viewer(title=f"{VIDEO} seg {SEGMENT} frame {global_frame()}: scale space (shift + click = curve)")
    viewer.add_image(diff, name="2 frame - background (%)", colormap="gray", contrast_limits=(-span, span))
    for grid in ("octave", "half_octave"):
        r = results[grid]
        viewer.add_image(r["bands"], name=f"4 bands {grid} (slider: band 0-{len(r['bands']) - 1}, fine -> coarse)",
                         colormap="gray", contrast_limits=(-span, span), visible=False)
        maps = r["maps"]
        amp_lim = float(np.nanpercentile(np.abs(maps["amplitude"]), 99))
        viewer.add_image(np.nan_to_num(maps["amplitude"]), name=f"4 amplitude at peak {grid} (blue dark, red bright)",
                         colormap="coolwarm", contrast_limits=(-amp_lim, amp_lim), visible=False)
        viewer.add_image(log_sigma_display(maps["sigma_sign_change"]),
                         name=f"4 sigma of sign change {grid} (log2: 0 = 1 px ... {LOG_MAX:g} = {2 ** LOG_MAX:g} px)",
                         colormap=cmap, contrast_limits=SIGMA_LIMITS, visible=False)
        viewer.add_image(log_sigma_display(maps["sigma_peak_interp"]),
                         name=f"4 sigma peak {grid} (log2: 0 = 1 px ... {LOG_MAX:g} = {2 ** LOG_MAX:g} px, black = noise)",
                         colormap=cmap, contrast_limits=SIGMA_LIMITS, visible=grid == "half_octave", opacity=0.6)
    sharp = viewer.add_image(sharpen(results, weights, valid), name="4 sharpened (band weights)", colormap="gray",
                             contrast_limits=(-span, span), visible=False)
    lh = compute_log_hessian(diff, valid)
    log_lim = float(np.percentile(np.abs(lh["log"][0][valid]), 99.5))
    viewer.add_image(-lh["log"], name=f"4 -LoG {LOG_GRID} (slider: sigma {', '.join(f'{s:.3g}' for s in lh['sigmas'])}; "
                     "black dark, as B)", colormap="gray", contrast_limits=(-log_lim, log_lim), visible=False)
    hm = lh["maps"]
    viewer.add_image(log_sigma_display(hm["sigma"]), name=f"4 hessian sigma (log2: 0 = 1 px ... {LOG_MAX:g}, black = noise)",
                     colormap=cmap, contrast_limits=SIGMA_LIMITS, visible=False, opacity=0.6)
    lam_lim = float(np.nanpercentile(np.abs(hm["lambda1"]), 99))
    viewer.add_image(-np.nan_to_num(hm["lambda1"]), name="4 hessian -lambda1 (blue dark, red bright)",
                     colormap="coolwarm", contrast_limits=(-lam_lim, lam_lim), visible=False)
    viewer.add_image(np.nan_to_num(hm["anisotropy"], nan=-0.1),
                     name="4 hessian anisotropy (0 round ... 1 rod, black = noise)",
                     colormap=black_below_colormap("anisotropy", (-0.1, 1), "viridis"), contrast_limits=(-0.1, 1),
                     visible=False)
    vectors, vector_colors = axis_vectors(hm)
    viewer.add_vectors(vectors, name=f"4 hessian long axis (every {AXIS_STEP} px, anisotropy >= "
                       f"{AXIS_MIN_ANISOTROPY:g}; yellow dark, violet bright)", edge_color=vector_colors,
                       edge_width=1, vector_style="line", visible=False)
    blobs = lh["blobs"]
    viewer.add_points(blobs[["y", "x"]].to_numpy(dtype=float), name="4 LoG blobs (yellow dark, violet bright)",
                      size=2 * blobs["radius"].to_numpy(), face_color="transparent",
                      border_color=[BLOB_COLORS[p] for p in blobs["polarity"]] or "white", border_width=0.08,
                      features=blobs, visible=False)
    points = viewer.add_points(np.empty((0, 2)), name="clicked pixels", size=9, border_color="black")
    viewer.window.add_dock_widget(curves_widget(viewer, results, points), name="B(sigma) curves", area="right")

    sigma_values = log_sigma_display(results[SHARPEN_GRID]["maps"]["sigma_peak_interp"])
    vertices, faces, values = heightmap(sharp.data, sigma_values)
    relief = napari.Viewer(title=f"{VIDEO} frame {global_frame()}: heightmap (altitude = -sharpened, color = sigma peak)",
                           ndisplay=3)
    surface = relief.add_surface((vertices, faces, values), name="heightmap", colormap=cmap,
                                 contrast_limits=SIGMA_LIMITS, shading="smooth")

    def update(new_weights: np.ndarray) -> None:
        sharp.data = sharpen(results, new_weights, valid)
        sharp.visible = True  # on top of the maps: what the sliders change is what is shown
        surface.data = heightmap(sharp.data, sigma_values, faces)
        print(f"  weights: {', '.join(f'{w:.1f}' for w in new_weights)}")

    weights_panel = weights_widget(results, update)
    viewer.window.add_dock_widget(weights_panel, name="sharpening weights", area="right")
    return {"viewer": viewer, "relief": relief, "results": results, "diff": diff, "valid": valid, "noise": noise,
            "sharp": sharp, "surface": surface, "faces": faces, "sigma_values": sigma_values, "points": points,
            "weights_panel": weights_panel, "update": update, "log_hessian": lh}


def main() -> None:
    import napari

    build_viewers()
    napari.run()


if __name__ == "__main__":
    main()
