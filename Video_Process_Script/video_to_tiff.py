"""Downscale a video and save it as one multipage 8-bit grayscale TIFF (BigTIFF, lossless zlib).

ffmpeg decodes, downscales (area filter: mean of the source pixels) and keeps only the luma plane (Y)
of the video, so the gray levels are the encoded ones (no color conversion). Frames are streamed to
the TIFF one by one: the video never has to fit in RAM. Each decoded frame is kept once (no frame
duplicated to fill a constant rate) and its real timestamp is stored in the TIFF metadata, since
microscope recordings can have a variable frame rate. The original video is never modified and an
existing output is never overwritten. --start / --end keep only a time range of the video (player
time, s or mm:ss): the frames are selected by their timestamp, so the pages are the same as a full
conversion cut afterwards.
Requires ffmpeg (brew install ffmpeg) and tifffile.
"""
# python video_to_tiff.py /path/to/video.mp4 -o /path/to/output.tif [--start 11:18 --end 13:00]
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

HEIGHT = 1080     # output height (no upscaling); the width follows the aspect ratio and stays even


def parse_time(text: str) -> float:
    """'678', '11:18' or '0:11:18' -> seconds."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def probe(video: Path, start: float | None = None, end: float | None = None) -> tuple[dict, list[float]]:
    """Stream size and the timestamps (s) of every decoded frame (inside [start, end] if given).

    Frames are counted after decoding, not from the container packets: a packet that cannot be
    decoded gives no frame, and the TIFF must have exactly one page per decoded frame.
    """
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height:frame=pts_time", "-of", "json"]
    if start is not None:
        cmd += ["-read_intervals", f"{start}%{end}"]  # seeks to the keyframe before start: filtered below
    out = subprocess.run(cmd + [str(video)], check=True, capture_output=True, text=True)
    data = json.loads(out.stdout)
    times = [float(f["pts_time"]) for f in data["frames"] if "pts_time" in f]
    if start is not None:
        times = [t for t in times if start <= t <= end]
    return data["streams"][0], times


def output_size(width: int, height: int) -> tuple[int, int]:
    h = min(HEIGHT, height)
    w = round(width * h / height / 2) * 2
    return w, h


def frames(video: Path, w: int, h: int, start: float | None = None, end: float | None = None):
    seek, select = [], ""
    if start is not None:
        # fast seek a bit before start, original timestamps kept (-copyts), then exact selection on them
        seek = ["-ss", str(max(start - 10, 0)), "-copyts"]
        select = f"select='between(t,{start},{end})',"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *seek, "-i", str(video), "-map", "0:v:0",
           "-vf", f"{select}scale={w}:{h}:flags=area,format=gray", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h
    while (buf := proc.stdout.read(size)) and len(buf) == size:
        yield np.frombuffer(buf, np.uint8).reshape(h, w)
    proc.stdout.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Video -> {HEIGHT}p 8-bit grayscale multipage TIFF.")
    parser.add_argument("video", help="path of the video to convert")
    parser.add_argument("-o", "--output", help="output path (default: <video>_<HEIGHT>p_gray.tif next to the input)")
    parser.add_argument("--start", help="first time kept (player time: s, mm:ss or h:mm:ss)")
    parser.add_argument("--end", help="last time kept (same format)")
    args = parser.parse_args()
    if (args.start is None) != (args.end is None):
        sys.exit("--start and --end go together")
    start = parse_time(args.start) if args.start else None
    end = parse_time(args.end) if args.end else None
    if start is not None and end <= start:
        sys.exit("--end must be after --start")
    clip = f"_{int(start)}-{int(end)}s" if start is not None else ""

    video = Path(args.video).expanduser()
    if not video.is_file():
        sys.exit(f"Video not found: {video}")
    output = Path(args.output).expanduser() if args.output else video.with_name(f"{video.stem}{clip}_{HEIGHT}p_gray.tif")
    if output.exists():
        sys.exit(f"Output already exists, not overwritten: {output}")

    info, times = probe(video, start, end)
    if len(times) < 2:
        sys.exit(f"{len(times)} frame(s) in the range")
    w, h = output_size(info["width"], info["height"])
    n_total = len(times)
    scale = info["height"] / h
    fps = (n_total - 1) / (times[-1] - times[0])
    print(f"{info['width']}x{info['height']} -> {w}x{h} (factor {scale:.4f}), {n_total} frames, "
          f"mean {fps:.3f} fps" + (f", video time {times[0]:.3f}-{times[-1]:.3f} s" if start is not None else ""))

    # Kept in the TIFF description: the scale factor converts a µm/px calibration back to the source,
    # the timestamps give the real time of each frame (variable frame rate).
    meta = {"source": video.name, "source_size": [info["width"], info["height"]],
            "downscale_factor": scale, "mean_fps": fps,
            "timestamps_s": [round(t - times[0], 4) for t in times]}
    if start is not None:
        meta["clip_s"] = [start, end]                      # range asked (video time)
        meta["source_first_frame_s"] = round(times[0], 4)  # video time of page 0
    count = 0

    def counted():
        nonlocal count
        for frame in frames(video, w, h, start, end):
            count += 1
            if count % 100 == 0:
                print(f"\r{count}/{n_total} frames", end="", flush=True)
            yield frame

    # One series of n_total pages (napari / tifffile.imread read it as a (t, y, x) stack).
    with tifffile.TiffWriter(output, bigtiff=True) as tif:
        tif.write(counted(), shape=(n_total, h, w), dtype=np.uint8, compression="zlib",
                  photometric="minisblack", metadata=meta)
    if count != n_total:
        print(f"\nWarning: {count} frames decoded, {n_total} expected")
    print(f"\rSaved {count} frames: {output}")


if __name__ == "__main__":
    main()
