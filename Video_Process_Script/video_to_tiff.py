"""Downscale a video and save it as one multipage 8-bit grayscale TIFF (BigTIFF, lossless zlib).

ffmpeg decodes, downscales (area filter: mean of the source pixels) and keeps only the luma plane (Y)
of the video, so the gray levels are the encoded ones (no color conversion). Frames are streamed to
the TIFF one by one: the video never has to fit in RAM. Each decoded frame is kept once (no frame
duplicated to fill a constant rate) and its real timestamp is stored in the TIFF metadata, since
microscope recordings can have a variable frame rate. The original video is never modified and an
existing output is never overwritten.
Requires ffmpeg (brew install ffmpeg) and tifffile.
"""
# python video_to_tiff.py /path/to/video.mp4 -o /path/to/output.tif
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

HEIGHT = 1080     # output height (no upscaling); the width follows the aspect ratio and stays even


def probe(video: Path) -> tuple[dict, list[float]]:
    """Stream size and the timestamps (s) of every decoded frame.

    Frames are counted after decoding, not from the container packets: a packet that cannot be
    decoded gives no frame, and the TIFF must have exactly one page per decoded frame.
    """
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height:frame=pts_time",
                          "-of", "json", str(video)], check=True, capture_output=True, text=True)
    data = json.loads(out.stdout)
    times = [float(f["pts_time"]) for f in data["frames"]]
    return data["streams"][0], times


def output_size(width: int, height: int) -> tuple[int, int]:
    h = min(HEIGHT, height)
    w = round(width * h / height / 2) * 2
    return w, h


def frames(video: Path, w: int, h: int):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-map", "0:v:0",
           "-vf", f"scale={w}:{h}:flags=area,format=gray", "-fps_mode", "passthrough",
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
    args = parser.parse_args()

    video = Path(args.video).expanduser()
    if not video.is_file():
        sys.exit(f"Video not found: {video}")
    output = Path(args.output).expanduser() if args.output else video.with_name(f"{video.stem}_{HEIGHT}p_gray.tif")
    if output.exists():
        sys.exit(f"Output already exists, not overwritten: {output}")

    info, times = probe(video)
    w, h = output_size(info["width"], info["height"])
    n_total = len(times)
    scale = info["height"] / h
    fps = (n_total - 1) / (times[-1] - times[0])
    print(f"{info['width']}x{info['height']} -> {w}x{h} (factor {scale:.4f}), {n_total} frames, "
          f"mean {fps:.3f} fps")

    # Kept in the TIFF description: the scale factor converts a µm/px calibration back to the source,
    # the timestamps give the real time of each frame (variable frame rate).
    meta = {"source": video.name, "source_size": [info["width"], info["height"]],
            "downscale_factor": scale, "mean_fps": fps,
            "timestamps_s": [round(t - times[0], 4) for t in times]}
    count = 0

    def counted():
        nonlocal count
        for frame in frames(video, w, h):
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
