"""Lower the quality of a video to 720p with ffmpeg.

The original video is never modified: the result is written next to it as <name>_720p.<ext>
(or to the path given with -o). Videos already at or below 720p are not upscaled.
Requires ffmpeg (brew install ffmpeg).
"""
# python downscale_video.py /path/to/video.mp4
import argparse
import subprocess
import sys
from pathlib import Path

HEIGHT = 720
CRF = 23          # x264 quality: lower = better quality / bigger file (18-28 is the useful range)
PRESET = "medium"


def downscale(video: Path, output: Path) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-y", "-i", str(video),
           "-map", "0:v:0", "-map", "0:a?",
           # Height capped at 720 (no upscaling); width follows the aspect ratio and stays even.
           "-vf", f"scale=-2:'min({HEIGHT},ih)'",
           "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET, "-pix_fmt", "yuv420p",
           "-c:a", "copy", str(output)]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Downscale a video to {HEIGHT}p with ffmpeg.")
    parser.add_argument("video", help="path of the video to modify")
    parser.add_argument("-o", "--output", help="output path (default: <video>_720p.<ext> next to the input)")
    args = parser.parse_args()

    video = Path(args.video).expanduser()
    if not video.is_file():
        sys.exit(f"Video not found: {video}")

    output = Path(args.output).expanduser() if args.output else video.with_name(f"{video.stem}_{HEIGHT}p{video.suffix}")
    if output.resolve() == video.resolve():
        sys.exit("Output path must differ from the input path.")

    downscale(video, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
