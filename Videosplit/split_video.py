"""Split a video into 3-second segments, sorted into Death / Expo / Stationary folders.

The original video is never modified: ffmpeg only reads it and writes new files.
Requires ffmpeg (brew install ffmpeg).
"""
# python split_video.py
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH = "/Users/juwul/Movies/Stationary2.mp4"  
OUTPUT_ROOT = Path("/Users/juwul/Desktop/Internship/Videosplit")
CATEGORIES = ["Death", "Expo", "Stationary"]  
SEGMENT_SECONDS = 10

# "copy"  -> stream copy: pixel-identical to the original (no re-encoding), but ffmpeg can only
#            cut on keyframes, so segments may be a bit longer than 3 s (and the last one shorter).
# "exact" -> re-encodes losslessly (x264 CRF 0) so every cut is exactly 3 s. Slower, much bigger files.
MODE = "copy"
# ---------------------------------------------------------------------------


def find_category(video: Path) -> str | None:
    name = video.stem.lower()
    for category in CATEGORIES:
        if category.lower() in name:
            return category
    return None


def split_video(video: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = video.stem

    # Remove splits from a previous run of the same video so counts stay correct.
    for old in out_dir.glob(f"{prefix}_split_*{video.suffix}"):
        old.unlink()

    pattern = out_dir / f"{prefix}_split_%03d{video.suffix}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-map", "0:v:0", "-map", "0:a?"]
    if MODE == "exact":
        cmd += ["-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-c:a", "copy",
                "-force_key_frames", f"expr:gte(t,n_forced*{SEGMENT_SECONDS})"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-f", "segment", "-segment_time", str(SEGMENT_SECONDS),
            "-reset_timestamps", "1", str(pattern)]

    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob(f"{prefix}_split_*{video.suffix}"))


def write_report(video: Path, category: str, splits: list[Path]) -> Path:
    report = OUTPUT_ROOT / f"{video.stem}_report.txt"
    lines = [
        f"Video:           {video}",
        f"Category:        {category}",
        f"Mode:            {MODE}",
        f"Segment length:  {SEGMENT_SECONDS} s",
        f"Total splits:    {len(splits)}",
        f"Saved in:        {splits[0].parent if splits else OUTPUT_ROOT / category}",
        "",
        "Files:",
        *[f"  {s.name}" for s in splits],
    ]
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    video = Path(VIDEO_PATH).expanduser()
    if not video.is_file():
        sys.exit(f"Video not found: {video}\nEdit VIDEO_PATH at the top of the script.")

    category = find_category(video)
    if category is None:
        sys.exit(f"Can't classify '{video.name}': its name must contain one of {CATEGORIES}.")

    splits = split_video(video, OUTPUT_ROOT / category)
    report = write_report(video, category, splits)
    print(f"{len(splits)} splits saved in {OUTPUT_ROOT / category}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
