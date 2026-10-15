"""Bounded, synchronous media I/O for an offline CLI; never called by the UI."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tool(name):
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"missing_tool: {name}")
    return executable


def run(args, **kwargs):
    return subprocess.run(args, check=True, creationflags=FLAGS, timeout=1800, **kwargs)


def ffmpeg(*args):
    return [tool("ffmpeg"), "-v", "error", "-nostdin", *map(str, args)]


def probe(path, count=False):
    args = [tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count:
        args.append("-count_frames")
    return json.loads(run([*args, str(path)], capture_output=True).stdout)


@contextmanager
def pipe(args, log, writing=False):
    """One bounded OS pipe; stderr goes to disk to prevent deadlocks."""
    with Path(log).open("wb") as errors:
        process = subprocess.Popen(args, stdin=subprocess.PIPE if writing else subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL if writing else subprocess.PIPE,
                                   stderr=errors, creationflags=FLAGS)
        stream = process.stdin if writing else process.stdout
        try:
            yield stream
            stream.close()
            if process.wait(timeout=180):
                raise RuntimeError(f"media_process_failed: {log}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if not stream.closed:
                stream.close()


def frames(path, size, count, log):
    width, height = size
    args = ffmpeg("-threads", 3, "-filter_threads", 1, "-i", path, "-an", "-frames:v", count,
                  "-vf", f"scale={width}:{height}:flags=area,format=rgb24",
                  "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1")
    with pipe(args, log) as stream:
        for _ in range(count):
            raw = stream.read(width * height * 3)
            if len(raw) != width * height * 3:
                raise RuntimeError("truncated_video")
            yield raw
        if stream.read(1):
            raise RuntimeError("extra_video_frames")


def frame_hashes(path, count, start=None):
    args = ffmpeg("-threads", 3)
    if start is not None:
        args += ["-ss", str(start)]
    result = run([*args, "-i", str(path), "-map", "0:v:0", "-an", "-frames:v", str(count),
                  "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-f", "framemd5", "pipe:1"],
                 capture_output=True).stdout.decode("utf-8")
    hashes = [line.rsplit(",", 1)[-1].strip() for line in result.splitlines()
              if line and not line.startswith("#")]
    if len(hashes) != count:
        raise RuntimeError("frame_hash_count_mismatch")
    return hashes
