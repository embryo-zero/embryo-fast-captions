#!/cluster/svc/venv-captions/bin/python
"""Embryo Fast Captions (our faster-whisper fork, MIT; Whisper "small" model, MIT). Written for Embryo-Zero 2026-11-24.
Usage: embryo-captions <audio-or-video> [--out DIR] [--lang en] [--cpu] [--words]
Writes <name>.srt and <name>.vtt next to the input (or into --out) and prints one JSON line with the result.
Runs on the processor (int8, ~2x faster than real time); EMBRYO_CAPTIONS_GPU=1 tries the card."""
import argparse, json, os, sys, time

MODEL = "/cluster/svc/models/captions/small"


def ts(t, sep):
    h, r = divmod(max(0.0, t), 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"


def pick_device(force_cpu):
    if force_cpu or os.environ.get("EMBRYO_CAPTIONS_GPU") != "1":   # processor by default: the card is shared with
        return "cpu", "int8"                                          # training + Blender, and cuBLAS 12 isn't installed
    try:
        import ctranslate2, subprocess
        if ctranslate2.get_cuda_device_count() > 0:
            free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=10).stdout.split()[0])
            if free > 1500:                       # leave room for training / Blender on the shared card
                return "cuda", "int8"              # Pascal cards (Quadro P2000) have no fast float16
    except Exception:
        pass
    return "cpu", "int8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media"); ap.add_argument("--out"); ap.add_argument("--lang")
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--words", action="store_true")
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    dev, ct = pick_device(a.cpu)
    t0 = time.time()
    try:
        model = WhisperModel(MODEL, device=dev, compute_type=ct, cpu_threads=4)
    except Exception:
        dev, ct = "cpu", "int8"
        model = WhisperModel(MODEL, device=dev, compute_type=ct, cpu_threads=4)
    segs, info = model.transcribe(a.media, language=a.lang, vad_filter=True, word_timestamps=a.words, beam_size=5)
    segs = list(segs)
    base = os.path.splitext(os.path.basename(a.media))[0]
    out = a.out or os.path.dirname(os.path.abspath(a.media))
    os.makedirs(out, exist_ok=True)
    srt, vtt = f"{out}/{base}.srt", f"{out}/{base}.vtt"
    with open(srt, "w") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{ts(s.start, ',')} --> {ts(s.end, ',')}\n{s.text.strip()}\n\n")
    with open(vtt, "w") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            f.write(f"{ts(s.start, '.')} --> {ts(s.end, '.')}\n{s.text.strip()}\n\n")
    res = {"ok": True, "srt": srt, "vtt": vtt, "language": info.language, "seconds": round(info.duration, 2),
           "segments": len(segs), "text": " ".join(s.text.strip() for s in segs), "device": dev,
           "took": round(time.time() - t0, 1),
           "list": [[round(s.start, 2), round(s.end, 2), s.text.strip()] for s in segs]}
    if a.words:
        res["words"] = [[w.word, round(w.start, 2), round(w.end, 2)] for s in segs for w in (s.words or [])]
    print(json.dumps(res))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:300]})); sys.exit(1)
