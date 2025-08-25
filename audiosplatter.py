# =============================================================================
# AudioSplatter – Aufrufe & Beispiele
#
# Voraussetzungen:
#   • Modelle: verwende die exakten Dateinamen aus `audio-separator --list_models`
#   • --device:
#       - Linux/NVIDIA:   installiere `onnxruntime-gpu`  → GPU = CUDAExecutionProvider
#       - Windows/NVIDIA: installiere `onnxruntime-gpu`  → GPU = CUDAExecutionProvider
#       - Windows/AMD/Intel: installiere `onnxruntime-directml` → GPU = DmlExecutionProvider
#       - CPU-Only:       installiere `onnxruntime` (oder erzwinge `--device cpu`)
#
# Standard (1-Pass, 10s/0.1s, linear):
#   Linux:
#       python audiosplatter.py song.mp3
#       python audiosplatter.py song.mp3 --device cuda          # GPU erzwingen (NVIDIA)
#       python audiosplatter.py song.mp3 --device cpu --threads 8
#   Windows (CLI-EXE):
#       .\AudioSplatter.exe song.mp3
#       .\AudioSplatter.exe song.mp3 --device dml --threads 4   # AMD/Intel-GPU via DirectML
#       .\AudioSplatter.exe song.mp3 --device cpu --threads 4
#
# Robust (2-Pass, Aussetzer-ärmer):
#   Linux:
#       python audiosplatter.py song.mp3 --robust
#   Windows:
#       .\AudioSplatter.exe song.mp3 --robust
#
# Modelle & Ordner:
#   python audiosplatter.py song.mp3 \
#       --models-dir ./models \
#       --vocals-model UVR_MDXNET_KARA_2.onnx \
#       --drums-model  kuielab_a_drums.onnx \
#       --bass-model   kuielab_a_bass.onnx \
#       -o stems4 --format wav
#
# Performance-Tipps (v. a. auf CPU/Windows):
#   • Mehr Threads setzen:       --threads <physische_Kerne>  (z. B. 4 beim Ryzen 7 3700U)
#   • Größere Chunks reduzieren Overhead:
#       --chunk-length 20 --chunk-overlap 0.1     # ~halbiert Modellstarts
#   • 2-Pass nur bei Bedarf (–robust), sonst 1-Pass für Tempo.
#
# Debug-Hinweis:
#   Beim Start meldet das Script z. B.: [INFO] Device: auto → CUDA / DirectML / CPU
# =============================================================================

from __future__ import annotations
# -------------------------- Warnungen zähmen -------------------------- #
import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=r".*torch\.distributed\._sharded_tensor.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*torch\.cuda\.amp\.autocast.*")

# ------------------------------ Imports ------------------------------ #
import os
import sys
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from audio_separator.separator import Separator as _Separator


# -------------------------- Hilfsfunktionen -------------------------- #

def _fade_windows(n: int, shape: str = "linear"):
    """Gibt (fade_in, fade_out) zurück. Equal-Power: konstante Leistung im Crossfade."""
    if n <= 0:
        return None, None
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if shape == "equal_power":
        fade_in = np.sin(0.5 * np.pi * t)
        fade_out = np.cos(0.5 * np.pi * t)
    else:  # linear
        fade_in = t
        fade_out = 1.0 - t
    return fade_in, fade_out


def _chunk_grid(n_samples: int, L: int, O: int, offset: int = 0):
    """Startpositionen mit Schritt (L-O); optional um 'offset' Samples verschoben."""
    if L <= 0:
        raise ValueError("chunk_length muss > 0 sein")
    if not (0 <= O < L):
        raise ValueError("chunk_overlap muss >= 0 und < chunk_length sein")
    step = L - O
    s = max(0, offset)
    starts = []
    while s < n_samples:
        starts.append(s)
        if s + L >= n_samples:
            break
        s += step
    return starts


def _ensure_stereo(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return np.stack([wav, wav], axis=-1).astype(np.float32, copy=False)
    if wav.shape[1] == 1:
        return np.repeat(wav, 2, axis=1).astype(np.float32, copy=False)
    return wav.astype(np.float32, copy=False)


def _load_audio(path: Path):
    audio, sr = sf.read(str(path), always_2d=False)
    audio = _ensure_stereo(audio)
    return audio, sr


def _write_audio(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def _resolve_sep_output(base: Path, p: str | Path) -> Path:
    """Mappt vom Separator zurückgegebene (evtl. relative) Pfade sicher auf base."""
    p = Path(p)
    if p.is_absolute():
        return p
    candidate = base / p
    if candidate.exists():
        return candidate
    matches = list(base.rglob(p.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Konnte Output-Datei nicht finden: {p} (gesucht unter {base})")


def _resample_to_len(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linear-Resampling auf target_len Samples (shape: (N,2))."""
    n = x.shape[0]
    if n == target_len:
        return x.astype(np.float32, copy=False)
    if n < 2:
        out = np.zeros((target_len, x.shape[1]), dtype=np.float32)
        if n == 1:
            out[:] = x[0]
        return out
    idx_old = np.arange(n, dtype=np.float64)
    idx_new = np.linspace(0, n - 1, target_len, dtype=np.float64)
    out = np.empty((target_len, x.shape[1]), dtype=np.float32)
    for ch in range(x.shape[1]):
        out[:, ch] = np.interp(idx_new, idx_old, x[:, ch].astype(np.float64))
    return out


def _align_to_chunk(stem: np.ndarray, ref_chunk: np.ndarray, max_lag: int = 1024) -> np.ndarray:
    """Kleine Kreuzkorrelation (±max_lag), um Modell-Latenzen zu kompensieren."""
    L = ref_chunk.shape[0]
    if stem.shape[0] != L:
        stem = _resample_to_len(stem, L)

    a = ref_chunk.mean(axis=1)
    b = stem.mean(axis=1)
    best_lag, best_val = 0, -1e30

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[-lag:], b[:L + lag]
        elif lag > 0:
            aa, bb = a[:L - lag], b[lag:]
        else:
            aa, bb = a, b
        if aa.size < 256:
            continue
        val = float(np.dot(aa, bb))
        if val > best_val:
            best_val, best_lag = val, lag

    if best_lag < 0:
        pad = np.zeros((-best_lag, stem.shape[1]), dtype=np.float32)
        stem = np.vstack([stem[-best_lag:], pad])
    elif best_lag > 0:
        pad = np.zeros((best_lag, stem.shape[1]), dtype=np.float32)
        stem = np.vstack([pad, stem[:-best_lag]])
    return stem


def _rebalance_gains(chunk: np.ndarray, v: np.ndarray, d: np.ndarray, b: np.ndarray):
    """Least-Squares für g_v, g_d, g_b (>=0), um den Chunk bestmöglich zu erklären."""
    M = np.stack([v.mean(axis=1), d.mean(axis=1), b.mean(axis=1)], axis=1)  # (L,3)
    y = chunk.mean(axis=1)
    g, *_ = np.linalg.lstsq(M, y, rcond=None)
    g = np.clip(g, 0.0, 2.0)
    return float(g[0]), float(g[1]), float(g[2])


def _bass_rescue_ffmpeg(orig_path: Path, bass_path: Path, freq: float, gain: float) -> Path:
    """Optional: Subbass aus Original (<freq Hz, gain) in den Bass-Stem mischen (per ffmpeg)."""
    out = bass_path.with_name(bass_path.stem + "_fixed.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(bass_path), "-i", str(orig_path),
        "-filter_complex", f"[1:a]lowpass=f={freq},volume={gain}[low];"
                           f"[0:a][low]amix=inputs=2:weights=1 1[out]",
        "-map", "[out]", str(out)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def _run_model_on_file(sep: _Separator, chunk_file: Path, stem_key: str, sep_base: Path) -> Path:
    """
    Führt die Separation aus und wählt robust die *positiven* Stems:
    - vocals  → vermeidet 'instrumental', 'no/without/minus vocals', 'karaoke'
    - drums   → vermeidet 'no/without/minus drums'
    - bass    → vermeidet 'no/without/minus bass'
    Fällt notfalls auf energie-basiertes Ranking zurück.
    """
    import re
    out_files = sep.separate(str(chunk_file))
    if not isinstance(out_files, (list, tuple)):
        out_files = [out_files]
    candidates = [_resolve_sep_output(sep_base, f) for f in out_files]
    sk = stem_key.lower()

    POS = {"vocals": [r"\bvoc(?:al)?s?\b", r"\bvoice(?:s)?\b"],
           "drums":  [r"\bdrum(?:s)?\b"],
           "bass":   [r"\bbass\b"]}
    NEG = {"vocals": [r"\binstrumental\b", r"\bno[ _-]?voc(?:al)?s?\b",
                      r"\bwithout[ _-]?voc(?:al)?s?\b", r"\bminus[ _-]?voc(?:al)?s?\b",
                      r"\bkara(?:oke)?\b"],
           "drums":  [r"\bno[ _-]?drum(?:s)?\b", r"\bwithout[ _-]?drum(?:s)?\b",
                      r"\bminus[ _-]?drum(?:s)?\b"],
           "bass":   [r"\bno[ _-]?bass\b", r"\bwithout[ _-]?bass\b", r"\bminus[ _-]?bass\b"]}

    best_path, best_score = None, -1e9
    for p in candidates:
        name = p.name.lower()
        score = 0.0
        for pat in NEG.get(sk, []):
            if re.search(pat, name): score -= 100.0
        for pat in POS.get(sk, []):
            if re.search(pat, name): score += 10.0
        try:
            x, _ = sf.read(str(p), always_2d=False)
            x = _ensure_stereo(x)
            score += float(np.sqrt(np.mean(x**2)))
        except Exception:
            pass
        if score > best_score:
            best_path, best_score = p, score

    if best_path is None:
        raise RuntimeError(f"Kein passendes {stem_key}-Stem gefunden. Kandidaten: {[c.name for c in candidates]}")
    return best_path


# ------------------------- Pass-Verarbeitung ------------------------- #

def _process_pass(mix: np.ndarray, sr: int, L: int, O: int, offset: int,
                  fade_in: np.ndarray | None, fade_out: np.ndarray | None,
                  sep_v: _Separator, sep_d: _Separator, sep_b: _Separator,
                  sep_base: Path, temp_root: Path, align: bool, nnls: bool,
                  pass_name: str = "P1"):
    """Ein kompletter Durchlauf über alle Chunks mit gegebenem Raster-Offset."""
    n = mix.shape[0]
    stems = {k: np.zeros_like(mix, dtype=np.float32) for k in ["vocals", "drums", "bass"]}
    weight = np.zeros((n,), dtype=np.float32)

    starts = _chunk_grid(n, L, O, offset)
    total = len(starts)

    for i, s0 in enumerate(starts):
        s1 = min(s0 + L, n)
        chunk = mix[s0:s1, :]
        Lc = chunk.shape[0]

        cfile = temp_root / f"chunk_off{offset}_{i:04d}.wav"
        _write_audio(cfile, chunk, sr)

        # Fortschritt ausgeben (nützlich für GUI/Logging)
        print(f"[{pass_name}] Chunk {i+1}/{total} @ {s0/sr:.2f}s - {s1/sr:.2f}s")

        vf = _run_model_on_file(sep_v, cfile, "vocals", sep_base)
        df = _run_model_on_file(sep_d, cfile, "drums",  sep_base)
        bf = _run_model_on_file(sep_b, cfile, "bass",   sep_base)

        v, srv = _load_audio(vf)
        d, srd = _load_audio(df)
        b, srb = _load_audio(bf)

        if srv != sr or v.shape[0] != Lc: v = _resample_to_len(v, Lc)
        if srd != sr or d.shape[0] != Lc: d = _resample_to_len(d, Lc)
        if srb != sr or b.shape[0] != Lc: b = _resample_to_len(b, Lc)

        if align:
            v = _align_to_chunk(v, chunk)
            d = _align_to_chunk(d, chunk)
            b = _align_to_chunk(b, chunk)

        if nnls:
            gv, gd, gb = _rebalance_gains(chunk, v, d, b)
            v *= gv; d *= gd; b *= gb

        w = np.ones((Lc,), dtype=np.float32)
        if O > 0 and i > 0:      w[:O] = np.minimum(w[:O], fade_in)
        if O > 0 and s1 < n:     w[-O:] = np.minimum(w[-O:], fade_out)

        stems["vocals"][s0:s1, :] += v * w[:, None]
        stems["drums"][s0:s1, :]  += d * w[:, None]
        stems["bass"][s0:s1, :]   += b * w[:, None]
        weight[s0:s1] += w

        for f in (vf, df, bf):
            try: Path(f).unlink(missing_ok=True)
            except Exception: pass

    nz = weight > 0
    for k in stems:
        stems[k][nz, :] /= weight[nz, None]
    return stems, (weight > 0)


# ------------------------------- Main -------------------------------- #

def _arg_provided(flag: str) -> bool:
    for i, tok in enumerate(sys.argv[1:]):
        if tok == flag: return True
        if tok.startswith(flag + "="): return True
        if tok == flag and i + 2 <= len(sys.argv[1:]):  # "--flag value"
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Chunked 4-Stem Demixing (MDX) – 1-Pass Default, 2-Pass via --robust; CPU/GPU umschaltbar"
    )
    ap.add_argument("input", help="Audio-Datei (z. B. .wav, .flac, .mp3)")
    ap.add_argument("-o", "--outdir", default="stems4", help="Ausgabeordner")
    ap.add_argument("--models-dir", default="models", help="Ordner für Modelldateien/Cache")
    ap.add_argument("--format", default="wav", help="Ausgabeformat (empfohlen: wav/flac)")

    # Umschalter & Preset
    ap.add_argument("--robust", action="store_true",
                    help="Aktiviere 2-Pass (avg) mit konservativen Defaults (8s/0.3, linear).")
    ap.add_argument("--preset", choices=["robust"], default=None,
                    help="Alias für --robust (Kompatibilität).")

    # Chunking / Fades – 1-Pass Defaults (Comfy-like)
    ap.add_argument("--chunk-length", type=float, default=10.0, help="Chunk-Länge in Sekunden")
    ap.add_argument("--chunk-overlap", type=float, default=0.1, help="Überlappung in Sekunden")
    ap.add_argument("--fade-shape", choices=["linear", "equal_power"], default="linear",
                    help="Crossfade-Form")

    # Verbesserungen (optional)
    ap.add_argument("--align", action="store_true",
                    help="Pro Chunk Stems via Kreuzkorrelation an den Mix alignen")
    ap.add_argument("--nnls", action="store_true",
                    help="Per-Chunk Least-Squares-Rebalancing (Voc/Drums/Bass) aktivieren")

    # Zwei-Pass-Ensemble – Default: off
    ap.add_argument("--twopass", choices=["off", "avg"], default="off",
                    help="Zwei Durchläufe: 2. Pass versetzt; 'avg' mittelt beide.")

    # Device & Threads
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto",
                    help="Rechen-Backend: auto (GPU falls verfügbar), cpu (erzwingen), "
                         "cuda (NVIDIA), dml (DirectML für AMD/Intel)")
    ap.add_argument("--threads", type=int, default=0,
                    help="Max. CPU-Threads für ONNX/BLAS (0 = automatisch)")

    # MDX-Modelldateinamen (GENAU wie in `audio-separator --list_models`)
    ap.add_argument("--vocals-model", default="UVR_MDXNET_KARA_2.onnx")
    ap.add_argument("--drums-model",  default="kuielab_a_drums.onnx")
    ap.add_argument("--bass-model",   default="kuielab_a_bass.onnx")

    # Optionaler Bass-Rescue (aus)
    ap.add_argument("--bass-rescue", type=str, default=None,
                    help="Subbass aus Original beimischen: 'freq,gain' (z.B. 120,0.3) – benötigt ffmpeg")

    args = ap.parse_args()

    # robust-Preset anwenden, falls gewünscht (ohne explizite Overrides zu überfahren)
    if args.robust or args.preset == "robust":
        if not _arg_provided("--chunk-length"): args.chunk_length = 8.0
        if not _arg_provided("--chunk-overlap"): args.chunk_overlap = 0.3
        if not _arg_provided("--fade-shape"): args.fade_shape = "linear"
        if not _arg_provided("--twopass"): args.twopass = "avg"

    # --- Backend- & Thread-Setup ---
    if args.threads and args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception as e:
        providers = ["CPUExecutionProvider"]
        print(f"[WARN] onnxruntime nicht importierbar ({e}) → CPU-Modus.")

    def have(p): return p in providers

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU „unsichtbar“ → CPU
        print("[INFO] Device: CPU (erzwungen)")
    elif args.device == "cuda":
        if have("CUDAExecutionProvider"):
            print("[INFO] Device: CUDA")
        else:
            print("[WARN] CUDAExecutionProvider nicht verfügbar → CPU")
    elif args.device == "dml":
        if have("DmlExecutionProvider"):
            print("[INFO] Device: DirectML")
        else:
            print("[WARN] DmlExecutionProvider nicht verfügbar → CPU")
    else:
        if have("CUDAExecutionProvider"):
            print("[INFO] Device: auto → CUDA")
        elif have("DmlExecutionProvider"):
            print("[INFO] Device: auto → DirectML")
        else:
            print("[INFO] Device: auto → CPU")

    # -------------------- I/O & Chunk-Parameter -------------------- #
    in_path = Path(args.input)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)

    mix, sr = _load_audio(in_path)
    n = mix.shape[0]

    L = int(round(args.chunk_length * sr))
    O = int(round(args.chunk_overlap * sr))
    fade_in, fade_out = _fade_windows(O, args.fade_shape)

    # Temp-Root & gemeinsames Separator-Output
    work_root = Path(tempfile.mkdtemp(prefix="chunkdemix_"))
    sep_base = work_root / "sep_master"

    # Separatoren je Stem initialisieren (einmal; für beide Pässe reused)
    def make_sep(model_filename_only: str) -> _Separator:
        s = _Separator(
            output_dir=str(sep_base),
            output_format=args.format,
            model_file_dir=str(models_dir),
        )
        # WICHTIG: nur Dateiname (kein Pfad)
        s.load_model(model_filename=model_filename_only)
        return s

    try:
        sep_v = make_sep(args.vocals_model)
        sep_d = make_sep(args.drums_model)
        sep_b = make_sep(args.bass_model)
    except Exception as e:
        print(
            "\n[Fehler] Modell konnte nicht geladen werden.\n"
            "Nutze *exakt* die Namen aus `audio-separator --list_models` "
            "und übergib NUR den Dateinamen (z. B. UVR_MDXNET_KARA_2.onnx).\n"
            f"Details: {e}\n"
            "Tipp: Modelle anzeigen mit:\n  audio-separator --list_models | grep -i 'voc\\|drum\\|bass'\n"
        )
        shutil.rmtree(work_root, ignore_errors=True)
        return 2

    try:
        # Pass 1 (Default)
        stems1, cov1 = _process_pass(
            mix, sr, L, O, offset=0, fade_in=fade_in, fade_out=fade_out,
            sep_v=sep_v, sep_d=sep_d, sep_b=sep_b, sep_base=sep_base,
            temp_root=work_root, align=args.align, nnls=args.nnls, pass_name="P1"
        )

        if args.twopass == "off":
            final = stems1
        else:
            # Pass 2: halber Schritt versetzt
            step = L - O
            offset = max(1, step // 2)
            stems2, cov2 = _process_pass(
                mix, sr, L, O, offset=offset, fade_in=fade_in, fade_out=fade_out,
                sep_v=sep_v, sep_d=sep_d, sep_b=sep_b, sep_base=sep_base,
                temp_root=work_root, align=args.align, nnls=args.nnls, pass_name="P2"
            )
            final = {}
            denom = (cov1.astype(np.float32) + cov2.astype(np.float32))
            denom2 = np.maximum(denom, 1.0)[:, None]
            for k in ("vocals", "drums", "bass"):
                s = np.zeros_like(mix, dtype=np.float32)
                s += stems1[k] * cov1[:, None]
                s += stems2[k] * cov2[:, None]
                s /= denom2
                final[k] = s

        # Other = Mix − (V + D + B)
        other = (mix - (final["vocals"] + final["drums"] + final["bass"])).astype(np.float32)

        # Speichern
        def save(name: str, arr: np.ndarray):
            _write_audio(outdir / f"{in_path.stem}_{name}.{args.format}", arr, sr)

        save("Vocals", final["vocals"])
        save("Drums",  final["drums"])
        save("Bass",   final["bass"])
        save("Other",  other)

        # Optionaler Bass-Rescue
        if args.bass_rescue:
            try:
                f, g = args.bass_rescue.split(",")
                freq, gain = float(f), float(g)
                bass_file = outdir / f"{in_path.stem}_Bass.{args.format}"
                fixed = _bass_rescue_ffmpeg(in_path, bass_file, freq, gain)
                print(f"Bass-Rescue → {fixed.name}")
            except Exception as e:
                print(f"[WARN] Bass-Rescue fehlgeschlagen: {e}")

        print("Fertig:", outdir)
        return 0

    finally:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
