from pathlib import Path
import argparse, sys, shutil, subprocess

def have_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def run_demucs(input_path: Path, outdir: Path, model_name: str,
               segment: int, overlap: float, shifts: int, device: str):
    """Ruft die Demucs-CLI mit den gewünschten Inferenz-Parametern auf."""
    demucs_bin = shutil.which("demucs")
    demucs_cmd = [demucs_bin] if demucs_bin else [sys.executable, "-m", "demucs"]
    cmd = demucs_cmd + [
        "-n", model_name,
        "-o", str(outdir),
        "-d", device,
        "--segment", str(segment),
        "--overlap", str(overlap),
        "--shifts", str(shifts),
        str(input_path),
    ]
    print(">>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Standalone Audio Splitter (MDX/Demucs)")
    ap.add_argument("input", help="Audio/Video-Datei (z.B. .wav, .flac, .mp3)")
    ap.add_argument("-o", "--outdir", default="stems_out", help="Ausgabeordner")
    ap.add_argument("--format", default="wav", help="Ausgabeformat (für MDX/ONNX)")
    ap.add_argument("--model-filename", default=None,
                    help="Explizites Modell (z.B. UVR_MDXNET_KARA_2.onnx oder htdemucs*.yaml)")

    # Demucs-Preset & Inferenz-Feintuning
    ap.add_argument("--demucs", choices=["fast", "mmi", "ft"],
                    help="Demucs-4-Stem: fast=htdemucs, mmi=hdemucs_mmi, ft=htdemucs_ft")
    ap.add_argument("--segment", type=int, default=45, help="Segmentlänge in s (Demucs)")
    ap.add_argument("--overlap", type=float, default=0.5, help="Überlappung 0..1 (Demucs)")
    ap.add_argument("--shifts", type=int, default=2, help="Test-Time Augmentation Shifts (Demucs)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Rechen-Device für Demucs")

    args = ap.parse_args()
    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Entscheiden: Demucs (YAML/Preset) oder MDX/ONNX via audio-separator?
    demucs_model = None
    if args.model_filename and args.model_filename.lower().endswith(".yaml"):
        demucs_model = Path(args.model_filename).stem
    elif args.demucs:
        mapping = {"fast": "htdemucs", "mmi": "hdemucs_mmi", "ft": "htdemucs_ft"}
        demucs_model = mapping[args.demucs]

    if demucs_model:
        # Gerätauswahl
        device = args.device
        if device == "auto":
            device = "cuda" if have_cuda() else "cpu"

        run_demucs(
            input_path=inp,
            outdir=outdir,
            model_name=demucs_model,
            segment=args.segment,
            overlap=args.overlap,
            shifts=args.shifts,
            device=device,
        )
        print(f"Demucs-Output liegt unter: {outdir}/separated/{demucs_model}/{inp.stem}")
        return 0

    # Fallback: MDX/ONNX über audio-separator (ohne die Demucs-Flags)
    from audio_separator.separator import Separator
    sep = Separator(output_dir=str(outdir), output_format=args.format)
    if args.model_filename:
        sep.load_model(model_filename=args.model_filename)
    else:
        sep.load_model()
    out_files = sep.separate(str(inp))
    for f in (out_files if isinstance(out_files, (list, tuple)) else [out_files]):
        print("OK ->", f)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
