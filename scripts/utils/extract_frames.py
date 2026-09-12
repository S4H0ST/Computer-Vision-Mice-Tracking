"""
Extrae frames muestreados de un video para aumentar el dataset de entrenamiento.
Salta N frames entre cada extraccion para evitar duplicados y reducir overfitting.

Uso:
    python scripts/utils/extract_frames.py [--video PATH] [--step N] [--quality Q]

Por defecto extrae 1 de cada 30 frames (~1 fps en video de 30fps).
"""

import argparse
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def extract_frames(video_path: Path, out_dir: Path, step: int = 30, quality: int = 92) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            fname = out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Video:   {video_path.name}  ({total} frames @ {fps:.1f} fps)")
    print(f"Paso:    1 de cada {step} frames")
    print(f"Guardados: {saved} frames  ->  {out_dir}")
    return saved


def main() -> None:
    default_video = PROJECT_ROOT / "media_original" / "videos" / "testRata4.mp4"
    default_out   = PROJECT_ROOT / "media_original" / "frames_testRata4"

    parser = argparse.ArgumentParser(description="Extrae frames muestreados de un video")
    parser.add_argument("--video",   type=Path, default=default_video,
                        help="Ruta al video de entrada")
    parser.add_argument("--out",     type=Path, default=default_out,
                        help="Carpeta de salida para los frames")
    parser.add_argument("--step",    type=int,  default=30,
                        help="Extraer 1 frame de cada N (default: 30)")
    parser.add_argument("--quality", type=int,  default=92,
                        help="Calidad JPEG 1-100 (default: 92)")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: no se encontro el video en {args.video}")
        return

    extract_frames(args.video, args.out, step=args.step, quality=args.quality)


if __name__ == "__main__":
    main()
