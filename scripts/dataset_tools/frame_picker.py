"""
frame_picker.py — Selector interactivo de frames para etiquetado YOLO-Pose.

Modo normal (por defecto):
  Guarda frames en subcarpetas por clase dentro de --out.
  Útil para construir un dataset local con anotaciones propias.

Modo plano (--flat):
  Guarda todos los frames en una carpeta única, sin subcarpetas.
  Ideal para subir a Roboflow: una carpeta limpia, etiquetas allí.
  Con 1-5 el nombre de clase se añade como prefijo al fichero.

Controles:
  D / →        Avanzar 1 frame (en pausa)
  A / ←        Retroceder 1 frame (en pausa)
  ESPACIO      Play/Pausa (modo normal) | Guardar sin clase (modo --flat)
  P            Play / Pausa
  1            Guardar → rearing
  2            Guardar → grooming
  3            Guardar → horizontal
  4            Guardar → climbing
  5            Guardar → head_dipping
  Q            Salir

Uso:
    python frame_picker.py --video ruta/video.mp4 --out carpeta_salida
    python frame_picker.py --video ruta/video.mp4 --out carpeta_salida --flat
"""

import argparse
import cv2
from pathlib import Path

CLASSES = {
    ord('1'): "rearing",
    ord('2'): "grooming",
    ord('3'): "horizontal",
    ord('4'): "climbing",
    ord('5'): "head_dipping",
}

BLUR_THRESHOLD = 100.0


def blur_score(frame) -> float:
    """Varianza del Laplaciano: cuanto más alto, más nítida."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Selector interactivo de frames.")
    parser.add_argument("--video", required=True, help="Ruta al vídeo fuente (.mp4)")
    parser.add_argument("--out",   required=True, help="Carpeta de salida")
    parser.add_argument("--flat",  action="store_true",
                        help="Modo plano: carpeta única sin subcarpetas (ideal para Roboflow)")
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    out_dir    = Path(args.out)

    if not video_path.exists():
        print(f"[X] No encuentro el vídeo: {video_path}")
        return

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[!] AVISO: '{out_dir}' ya tiene contenido.")
        resp = input("    ¿Continuar igualmente? [s/N]: ").strip().lower()
        if resp != 's':
            print("[*] Abortado.")
            return

    if args.flat:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        for cls_name in CLASSES.values():
            (out_dir / cls_name).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

    win = "SELECTOR DE FRAMES"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1100, 750)

    playing     = False
    frame_idx   = 0
    saved_count = 0
    counts      = {name: 0 for name in CLASSES.values()}

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        return f if ret else None

    frame = read_frame(frame_idx)

    while frame is not None:
        display   = frame.copy()
        score     = blur_score(frame)
        sharpness = "NITIDA" if score > BLUR_THRESHOLD else "BORROSA"
        mode_tag  = "[→ROBOFLOW]" if args.flat else "[CLASES]"

        # ── barra de estado ───────────────────────────────────────────────
        cv2.rectangle(display, (0, 0), (1050, 95), (0, 0, 0), -1)
        cv2.putText(
            display,
            f"{mode_tag}  Frame {frame_idx}/{total_frames}  "
            f"{'PLAY' if playing else 'PAUSA'}  "
            f"Nitidez:{score:.0f}({sharpness})  Guardados:{saved_count}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        if args.flat:
            hint = ("ESPACIO=guardar | 1=rear 2=groom 3=horiz 4=climb 5=dip "
                    "| A/D=frame | P=play | Q=salir")
        else:
            abbr = {"rearing": "rear", "grooming": "groom", "horizontal": "horiz",
                    "climbing": "climb", "head_dipping": "dip"}
            cls_str = "  ".join(f"{abbr[n]}:{counts[n]}" for n in CLASSES.values())
            hint = f"1-5=guardar  |  {cls_str}  |  A/D=frame  ESPACIO=play  Q=salir"
        cv2.putText(display, hint, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(win, display)
        delay = max(1, int(1000 / fps)) if playing else 20
        key   = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('p'):
            playing = not playing

        elif key == ord(' '):
            if args.flat:
                fname = f"{video_path.stem}_f{frame_idx:05d}_blur{int(score)}.jpg"
                cv2.imwrite(str(out_dir / fname), frame)
                saved_count += 1
                print(f"[+] {fname}")
            else:
                playing = not playing

        elif key in (ord('d'), 83) and not playing:  # 83 = flecha derecha
            frame_idx = min(frame_idx + 1, total_frames - 1)
            frame     = read_frame(frame_idx)

        elif key in (ord('a'), 81) and not playing:  # 81 = flecha izquierda
            frame_idx = max(frame_idx - 1, 0)
            frame     = read_frame(frame_idx)

        elif key in CLASSES:
            cls_name = CLASSES[key]
            if args.flat:
                fname = f"{cls_name}_{video_path.stem}_f{frame_idx:05d}_blur{int(score)}.jpg"
                dst   = out_dir / fname
            else:
                fname = f"{video_path.stem}_f{frame_idx:05d}_blur{int(score)}.jpg"
                dst   = out_dir / cls_name / fname
            cv2.imwrite(str(dst), frame)
            saved_count         += 1
            counts[cls_name]    += 1
            print(f"[+] ({cls_name}) {dst.name}")

        if playing:
            frame_idx = min(frame_idx + 1, total_frames - 1)
            frame     = read_frame(frame_idx)
            if frame_idx == total_frames - 1:
                playing = False

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n--- RESUMEN — {saved_count} frames guardados ---")
    for cls_name, n in counts.items():
        if n > 0:
            print(f"  {cls_name}: {n}")
    unlabeled = saved_count - sum(counts.values())
    if unlabeled > 0:
        print(f"  sin clase: {unlabeled}")


if __name__ == "__main__":
    main()
