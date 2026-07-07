# scripts/dataset_tools/main_dataFrames_config.py

import sys
from pathlib import Path
from helpers.configuracion import paths, data_cfg
from dataset_tools.dataset_builder import DatasetBuilder
from dataset_tools.frame_picker import main as _pick_frames


def main():
    paths.check_dirs()

    while True:
        print("\n" + "=" * 52)
        print(" ;) RAT DATASET TOOLKIT")
        print("=" * 52)
        print("1. Seleccionar frames de vídeo  →  subir a Roboflow")
        print("2. Compilar Dataset Final  (split train/valid local)")
        print("3. Salir")
        print()
        print("[i] Flujo Roboflow:")
        print("    Opción 1 → sube la carpeta a Roboflow → etiqueta")
        print("    → exporta YOLOv8 Pose → descomprime el .zip en datasets/")
        print("    → Opción 2 del menú PRINCIPAL para entrenar YOLO.")

        opt = input("\n[?] Opción: ").strip()

        if opt == "1":
            _launch_picker()
        elif opt == "2":
            print(f"\n[i] Recopilando desde: {paths.raw_images}")
            DatasetBuilder(data_cfg).run()
        elif opt == "3":
            sys.exit()
        else:
            print("[X] Opción inválida.")


def _launch_picker():
    """Lista vídeos disponibles y lanza frame_picker en modo plano (Roboflow)."""
    videos_dir = paths.raw_videos
    if not videos_dir.exists():
        print(f"[X] No encuentro la carpeta de vídeos: {videos_dir}")
        return

    videos = sorted(videos_dir.glob("*.mp4")) + sorted(videos_dir.glob("*.MP4"))
    if not videos:
        print(f"[X] No hay archivos .mp4 en: {videos_dir}")
        return

    print("\n[i] Vídeos disponibles:")
    for i, v in enumerate(videos, 1):
        mb = v.stat().st_size / 1e6
        print(f"    {i}.  {v.name}  ({mb:.1f} MB)")

    sel = input("\n[?] Número de vídeo (0 para cancelar): ").strip()
    if not sel.isdigit() or int(sel) == 0 or int(sel) > len(videos):
        print("[*] Cancelado.")
        return

    video = videos[int(sel) - 1]

    # Proponer una carpeta nueva junto a frames_original/
    default_out = paths.raw_videos.parent / f"roboflow_{video.stem}"
    print(f"\n[i] Carpeta de salida sugerida: {default_out}")
    custom = input("[?] ENTER para aceptar, o escribe otra ruta: ").strip()
    out_dir = Path(custom) if custom else default_out

    print(f"\n  Controles dentro del selector:")
    print(f"  ESPACIO  →  guardar frame (sin clase)")
    print(f"  1        →  rat_climbing      (escalando)")
    print(f"  2        →  rat_grooming      (acicalándose)")
    print(f"  3        →  rat_head_dipping  (asomando la cabeza)")
    print(f"  4        →  rat_horizontal    (explorando el suelo)")
    print(f"  5        →  rat_rearing       (empinada / de pie)")
    print(f"  P        →  play / pausa")
    print(f"  A / D    →  frame anterior / siguiente (en pausa)")
    print(f"  Q        →  salir\n")

    _pick_frames(["--video", str(video), "--out", str(out_dir), "--flat"])

    print(f"\n[✓] Frames guardados en: {out_dir}")
    print(f"[→] Sube esa carpeta a Roboflow, etiqueta con bboxes + 3 keypoints,")
    print(f"    exporta en formato 'YOLOv8 Pose' y descomprime el .zip en:")
    print(f"    {paths.root / 'datasets'}/")


if __name__ == "__main__":
    main()
