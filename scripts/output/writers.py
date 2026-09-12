"""
Escritura de video anotado y CSV de resultados frame a frame.

Clases:
    VideoOutput — gestiona uno o dos cv2.VideoWriter (overlay + limpio opcional).
    CsvOutput   — gestiona apertura, escritura por fila y cierre del CSV de deteccion.
"""

import csv
import cv2
from pathlib import Path


class VideoOutput:
    """
    Envuelve cv2.VideoWriter con soporte para salida dual (overlay + limpio).

    dual_output=True genera un segundo fichero <stem>_limpio.mp4 sin zonas calibradas.
    """

    def __init__(self, video_path: Path, fps: float, w: int, h: int,
                 dual_output: bool = False) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._main = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        self._clean: cv2.VideoWriter | None = None
        self._clean_path: Path | None = None

        if dual_output:
            self._clean_path = video_path.with_name(video_path.stem + "_limpio.mp4")
            self._clean = cv2.VideoWriter(str(self._clean_path), fourcc, fps, (w, h))

    @property
    def clean_path(self) -> Path | None:
        return self._clean_path

    def write(self, img_main, img_clean=None) -> None:
        self._main.write(img_main)
        if self._clean is not None and img_clean is not None:
            self._clean.write(img_clean)

    def release(self) -> None:
        self._main.release()
        if self._clean is not None:
            self._clean.release()


class CsvOutput:
    """
    Gestiona el CSV de resultados frame a frame.

    Columnas: frame, time_s, yolo_label, final_label,
              x1, y1, x2, y2, snout_x, snout_y, speed, tail_x, tail_y, hole_idx
    """

    HEADER = ["frame", "time_s", "yolo_label", "final_label",
               "x1", "y1", "x2", "y2",
               "snout_x", "snout_y", "speed", "tail_x", "tail_y", "hole_idx"]

    def __init__(self, csv_path: Path) -> None:
        self._path = csv_path
        self._f = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        self._writer.writerow(self.HEADER)

    @property
    def path(self) -> Path:
        return self._path

    def write_row(self, frame_idx: int, fps: float,
                  yolo_label: str, final_label: str,
                  box, snout_kp, tail_kp, speed: float,
                  hole_idx: int = -1) -> None:
        rx1, ry1, rx2, ry2 = map(int, box)
        snout_x = float(snout_kp[0]) if snout_kp is not None else -1.0
        snout_y = float(snout_kp[1]) if snout_kp is not None else -1.0
        tail_x  = float(tail_kp[0])  if tail_kp  is not None else -1.0
        tail_y  = float(tail_kp[1])  if tail_kp  is not None else -1.0
        self._writer.writerow([
            frame_idx, f"{frame_idx / fps:.2f}",
            yolo_label, final_label,
            rx1, ry1, rx2, ry2,
            f"{snout_x:.1f}", f"{snout_y:.1f}",
            f"{speed:.3f}",
            f"{tail_x:.1f}", f"{tail_y:.1f}",
            hole_idx,
        ])

    def close(self) -> None:
        self._f.close()
