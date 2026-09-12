"""
Generador de estadisticas post-deteccion para experimentos Open Field Test (Holeboard).

Uso:
    from utils.stats_generator import StatsGenerator
    StatsGenerator(csv_path, coords_json=paths.coords_json).generate(output_dir)

Salidas (en output_dir/):
    trajectory_<stem>.png  - trayectoria del snout sobre la plantilla de la caja.
    heatmap_<stem>.png     - mapa de calor: el color cambia segun el tiempo parado.
    stats_<stem>.xlsx      - hoja Comportamiento + hoja Metricas OFT.

Clases:
    StatsGenerator - carga un CSV de deteccion y genera todas las salidas.

Requiere openpyxl para el Excel (pip install openpyxl).
Si no esta instalado se generan solo las imagenes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter
from itertools import groupby

import numpy as np
import cv2

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.chart import BarChart, Reference
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False


class StatsGenerator:
    """Genera imagen de trayectoria y Excel de estadisticas a partir de un CSV de deteccion."""

    CANVAS_SIZE:   int = 700
    CANVAS_MARGIN: int = 50

    def __init__(self, csv_path: Path, coords_json: Path | None = None) -> None:
        self.csv_path    = Path(csv_path)
        self.coords_json = Path(coords_json) if coords_json else None

        self.rows:         list[dict] = []
        self.fps:          float      = 30.0
        self.inner_limits: dict | None = None
        self.outer_limits: dict | None = None
        self.limits_center: dict | None = None
        self.holes:        list[tuple] = []
        self.hole_radius:  int = 20
        self.px_per_cm:    float | None = None

        self._load()

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        if len(self.rows) >= 2:
            try:
                frames = int(self.rows[-1]["frame"])
                t_s    = float(self.rows[-1]["time_s"])
                if t_s > 0 and frames > 0:
                    self.fps = frames / t_s
            except (KeyError, ValueError):
                pass

        if self.coords_json and self.coords_json.exists():
            with open(self.coords_json) as f:
                data = json.load(f)
            self.inner_limits  = data.get("limits_inner")
            self.outer_limits  = data.get("limits_outer")
            self.limits_center = data.get("limits_center")
            self.holes         = [tuple(h) for h in data.get("holes", [])]
            self.hole_radius   = data.get("hole_radius", 20)

            box_w_cm = data.get("box_width_cm")
            box_h_cm = data.get("box_height_cm")
            if box_w_cm and box_h_cm and self.outer_limits:
                lim = self.outer_limits
                px_w = lim["x_max"] - lim["x_min"]
                px_h = lim["y_max"] - lim["y_min"]
                self.px_per_cm = ((px_w / box_w_cm) + (px_h / box_h_cm)) / 2

    # ------------------------------------------------------------------
    # Punto de entrada publico
    # ------------------------------------------------------------------

    def generate(self, output_dir: Path) -> None:
        """Genera todos los archivos de estadisticas en output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.csv_path.stem

        traj_path = output_dir / f"trajectory_{stem}.png"
        self._generate_trajectory(traj_path)
        print(f"  [stats] Trayectoria: {traj_path.name}")

        heat_path = output_dir / f"heatmap_{stem}.png"
        self._generate_heatmap(heat_path)
        print(f"  [stats] Mapa de calor: {heat_path.name}")

        if _HAS_OPENPYXL:
            xl_path = output_dir / f"stats_{stem}.xlsx"
            try:
                self._generate_excel(xl_path)
                print(f"  [stats] Excel: {xl_path.name}")
            except PermissionError:
                print(f"  [stats] AVISO: {xl_path.name} esta abierto en otra aplicacion.")
                print(f"          Cierra el archivo Excel y vuelve a ejecutar.")
        else:
            print("  [stats] AVISO: openpyxl no instalado — omitiendo Excel.")
            print("          Instala con:  pip install openpyxl")

    # ------------------------------------------------------------------
    # Utilidades de coordenadas
    # ------------------------------------------------------------------

    def _canvas_mapping(self) -> tuple[float, float, float, float]:
        """Devuelve (x_min, y_min, x_scale, y_scale) para video -> canvas."""
        m    = self.CANVAS_MARGIN
        area = self.CANVAS_SIZE - 2 * m

        if self.inner_limits:
            lim     = self.inner_limits
            x_min   = float(lim["x_min"])
            y_min   = float(lim["y_min"])
            x_scale = area / max(lim["x_max"] - x_min, 1)
            y_scale = area / max(lim["y_max"] - y_min, 1)
        else:
            xs = [float(r["tail_x"]) for r in self.rows if float(r.get("tail_x", -1)) > 0]
            ys = [float(r["tail_y"]) for r in self.rows if float(r.get("tail_y", -1)) > 0]
            x_min = min(xs) if xs else 0.0
            y_min = min(ys) if ys else 0.0
            x_scale = area / max((max(xs) if xs else 1.0) - x_min, 1)
            y_scale = area / max((max(ys) if ys else 1.0) - y_min, 1)

        return x_min, y_min, x_scale, y_scale

    def _to_canvas(self, x: float, y: float,
                   x_min: float, y_min: float,
                   x_scale: float, y_scale: float) -> tuple[int, int]:
        m  = self.CANVAS_MARGIN
        sz = self.CANVAS_SIZE
        cx = max(m, min(sz - m, int(m + (x - x_min) * x_scale)))
        cy = max(m, min(sz - m, int(m + (y - y_min) * y_scale)))
        return cx, cy

    # Colores BGR por etiqueta para la trayectoria (mismo esquema que el video anotado)
    # Orden: de menor a mayor actividad/relevancia conductual en OFT
    _TRAJ_COLORS: list[tuple[str, str, tuple[int, int, int]]] = [
        ("immobile",         "Inmovil",      (180, 180, 180)),
        ("walking",          "Caminando",    (0,   255, 255)),
        ("sniffing",         "Olfateando",   (0,   200, 255)),
        ("rat_head_dipping", "Agujero",      (0,   165, 255)),
        ("rat_rearing",      "Erguido",      (0,   255,   0)),
        ("rat_climbing",     "Escalando",    (255,   0, 255)),
        ("rat_grooming",     "Acicalamiento",(180, 255, 180)),
    ]
    _DEFAULT_TRAJ_COLOR: tuple[int, int, int] = (120, 120, 120)

    @classmethod
    def _traj_color(cls, label: str) -> tuple[int, int, int]:
        for key, _, color in cls._TRAJ_COLORS:
            if key in label:
                return color
        return cls._DEFAULT_TRAJ_COLOR

    # ------------------------------------------------------------------
    # Imagen de trayectoria
    # ------------------------------------------------------------------

    def _generate_trajectory(self, out_path: Path) -> None:
        sz     = self.CANVAS_SIZE
        m      = self.CANVAS_MARGIN
        leg_h  = 52   # franja extra debajo del canvas para la leyenda

        img = np.full((sz + leg_h, sz, 3), 255, dtype=np.uint8)

        step = (sz - 2 * m) // 12
        for i in range(1, 12):
            o = m + i * step
            cv2.line(img, (o, m),     (o, sz - m), (235, 235, 235), 1)
            cv2.line(img, (m, o), (sz - m, o),     (235, 235, 235), 1)

        cv2.rectangle(img, (m, m), (sz - m, sz - m), (60, 60, 60), 2)

        x_min, y_min, x_scale, y_scale = self._canvas_mapping()

        # Borde central — linea negra suave para separar zona central de zona periferica
        if self.limits_center:
            lim = self.limits_center
            p1 = self._to_canvas(lim["x_min"], lim["y_min"], x_min, y_min, x_scale, y_scale)
            p2 = self._to_canvas(lim["x_max"], lim["y_max"], x_min, y_min, x_scale, y_scale)
            cv2.rectangle(img, p1, p2, (0, 0, 0), 1, cv2.LINE_AA)

        # Agujeros
        if self.holes:
            r_canvas = max(6, int(self.hole_radius * min(x_scale, y_scale)))
            for hx, hy in self.holes:
                cx, cy = self._to_canvas(hx, hy, x_min, y_min, x_scale, y_scale)
                cv2.circle(img, (cx, cy), r_canvas, (80, 80, 80), 2)

        # Trayectoria del snout — cada segmento coloreado segun la etiqueta en ese frame
        prev_pt:    tuple[int, int] | None = None
        prev_label: str                    = ""
        for row in self.rows:
            try:
                sx = float(row.get("snout_x", -1))
                sy = float(row.get("snout_y", -1))
            except ValueError:
                prev_pt = None
                continue
            if sx < 0 or sy < 0:
                prev_pt = None
                continue
            lbl   = row.get("final_label", "")
            color = self._traj_color(lbl)
            pt    = self._to_canvas(sx, sy, x_min, y_min, x_scale, y_scale)
            if prev_pt is not None:
                cv2.line(img, prev_pt, pt, color, 1)
            prev_pt    = pt
            prev_label = lbl

        self._draw_traj_legend(img, sz, m, leg_h)
        cv2.imwrite(str(out_path), img)

    def _draw_traj_legend(self, img: np.ndarray, sz: int, m: int, leg_h: int = 52) -> None:
        """Leyenda de comportamientos en franja horizontal debajo del canvas."""
        box_s  = 12
        font   = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.38
        fthick = 1
        gap    = 6    # espacio entre cuadrado y texto
        sep    = 18   # espacio entre un item y el siguiente

        labels = self._TRAJ_COLORS
        # Calcular ancho de cada item: box + gap + text_w + sep
        item_widths = [
            box_s + gap + cv2.getTextSize(name, font, fscale, fthick)[0][0] + sep
            for _, name, _ in labels
        ]
        total_w = sum(item_widths)

        # Centrar horizontalmente; franja comienza en y=sz
        x = (sz - total_w) // 2
        cy = sz + (leg_h - box_s) // 2   # centra verticalmente en la franja

        # Linea separadora tenue entre arena y franja
        cv2.line(img, (m, sz + 1), (sz - m, sz + 1), (210, 210, 210), 1)

        for (_, name, color), iw in zip(labels, item_widths):
            # Cuadrado de color con borde
            cv2.rectangle(img, (x, cy), (x + box_s, cy + box_s), color, -1)
            cv2.rectangle(img, (x, cy), (x + box_s, cy + box_s), (100, 100, 100), 1)
            # Texto
            cv2.putText(img, name, (x + box_s + gap, cy + box_s - 1),
                        font, fscale, (40, 40, 40), fthick, cv2.LINE_AA)
            x += iw

    # ------------------------------------------------------------------
    # Mapa de calor
    # ------------------------------------------------------------------

    def _generate_heatmap(self, out_path: Path) -> None:
        """
        Genera un mapa de calor de densidad de presencia y lo guarda en out_path.

        Algoritmo (acumulacion + desenfoque gaussiano, 3 pasos):
          1. Por cada frame con snout detectado se suma 1 en la posicion (x,y)
             del snout sobre una matriz float32 del tamano del canvas.
          2. cv2.GaussianBlur (sigma=18 px) convierte los puntos en manchas
             suaves que representan la zona de influencia del raton.
          3. La matriz normalizada [0,1] se convierte a uint8 y se aplica
             cv2.COLORMAP_JET: azul oscuro = zona poco visitada,
             rojo = hotspot donde el raton paso mas tiempo.

        El resultado es un heatmap de densidad puro, sin lineas de trayectoria.

        Nota historica (guardada para el profesor):
          La primera implementacion coloreaba cada segmento de trayectoria
          segun un contador de inmovilidad (dwell_frames). El problema es que
          el raton esta en movimiento la mayor parte del tiempo, por lo que
          el contador casi nunca se acumulaba y el mapa salia todo azul sin
          variacion perceptible. El archivo de esa prueba queda en
          outputs/detections/testRata4_20260903_211004/stats/ como referencia.
          La acumulacion de presencia es mas sencilla, robusta e informativa.
        """
        sz = self.CANVAS_SIZE
        m  = self.CANVAS_MARGIN

        x_min, y_min, x_scale, y_scale = self._canvas_mapping()

        accum = np.zeros((sz, sz), dtype=np.float32)
        for row in self.rows:
            try:
                sx = float(row.get("snout_x", -1))
                sy = float(row.get("snout_y", -1))
            except ValueError:
                continue
            if sx < 0 or sy < 0:
                continue
            cx, cy = self._to_canvas(sx, sy, x_min, y_min, x_scale, y_scale)
            accum[cy, cx] += 1.0

        accum = cv2.GaussianBlur(accum, (0, 0), sigmaX=12)

        if accum.max() > 0:
            accum = accum / accum.max()
        img = cv2.applyColorMap((accum * 255).astype(np.uint8), cv2.COLORMAP_JET)

        img[:m,    :]  = (255, 255, 255)
        img[sz - m:, :] = (255, 255, 255)
        img[:,    :m]  = (255, 255, 255)
        img[:, sz - m:] = (255, 255, 255)

        cv2.rectangle(img, (m, m), (sz - m, sz - m), (40, 40, 40), 2)

        if self.limits_center:
            lim = self.limits_center
            p1 = self._to_canvas(lim["x_min"], lim["y_min"], x_min, y_min, x_scale, y_scale)
            p2 = self._to_canvas(lim["x_max"], lim["y_max"], x_min, y_min, x_scale, y_scale)
            cv2.rectangle(img, p1, p2, (255, 255, 255), 1)

        if self.holes:
            r_canvas = max(6, int(self.hole_radius * min(x_scale, y_scale)))
            for hx, hy in self.holes:
                cx, cy = self._to_canvas(hx, hy, x_min, y_min, x_scale, y_scale)
                cv2.circle(img, (cx, cy), r_canvas, (255, 255, 255), 2)

        self._draw_heatmap_legend(img, sz, m)
        cv2.imwrite(str(out_path), img)

    def _draw_heatmap_legend(self, img: np.ndarray, sz: int, m: int) -> None:
        """Leyenda de gradiente COLORMAP_JET en el margen blanco derecho (fuera del canvas)."""
        legend_h = 80
        legend_w = 12
        lx = sz - m + 4   # dentro del margen blanco derecho, a la derecha del borde del canvas
        ly = m + 10

        for i in range(legend_h):
            val = int((legend_h - 1 - i) * 255 / (legend_h - 1))
            color_px  = np.array([[[val]]], dtype=np.uint8)
            color_bgr = cv2.applyColorMap(color_px, cv2.COLORMAP_JET)[0, 0]
            cv2.line(img, (lx, ly + i), (lx + legend_w, ly + i),
                     (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])), 1)

        cv2.rectangle(img, (lx, ly), (lx + legend_w, ly + legend_h), (80, 80, 80), 1)
        cv2.putText(img, "alto",  (lx + legend_w + 3, ly + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (40, 40, 40), 1)
        cv2.putText(img, "bajo",  (lx + legend_w + 3, ly + legend_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (40, 40, 40), 1)

    # ------------------------------------------------------------------
    # Utilidades Excel
    # ------------------------------------------------------------------

    def _count_bouts(self) -> dict[str, int]:
        """Cuenta bouts (rachas consecutivas) por etiqueta sobre todas las filas."""
        bouts: dict[str, int] = {}
        for lbl, _ in groupby(self.rows, key=lambda r: r.get("final_label", "")):
            bouts[lbl] = bouts.get(lbl, 0) + 1
        return bouts

    def _bouts_for_label_in(self, rows_subset: list[dict], label: str) -> int:
        """Cuenta bouts de una etiqueta concreta en un subconjunto de filas."""
        return sum(
            1 for lbl, _ in groupby(rows_subset, key=lambda r: r.get("final_label", ""))
            if lbl == label
        )

    def _count_hole_usage(self) -> tuple[dict[int, int], dict[int, int]]:
        """
        Cuenta frames y bouts de head dipping por agujero.
        Devuelve (frames_per_hole, bouts_per_hole) donde la clave es el indice (0-3).
        """
        n_holes = len(self.holes)
        frames_ph: dict[int, int] = {i: 0 for i in range(n_holes)}
        bouts_ph:  dict[int, int] = {i: 0 for i in range(n_holes)}
        prev_hi = -1

        for r in self.rows:
            if r.get("final_label") != "rat_head_dipping":
                prev_hi = -1
                continue
            try:
                hi = int(r.get("hole_idx", -1))
            except (ValueError, TypeError):
                hi = -1
            if hi < 0 or hi >= n_holes:
                prev_hi = -1
                continue
            frames_ph[hi] += 1
            if hi != prev_hi:
                bouts_ph[hi] += 1
            prev_hi = hi

        return frames_ph, bouts_ph

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------

    def _generate_excel(self, out_path: Path) -> None:
        wb   = openpyxl.Workbook()
        rows = self.rows
        n    = len(rows)

        if n == 0:
            wb.save(str(out_path))
            return

        duration_s = float(rows[-1]["time_s"])
        mins       = duration_s / 60.0
        label_cnt  = Counter(r.get("final_label", "") for r in rows)
        bouts_cnt  = self._count_bouts()

        # ---- Hoja 1: Comportamiento (presupuesto de tiempo) ---------- #
        ws = wb.active
        ws.title = "Comportamiento"

        hdr_fill = PatternFill("solid", fgColor="1F4E79")
        hdr_font = Font(bold=True, color="FFFFFF")

        for col, h in enumerate(
            ["Comportamiento", "Frames", "Duracion (s)", "% Tiempo",
             "Bouts", "Dur. media por bout (s)"], 1
        ):
            c = ws.cell(row=1, column=col, value=h)
            c.font = hdr_font
            c.fill = hdr_fill
            c.alignment = Alignment(horizontal="center")

        all_labels = sorted(label_cnt, key=lambda l: -label_cnt[l])
        for i, lbl in enumerate(all_labels, 2):
            frames   = label_cnt[lbl]
            dur      = frames / self.fps
            pct      = frames / n * 100
            bouts    = bouts_cnt.get(lbl, 0)
            avg_bout = dur / bouts if bouts > 0 else 0.0
            for col, val in enumerate(
                [lbl, frames, round(dur, 2), round(pct, 1), bouts, round(avg_bout, 2)], 1
            ):
                ws.cell(row=i, column=col, value=val)

        ws.column_dimensions["A"].width = 24
        for letter in ["B", "C", "D", "E", "F"]:
            ws.column_dimensions[letter].width = 18

        chart = BarChart()
        chart.type  = "col"
        chart.title = "Presupuesto de tiempo conductual"
        chart.y_axis.title = "% Tiempo"
        chart.x_axis.title = "Comportamiento"
        chart.style = 10
        chart.width = 20
        chart.height = 13
        n_lbl = len(all_labels)
        chart.add_data(Reference(ws, min_col=4, min_row=1, max_row=1 + n_lbl),
                       titles_from_data=True)
        chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=1 + n_lbl))
        ws.add_chart(chart, "H2")

        # ---- Hoja 2: Metricas OFT ------------------------------------ #
        ws2 = wb.create_sheet("Metricas OFT")
        ws2.column_dimensions["A"].width = 46
        ws2.column_dimensions["B"].width = 16
        ws2.column_dimensions["C"].width = 14

        sect_fill = PatternFill("solid", fgColor="2E4057")
        sect_font = Font(bold=True, color="FFFFFF")
        na        = "N/A (sin calibracion)"

        def _s(row: int, title: str) -> None:
            c = ws2.cell(row=row, column=1, value=title)
            c.font = sect_font
            c.fill = sect_fill

        def _m(row: int, label: str, value, unit: str = "") -> None:
            ws2.cell(row=row, column=1, value=label)
            ws2.cell(row=row, column=2, value=value)
            ws2.cell(row=row, column=3, value=unit)

        # --- Calculo de metricas previo --------------------------------

        # Distancia y velocidad (tail)
        tail_pts = [
            (float(r["tail_x"]), float(r["tail_y"]))
            for r in rows if float(r.get("tail_x", -1)) > 0
        ]
        total_dist = float(sum(
            np.linalg.norm(np.array(tail_pts[i + 1]) - np.array(tail_pts[i]))
            for i in range(len(tail_pts) - 1)
        )) if len(tail_pts) > 1 else 0.0
        avg_speed = round(total_dist / duration_s, 1) if duration_s > 0 else 0.0

        # Transiciones conductuales
        transitions = sum(
            1 for i in range(1, len(rows))
            if rows[i].get("final_label") != rows[i - 1].get("final_label")
        )
        trans_pm = round(transitions / mins, 1) if mins > 0 else 0.0

        # Head-dipping global
        dipping_b   = bouts_cnt.get("rat_head_dipping", 0)
        dipping_fr  = label_cnt.get("rat_head_dipping", 0)
        dipping_dur = round(dipping_fr / self.fps, 2)
        dipping_avg = round(dipping_dur / dipping_b, 2) if dipping_b > 0 else 0.0
        dipping_pm  = round(dipping_b / mins, 2) if mins > 0 else 0.0

        # Latencia al primer head-dip
        first_hd_t = next(
            (float(r["time_s"]) for r in rows if r.get("final_label") == "rat_head_dipping"),
            None
        )
        latencia_hd: float | str = round(first_hd_t, 1) if first_hd_t is not None else "No detectado"

        # Habituacion: head-dips por cuarto del video
        q  = n // 4
        quarters = [rows[i * q:(i + 1) * q] for i in range(3)] + [rows[3 * q:]]
        hd_q = [self._bouts_for_label_in(qr, "rat_head_dipping") for qr in quarters]

        # Head-dipping por agujero
        hole_frames, hole_bouts = self._count_hole_usage()

        # Grooming
        grooming_b   = bouts_cnt.get("rat_grooming", 0)
        grooming_fr  = label_cnt.get("rat_grooming", 0)
        grooming_dur = round(grooming_fr / self.fps, 2)
        grooming_avg = round(grooming_dur / grooming_b, 2) if grooming_b > 0 else 0.0
        grooming_pm  = round(grooming_b / mins, 2) if mins > 0 else 0.0

        # Climbing (thigmotaxis en holeboard)
        climbing_b   = bouts_cnt.get("rat_climbing", 0)
        climbing_pct = round(label_cnt.get("rat_climbing", 0) / n * 100, 1)

        # Zonificacion interior/exterior (inner_limits)
        if self.inner_limits:
            lim = self.inner_limits
            central_fr   = 0
            periph_fr    = 0
            imm_central  = 0
            imm_periph   = 0
            for r in rows:
                try:
                    cx = (float(r["x1"]) + float(r["x2"])) / 2
                    cy = (float(r["y1"]) + float(r["y2"])) / 2
                except (KeyError, ValueError):
                    continue
                in_inner = (lim["x_min"] <= cx <= lim["x_max"] and
                            lim["y_min"] <= cy <= lim["y_max"])
                lbl = r.get("final_label", "")
                if in_inner:
                    central_fr += 1
                    if lbl == "immobile":
                        imm_central += 1
                else:
                    periph_fr += 1
                    if lbl == "immobile":
                        imm_periph += 1
            central_pct  = round(central_fr  / n * 100, 1)
            periph_pct   = round(periph_fr   / n * 100, 1)
            imm_c_pct    = round(imm_central / n * 100, 1)
            imm_p_pct    = round(imm_periph  / n * 100, 1)
        else:
            central_pct = periph_pct = imm_c_pct = imm_p_pct = na

        # Zona central definida por el usuario (limits_center)
        if self.limits_center:
            lim_c = self.limits_center
            center_zone_fr  = 0
            periph_zone_fr  = 0
            for r in rows:
                try:
                    cx = (float(r["x1"]) + float(r["x2"])) / 2
                    cy = (float(r["y1"]) + float(r["y2"])) / 2
                except (KeyError, ValueError):
                    continue
                if (lim_c["x_min"] <= cx <= lim_c["x_max"] and
                        lim_c["y_min"] <= cy <= lim_c["y_max"]):
                    center_zone_fr += 1
                else:
                    periph_zone_fr += 1
            center_zone_pct = round(center_zone_fr / n * 100, 1)
            periph_zone_pct = round(periph_zone_fr / n * 100, 1)
        else:
            center_zone_pct = periph_zone_pct = na

        # Sniffing
        sniff_walk_fr = label_cnt.get("sniffing_walking", 0)
        sniff_imm_fr  = label_cnt.get("sniffing_immobile", 0)
        sniff_total   = sniff_walk_fr + sniff_imm_fr
        sniff_pct     = round(sniff_total / n * 100, 1)
        sniff_walk_pct = round(sniff_walk_fr / n * 100, 1)
        sniff_imm_pct  = round(sniff_imm_fr  / n * 100, 1)
        walk_fr        = label_cnt.get("walking", 0)
        denom_efic     = sniff_total + walk_fr
        efic_explor    = round(sniff_total / denom_efic * 100, 1) if denom_efic > 0 else 0.0

        # Inmovilidad general
        immobile_pct = round(label_cnt.get("immobile", 0) / n * 100, 1)
        walking_pct  = round(walk_fr / n * 100, 1)

        # --- Escritura de la hoja OFT ----------------------------------
        r = 1

        if self.px_per_cm:
            dist_val  = round(total_dist  / self.px_per_cm / 100, 2)
            dist_unit = "m"
            speed_val = round(avg_speed   / self.px_per_cm, 1)
            speed_unit = "cm/s"
        else:
            dist_val  = round(total_dist, 0)
            dist_unit = "px"
            speed_val = avg_speed
            speed_unit = "px/s"

        _s(r, "Duracion y Actividad General"); r += 1
        _m(r, "Duracion total del video",              round(duration_s, 1), "s");       r += 1
        _m(r, "Distancia total recorrida (tail)",      dist_val,             dist_unit); r += 1
        _m(r, "Velocidad media (tail)",                speed_val,            speed_unit); r += 1
        _m(r, "Walking (deambulacion)",                walking_pct,          "%");    r += 1
        _m(r, "Transiciones conductuales totales",     transitions,          "");     r += 1
        _m(r, "Tasa de transicion",                    trans_pm,             "trans/min"); r += 1

        r += 1
        _s(r, "Head-dipping (Indice Principal de Exploracion)"); r += 1
        _m(r, "N. de head-dips totales (bouts)",       dipping_b,    "bouts");    r += 1
        _m(r, "Head-dips por minuto",                  dipping_pm,   "bouts/min"); r += 1
        _m(r, "Latencia al primer head-dip",           latencia_hd,  "s");        r += 1
        _m(r, "Duracion total head-dipping",           dipping_dur,  "s");        r += 1
        _m(r, "Duracion media por head-dip",           dipping_avg,  "s/bout");   r += 1
        _m(r, "Habituacion — head-dips (1er cuarto)",  hd_q[0],      "bouts");    r += 1
        _m(r, "Habituacion — head-dips (2o cuarto)",   hd_q[1],      "bouts");    r += 1
        _m(r, "Habituacion — head-dips (3er cuarto)",  hd_q[2],      "bouts");    r += 1
        _m(r, "Habituacion — head-dips (4o cuarto)",   hd_q[3],      "bouts");    r += 1

        # Head-dipping por agujero individual
        if self.holes:
            r += 1
            _s(r, "Head-dipping por Agujero"); r += 1
            for i in range(len(self.holes)):
                fr_h  = hole_frames.get(i, 0)
                bt_h  = hole_bouts.get(i, 0)
                dur_h = round(fr_h / self.fps, 2)
                _m(r, f"  Agujero {i + 1} — bouts",       bt_h,  "bouts"); r += 1
                _m(r, f"  Agujero {i + 1} — duracion",    dur_h, "s");     r += 1
                _m(r, f"  Agujero {i + 1} — % tiempo HD", round(fr_h / max(dipping_fr, 1) * 100, 1), "%"); r += 1

        r += 1
        _s(r, "Sniffing (Exploracion Olfativa — conducta mayoritaria)"); r += 1
        _m(r, "Sniffing total",                        sniff_pct,      "%");  r += 1
        _m(r, "  Sniffing en movimiento",              sniff_walk_pct, "%");  r += 1
        _m(r, "  Sniffing inmovil",                    sniff_imm_pct,  "%");  r += 1
        _m(r, "Eficiencia exploratoria (sniff/sniff+walk)", efic_explor, "%"); r += 1

        r += 1
        _s(r, "Distribucion Espacial y Conducta de Pared"); r += 1
        _m(r, "Thigmotaxis — climbing (conducta de pared)", climbing_pct, "%");    r += 1
        _m(r, "Climbing (n. bouts)",                    climbing_b,     "bouts");  r += 1
        _m(r, "Tiempo zona interior (centroide)",        central_pct,    "%");     r += 1
        _m(r, "Tiempo zona periferica (centroide)",      periph_pct,     "%");     r += 1

        r += 1
        _s(r, "Zona Central (cuadrado usuario — deteccion forfox vs control)"); r += 1
        _m(r, "Tiempo en zona central (cuadrado)",      center_zone_pct, "%"); r += 1
        _m(r, "Tiempo en zona periferica (fuera cuad.)",periph_zone_pct, "%"); r += 1

        r += 1
        _s(r, "Inactividad y Estado Emocional"); r += 1
        _m(r, "Inmovilidad total",                     immobile_pct,   "%");  r += 1
        _m(r, "  Inmovilidad zona interior (freezing)", imm_c_pct,     "%");  r += 1
        _m(r, "  Inmovilidad zona periferica",          imm_p_pct,     "%");  r += 1

        r += 1
        _s(r, "Grooming (Conducta de Desplazamiento de Estres)"); r += 1
        _m(r, "Grooming (bouts/min)",                  grooming_pm,    "bouts/min"); r += 1
        _m(r, "Grooming duracion total",               grooming_dur,   "s");         r += 1
        _m(r, "Grooming duracion media por bout",      grooming_avg,   "s/bout");    r += 1

        wb.save(str(out_path))
