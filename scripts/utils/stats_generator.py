"""
Generador de estadisticas post-deteccion para experimentos Open Field Test (Holeboard).

Uso:
    from tools.stats_generator import StatsGenerator
    StatsGenerator(csv_path, coords_json=paths.coords_json).generate(output_dir)

Salidas (en output_dir/):
    trajectory_<stem>.png  — trayectoria del tail sobre la plantilla de la caja (linea verde)
    stats_<stem>.xlsx      — hoja Comportamiento (presupuesto de tiempo) +
                             hoja Metricas OFT (indices farmacologicos)

Requiere openpyxl para el Excel (pip install openpyxl).
Si no esta instalado se genera solo la imagen de trayectoria.
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
        self.holes:        list[tuple] = []
        self.hole_radius:  int = 20

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
            self.inner_limits = data.get("limits_inner")
            self.outer_limits = data.get("limits_outer")
            self.holes        = [tuple(h) for h in data.get("holes", [])]
            self.hole_radius  = data.get("hole_radius", 20)

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

    # ------------------------------------------------------------------
    # Imagen de trayectoria
    # ------------------------------------------------------------------

    def _generate_trajectory(self, out_path: Path) -> None:
        sz = self.CANVAS_SIZE
        m  = self.CANVAS_MARGIN
        img = np.zeros((sz, sz, 3), dtype=np.uint8)

        # Rectangulo blanco de la caja
        cv2.rectangle(img, (m, m), (sz - m, sz - m), (255, 255, 255), 2)

        # Cuadricula gris semitransparente (4x4)
        grid = img.copy()
        step = (sz - 2 * m) // 4
        for i in range(1, 4):
            o = m + i * step
            cv2.line(grid, (o, m),     (o, sz - m), (70, 70, 70), 1)
            cv2.line(grid, (m, o), (sz - m, o),     (70, 70, 70), 1)
        cv2.addWeighted(grid, 0.55, img, 0.45, 0, img)

        x_min, y_min, x_scale, y_scale = self._canvas_mapping()

        # Trayectoria del tail en verde
        COLOR_TRACK = (0, 200, 0)
        prev_pt: tuple[int, int] | None = None
        for row in self.rows:
            try:
                tx = float(row.get("tail_x", -1))
                ty = float(row.get("tail_y", -1))
            except ValueError:
                prev_pt = None
                continue
            if tx < 0 or ty < 0:
                prev_pt = None
                continue
            pt = self._to_canvas(tx, ty, x_min, y_min, x_scale, y_scale)
            if prev_pt is not None:
                cv2.line(img, prev_pt, pt, COLOR_TRACK, 1)
            prev_pt = pt

        cv2.imwrite(str(out_path), img)

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

        # Head-dipping
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

        # Grooming
        grooming_b   = bouts_cnt.get("rat_grooming", 0)
        grooming_fr  = label_cnt.get("rat_grooming", 0)
        grooming_dur = round(grooming_fr / self.fps, 2)
        grooming_avg = round(grooming_dur / grooming_b, 2) if grooming_b > 0 else 0.0
        grooming_pm  = round(grooming_b / mins, 2) if mins > 0 else 0.0

        # Climbing (thigmotaxis en holeboard)
        climbing_b   = bouts_cnt.get("rat_climbing", 0)
        climbing_pct = round(label_cnt.get("rat_climbing", 0) / n * 100, 1)

        # Zonificacion (requiere calibracion)
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

        _s(r, "Duracion y Actividad General"); r += 1
        _m(r, "Duracion total del video",              round(duration_s, 1), "s");    r += 1
        _m(r, "Distancia total recorrida (tail)",      round(total_dist, 0), "px");   r += 1
        _m(r, "Velocidad media (tail)",                avg_speed,            "px/s"); r += 1
        _m(r, "Walking (deambulacion)",                walking_pct,          "%");    r += 1
        _m(r, "Transiciones conductuales totales",     transitions,          "");     r += 1
        _m(r, "Tasa de transicion",                    trans_pm,             "trans/min"); r += 1

        r += 1
        _s(r, "Head-dipping (Indice Principal de Exploracion)"); r += 1
        _m(r, "N. de head-dips (bouts)",               dipping_b,    "bouts");    r += 1
        _m(r, "Head-dips por minuto",                  dipping_pm,   "bouts/min"); r += 1
        _m(r, "Latencia al primer head-dip",           latencia_hd,  "s");        r += 1
        _m(r, "Duracion total head-dipping",           dipping_dur,  "s");        r += 1
        _m(r, "Duracion media por head-dip",           dipping_avg,  "s/bout");   r += 1
        _m(r, "Habituacion — head-dips (1er cuarto)",  hd_q[0],      "bouts");    r += 1
        _m(r, "Habituacion — head-dips (2o cuarto)",   hd_q[1],      "bouts");    r += 1
        _m(r, "Habituacion — head-dips (3er cuarto)",  hd_q[2],      "bouts");    r += 1
        _m(r, "Habituacion — head-dips (4o cuarto)",   hd_q[3],      "bouts");    r += 1

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
        _m(r, "Tiempo zona central (centroide interior)", central_pct,  "%");      r += 1
        _m(r, "Tiempo zona periferica (centroide exterior)", periph_pct, "%");     r += 1

        r += 1
        _s(r, "Inactividad y Estado Emocional"); r += 1
        _m(r, "Inmovilidad total",                     immobile_pct,   "%");  r += 1
        _m(r, "  Inmovilidad zona central (freezing)", imm_c_pct,      "%");  r += 1
        _m(r, "  Inmovilidad zona periferica",         imm_p_pct,      "%");  r += 1

        r += 1
        _s(r, "Grooming (Conducta de Desplazamiento de Estres)"); r += 1
        _m(r, "Grooming (bouts/min)",                  grooming_pm,    "bouts/min"); r += 1
        _m(r, "Grooming duracion total",               grooming_dur,   "s");         r += 1
        _m(r, "Grooming duracion media por bout",      grooming_avg,   "s/bout");    r += 1

        wb.save(str(out_path))
