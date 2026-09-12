"""
Comparacion estadistica entre grupos experimentales (Open Field Test).

Lee multiples archivos Excel (stats_*.xlsx) generados por StatsGenerator y produce:
  - comparacion_grupos.png   : panel con todos los graficos en cuadricula
  - <metrica>.png            : un PNG por metrica
  - comparacion_grupos.xlsx  : hoja "Datos" (individual) + hoja "Resumen" (media+-SEM, p-valor)

Uso:
    from utils.group_stats import GroupStatsGenerator
    entries = [
        {"path": "stats_rata1.xlsx", "group": "Control",  "session": "Semana 1"},
        {"path": "stats_rata2.xlsx", "group": "FOLFOX",   "session": "Semana 1"},
        ...
    ]
    GroupStatsGenerator(entries, output_dir, group_names=("Control", "FOLFOX")).generate()
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# Colores BGR->RGB por grupo
_COLOR_CTRL   = "#555555"
_COLOR_TREAT  = "#CC0000"
_HATCH_CTRL   = ""
_HATCH_TREAT  = "//"


class GroupStatsGenerator:
    """
    Genera graficas comparativas y Excel de resumen a partir de multiples
    archivos stats_*.xlsx, agrupados por grupo experimental y sesion.

    entries: lista de dicts con claves 'path' (str|Path), 'group' (str), 'session' (str)
    group_names: (nombre_control, nombre_tratamiento)
    """

    METRICS: list[tuple[str, str, str]] = [
        ("dist_total",      "Distancia total recorrida",    "m"),
        ("center_pct",      "Tiempo en zona central",       "%"),
        ("periph_pct",      "Tiempo en zona periferia",     "%"),
        ("dipping_dur",     "Tiempo en agujeros",           "s"),
        ("dipping_bouts",   "N. veces en agujeros",         "n"),
        ("climbing_dur",    "Tiempo escapadas en pared",    "s"),
        ("climbing_bouts",  "N. escapadas en pared",        "n"),
        ("rearing_dur",     "Tiempo erguidas",              "s"),
        ("rearing_bouts",   "N. erguidas",                  "n"),
    ]

    def __init__(
        self,
        entries: list[dict],
        output_dir: Path,
        group_names: tuple[str, str] = ("Control", "Tratamiento"),
    ) -> None:
        self._entries    = entries
        self._output_dir = Path(output_dir)
        self._group_names = group_names

    # ------------------------------------------------------------------
    # Punto de entrada
    # ------------------------------------------------------------------

    def generate(self, log_fn=None) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)

        def _log(msg: str) -> None:
            (log_fn or print)(msg)

        _log("Cargando archivos...")
        raw = self._load_all(_log)
        if not raw:
            _log("[!] No se pudo leer ningun archivo.")
            return self._output_dir

        data = self._build_data(raw)

        if _HAS_MPL:
            _log("Generando figuras...")
            self._make_figures(data, _log)
        else:
            _log("[!] matplotlib no instalado — omitiendo figuras.")

        if _HAS_OPENPYXL:
            _log("Generando Excel...")
            xl_path = self._output_dir / "comparacion_grupos.xlsx"
            self._make_excel(data, raw, xl_path, _log)
        else:
            _log("[!] openpyxl no instalado — omitiendo Excel.")

        _log(f"Listo. Archivos en: {self._output_dir}")
        return self._output_dir

    # ------------------------------------------------------------------
    # Carga y extraccion
    # ------------------------------------------------------------------

    def _load_all(self, log_fn) -> list[dict]:
        results = []
        for entry in self._entries:
            path  = Path(entry["path"])
            group = entry.get("group", "Control")
            sess  = entry.get("session", "Sesion 1")
            if not path.exists():
                log_fn(f"  [!] No encontrado: {path.name}")
                continue
            try:
                m = self._extract_from_excel(path)
                m["_group"]   = group
                m["_session"] = sess
                m["_file"]    = path.name
                results.append(m)
                log_fn(f"  OK: {path.name}  [{group} / {sess}]")
            except Exception as exc:
                log_fn(f"  [!] Error en {path.name}: {exc}")
        return results

    def _extract_from_excel(self, path: Path) -> dict:
        if not _HAS_OPENPYXL:
            raise RuntimeError("openpyxl no instalado")

        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        m: dict = {}

        # ---- Hoja "Metricas OFT" ----
        if "Metricas OFT" in wb.sheetnames:
            ws = wb["Metricas OFT"]
            for row in ws.iter_rows(values_only=True):
                if not row[0] or row[1] is None:
                    continue
                label = str(row[0]).strip()
                value = row[1]

                if "Distancia total recorrida" in label:
                    try:
                        m["dist_total"] = float(value)
                        m["dist_unit"]  = str(row[2]).strip() if row[2] else "?"
                    except (TypeError, ValueError):
                        pass
                elif "Duracion total head-dipping" in label:
                    try: m["dipping_dur"] = float(value)
                    except (TypeError, ValueError): pass
                elif "N. de head-dips totales" in label:
                    try: m["dipping_bouts"] = float(value)
                    except (TypeError, ValueError): pass
                elif "Climbing (n. bouts)" in label:
                    try: m["climbing_bouts"] = float(value)
                    except (TypeError, ValueError): pass
                elif "Tiempo en zona central (cuadrado)" in label:
                    try: m["center_pct"] = float(value)
                    except (TypeError, ValueError): pass
                elif "Tiempo en zona periferica (fuera cuad.)" in label:
                    try: m["periph_pct"] = float(value)
                    except (TypeError, ValueError): pass

        # ---- Hoja "Comportamiento" ----
        if "Comportamiento" in wb.sheetnames:
            ws = wb["Comportamiento"]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if not row[0]:
                    continue
                label = str(row[0]).strip()
                dur   = row[2]   # columna C: Duracion (s)
                bouts = row[4]   # columna E: Bouts

                if label == "rat_climbing":
                    try: m["climbing_dur"] = float(dur)
                    except (TypeError, ValueError): pass
                elif label == "rat_rearing":
                    try: m["rearing_dur"]   = float(dur)
                    except (TypeError, ValueError): pass
                    try: m["rearing_bouts"] = float(bouts)
                    except (TypeError, ValueError): pass

        wb.close()
        return m

    # ------------------------------------------------------------------
    # Construccion de la estructura de datos
    # ------------------------------------------------------------------

    def _build_data(self, raw: list[dict]) -> dict:
        """data[metric_key][session][group] = [val, val, ...]"""
        data: dict = {k: {} for k, _, _ in self.METRICS}
        sessions_order: list[str] = []

        for entry in raw:
            sess  = entry["_session"]
            group = entry["_group"]
            if sess not in sessions_order:
                sessions_order.append(sess)
            for key, _, _ in self.METRICS:
                data[key].setdefault(sess, {}).setdefault(group, [])
                if key in entry:
                    data[key][sess][group].append(entry[key])

        data["_sessions_order"] = sessions_order
        return data

    def _all_groups(self, data: dict) -> list[str]:
        groups: list[str] = []
        for key, _, _ in self.METRICS:
            for sess_data in data.get(key, {}).values():
                for g in sess_data:
                    if g not in groups:
                        groups.append(g)
        return groups

    # ------------------------------------------------------------------
    # Figuras matplotlib
    # ------------------------------------------------------------------

    def _color(self, group: str) -> str:
        if group == self._group_names[0]:
            return _COLOR_CTRL
        return _COLOR_TREAT

    def _hatch(self, group: str) -> str:
        if group == self._group_names[0]:
            return _HATCH_CTRL
        return _HATCH_TREAT

    def _make_figures(self, data: dict, log_fn) -> None:
        sessions = data.get("_sessions_order", [])
        groups   = self._all_groups(data)
        if not sessions or not groups:
            return

        n_cols = 3
        n_rows = (len(self.METRICS) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, (key, name, unit) in enumerate(self.METRICS):
            self._draw_bar(axes_flat[idx], data, key, name, unit, sessions, groups)

        for ax in axes_flat[len(self.METRICS):]:
            ax.set_visible(False)

        handles = [
            mpatches.Patch(color=self._color(g), hatch=self._hatch(g), label=g)
            for g in groups
        ]
        fig.legend(handles=handles, loc="lower right", fontsize=10, frameon=True)
        fig.suptitle("Comparacion de grupos — Open Field Test", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])

        panel_path = self._output_dir / "comparacion_grupos.png"
        fig.savefig(str(panel_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log_fn(f"  Panel: {panel_path.name}")

        # PNGs individuales por metrica
        rng = np.random.default_rng(42)
        for key, name, unit in self.METRICS:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            self._draw_bar(ax2, data, key, name, unit, sessions, groups, rng=rng)
            handles2 = [
                mpatches.Patch(color=self._color(g), hatch=self._hatch(g), label=g)
                for g in groups
            ]
            ax2.legend(handles=handles2, loc="upper right", fontsize=9, frameon=True)
            fig2.tight_layout()
            fig2.savefig(str(self._output_dir / f"{key}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig2)

        log_fn(f"  {len(self.METRICS)} graficas individuales guardadas.")

    def _draw_bar(
        self, ax, data: dict, key: str, title: str, unit: str,
        sessions: list[str], groups: list[str],
        rng: "np.random.Generator | None" = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng(42)

        n_g   = len(groups)
        x     = np.arange(len(sessions))
        width = 0.32

        for g_idx, grp in enumerate(groups):
            means, sems, all_vals = [], [], []
            for sess in sessions:
                vals = [v for v in data.get(key, {}).get(sess, {}).get(grp, []) if v is not None]
                means.append(float(np.mean(vals)) if vals else 0.0)
                sems.append(
                    float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                )
                all_vals.append(vals)

            offset = (g_idx - (n_g - 1) / 2) * (width + 0.04)
            bx     = x + offset
            color  = self._color(grp)
            hatch  = self._hatch(grp)

            ax.bar(bx, means, width, yerr=sems, capsize=4,
                   color=color, hatch=hatch, alpha=0.82,
                   error_kw={"linewidth": 1.2, "ecolor": "#333333"},
                   edgecolor="#333333", linewidth=0.8)

            for i, (bxi, vals) in enumerate(zip(bx, all_vals)):
                if vals:
                    jitter = rng.uniform(-width * 0.18, width * 0.18, len(vals))
                    ax.scatter(bxi + jitter, vals, color="black", s=18,
                               zorder=5, alpha=0.75, linewidths=0)

            # Marcadores de significancia (solo para 2 grupos, en el segundo)
            if _HAS_SCIPY and n_g == 2 and g_idx == 1:
                ctrl = groups[0]
                for s_idx, sess in enumerate(sessions):
                    v0 = [v for v in data.get(key, {}).get(sess, {}).get(ctrl, []) if v is not None]
                    v1 = [v for v in data.get(key, {}).get(sess, {}).get(grp, []) if v is not None]
                    if len(v0) >= 2 and len(v1) >= 2:
                        _, p = scipy_stats.ttest_ind(v0, v1)
                        sig = _p_stars(p)
                        if sig:
                            y_top = max(max(v0 + v1, default=0), means[s_idx]) * 1.14
                            ax.text(x[s_idx], y_top, sig, ha="center", va="bottom",
                                    fontsize=9, color="#CC0000", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(sessions, fontsize=8)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    # ------------------------------------------------------------------
    # Excel de resumen
    # ------------------------------------------------------------------

    def _make_excel(self, data: dict, raw: list[dict], out_path: Path, log_fn) -> None:
        wb    = openpyxl.Workbook()
        sessions = data.get("_sessions_order", [])
        groups   = self._all_groups(data)

        hdr_fill = PatternFill("solid", fgColor="1F4E79")
        hdr_font = Font(bold=True, color="FFFFFF")

        # ---- Hoja 1: Datos individuales ----
        ws = wb.active
        ws.title = "Datos"

        metric_names = [name for _, name, _ in self.METRICS]
        headers = ["Archivo", "Grupo", "Sesion"] + metric_names
        for col, h in enumerate(headers, 1):
            c = ws.cell(row=1, column=col, value=h)
            c.font = hdr_font
            c.fill = hdr_fill
            c.alignment = Alignment(horizontal="center")

        ws.column_dimensions["A"].width = 34
        ws.column_dimensions["B"].width = 14
        ws.column_dimensions["C"].width = 14
        for i in range(len(self.METRICS)):
            col_letter = _col_letter(4 + i)
            ws.column_dimensions[col_letter].width = 24

        for row_i, entry in enumerate(raw, 2):
            ws.cell(row=row_i, column=1, value=entry.get("_file", ""))
            ws.cell(row=row_i, column=2, value=entry.get("_group", ""))
            ws.cell(row=row_i, column=3, value=entry.get("_session", ""))
            for col_i, (key, _, _) in enumerate(self.METRICS, 4):
                v = entry.get(key)
                if v is not None:
                    ws.cell(row=row_i, column=col_i, value=round(float(v), 4))

        # ---- Hoja 2: Resumen (media +- SEM) ----
        ws2 = wb.create_sheet("Resumen")
        hdr2 = ["Metrica", "Unidad", "Sesion", "Grupo", "Media", "SEM", "N"]
        if _HAS_SCIPY:
            hdr2 += ["p-valor (t-test)", "Significancia"]
        for col, h in enumerate(hdr2, 1):
            c = ws2.cell(row=1, column=col, value=h)
            c.font = hdr_font
            c.fill = hdr_fill
            c.alignment = Alignment(horizontal="center")

        ws2.column_dimensions["A"].width = 32
        ws2.column_dimensions["B"].width = 8
        ws2.column_dimensions["C"].width = 12
        ws2.column_dimensions["D"].width = 14
        for letter in ["E", "F", "G", "H", "I"]:
            ws2.column_dimensions[letter].width = 16

        r = 2
        for key, name, unit in self.METRICS:
            first = True
            for sess in sessions:
                # Calcula p-valor entre los 2 primeros grupos una sola vez por sesion
                p_val = None
                if _HAS_SCIPY and len(groups) >= 2:
                    v0 = [v for v in data.get(key, {}).get(sess, {}).get(groups[0], []) if v is not None]
                    v1 = [v for v in data.get(key, {}).get(sess, {}).get(groups[1], []) if v is not None]
                    if len(v0) >= 2 and len(v1) >= 2:
                        _, p_val = scipy_stats.ttest_ind(v0, v1)

                for g_idx, grp in enumerate(groups):
                    vals = [v for v in data.get(key, {}).get(sess, {}).get(grp, []) if v is not None]
                    n    = len(vals)
                    mean = round(float(np.mean(vals)), 4)      if n > 0 else ""
                    sem  = round(float(np.std(vals, ddof=1) / np.sqrt(n)), 4) if n > 1 else (0.0 if n == 1 else "")

                    row_data = [
                        name if first else "",
                        unit if first else "",
                        sess,
                        grp,
                        mean,
                        sem,
                        n,
                    ]
                    first = False

                    if _HAS_SCIPY:
                        if p_val is not None and g_idx == 1:
                            row_data += [round(p_val, 4), _p_stars(p_val) or "ns"]
                        else:
                            row_data += ["", ""]

                    for col, val in enumerate(row_data, 1):
                        ws2.cell(row=r, column=col, value=val)
                    r += 1
            r += 1  # fila en blanco entre metricas

        wb.save(str(out_path))
        log_fn(f"  Excel: {out_path.name}")


# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------

def _p_stars(p: float) -> str:
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""


def _col_letter(n: int) -> str:
    """Convierte indice de columna (1-based) a letra Excel (A, B, ..., Z, AA, ...)."""
    result = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result
