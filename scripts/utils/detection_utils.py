"""
Constantes y utilidades compartidas entre el pipeline de deteccion y la GUI.

Constants:
    KP_SNOUT  — indice del keypoint de morro en kpt_shape [snout, spine, tail].
    KP_SPINE  — indice del keypoint de columna.
    KP_TAIL   — indice del keypoint de cola.
    LABEL_ES  — traducciones de etiquetas internas al espanol (para consola).

Functions:
    _get_color — devuelve el color BGR asociado a una etiqueta de comportamiento.
"""

from __future__ import annotations


# Indices de keypoints segun kpt_shape: [snout, spine, tail]
KP_SNOUT: int = 0
KP_SPINE: int = 1
KP_TAIL: int  = 2

# Traducciones de etiquetas internas para la leyenda impresa en consola
LABEL_ES: dict[str, str] = {
    "rat_climbing":      "Trepar / Escalar",
    "rat_grooming":      "Acicalamiento / Limpieza",
    "rat_head_dipping":  "Asomarse por agujero",
    "rat_rearing":       "Incorporarse / Erguirse",
    "walking":           "Caminando",
    "immobile":          "Inmovil",
    "sniffing_walking":  "Olfateando (en movimiento)",
    "sniffing_immobile": "Olfateando (parado)",
}


def _get_color(label: str) -> tuple[int, int, int]:
    """Devuelve el color BGR asociado a cada etiqueta de comportamiento."""
    label = label.lower()
    if "sniffing_immobile" in label: return (200, 100, 180)
    if "sniffing"   in label: return (0,   200, 255)
    if "inmobile" in label or "immobile" in label: return (180, 180, 180)
    if "walking"    in label: return (0,   255, 255)
    if "climbing"   in label: return (255,   0, 255)
    if "dipping"    in label: return (0,   165, 255)
    if "rearing"    in label: return (0,   255,   0)
    if "grooming"   in label: return (180, 255, 180)
    return (128, 128, 128)
