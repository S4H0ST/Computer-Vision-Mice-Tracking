# NOMBRE DEL ARCHIVO: modules/core/behavior_rules.py
import json
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple


class RatBehaviorRules:
    """
    Define las REGLAS DE NEGOCIO para interpretar el comportamiento.
    Traduce coordenadas y etiquetas crudas de YOLO a comportamientos reales.
    """

    def __init__(self, config_path: Path):
        self.config_path: Path = config_path
        self.limits: Dict[str, int] = {}
        self.holes: List[Tuple[int, int]] = []

        # Historial para calcular velocidad
        self.history: deque = deque(maxlen=5)

        # --- UMBRALES ---
        self.WALK_THRESH: float = 4.0
        self.WALL_DIST: int = 40
        self.HOLE_DIST: int = 30  # Valor por defecto (se sobrescribirá si el JSON tiene otro)

        self._load_config()

    def _load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError("[X] Faltan coordenadas. Ejecuta calibración primero.")

        with open(self.config_path, 'r') as f:
            data = json.load(f)
            self.limits = data['limits']
            self.holes = [tuple(h) for h in data['holes']]

            # --- CAMBIO IMPORTANTE: Leer el radio del agujero del JSON ---
            if "hole_radius" in data:
                self.HOLE_DIST = data["hole_radius"]
                # print(f"[i] Radio de agujero sincronizado: {self.HOLE_DIST}px")

    def _get_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def check_location(self, cx: float, cy: float) -> str:
        """Determina dónde está la rata: HOLE, WALL o CENTER."""

        # 1. ¿Está en un agujero?
        for hx, hy in self.holes:
            # Usamos el radio dinámico cargado desde la calibración
            if self._get_distance(np.array([cx, cy]), np.array([hx, hy])) < self.HOLE_DIST:
                return 'HOLE'

        # 2. ¿Está en la pared?
        l = self.limits
        margin = self.WALL_DIST
        if (cx < l['x_min'] + margin or cx > l['x_max'] - margin or
                cy < l['y_min'] + margin or cy > l['y_max'] - margin):
            return 'WALL'

        return 'CENTER'

    def apply_rules(self, box: List[float], yolo_label: str) -> Tuple[str, float]:
        """
        Aplica las reglas jerárquicas:
        Climbing > Walking > Sniffing/Dipping > Immobility
        """
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        location: str = self.check_location(cx, cy)

        # Calculo velocidad
        current_pos = np.array([cx, cy])
        speed: float = 0.0
        if len(self.history) > 0:
            speed = self._get_distance(current_pos, self.history[-1])
        self.history.append(current_pos)

        is_moving: bool = speed > self.WALK_THRESH

        # --- APLICANDO REGLAS ---

        # 1. CLIMBING (Prioridad Máxima si está en pared)
        if yolo_label == "rat_climbing":
            if location == 'WALL':
                return "rat_climbing", speed
            return "rat_rearing", speed

            # 2. WALKING (Movimiento rápido)
        if is_moving and yolo_label in ["rat_horizontal", "rat_sniffing", "rat_walking", "rat_immobility"]:
            return "rat_walking", speed

        # 3. COMPORTAMIENTOS ESTÁTICOS
        if location == 'HOLE':
            if yolo_label in ["rat_head_dipping", "rat_horizontal", "rat_sniffing"]:
                return "rat_head_dipping", speed

        if location == 'WALL':
            if yolo_label in ["rat_horizontal", "rat_immobility", "rat_sniffing"]:
                return "rat_sniffing", speed

        if location == 'CENTER':
            if yolo_label in ["rat_horizontal", "rat_walking", "rat_immobility"]:
                return "rat_immobility", speed

        return yolo_label, speed