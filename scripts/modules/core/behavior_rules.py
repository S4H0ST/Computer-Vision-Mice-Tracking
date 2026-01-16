# NOMBRE DEL ARCHIVO: modules/core/behavior_rules.py
import json
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple


class RatBehaviorRules:
    """
    Define las REGLAS DE NEGOCIO para interpretar el comportamiento.
    Corregido para priorizar movimiento y ubicación correctamente.
    """

    def __init__(self, config_path: Path):
        self.config_path: Path = config_path
        self.limits: Dict[str, int] = {}
        self.holes: List[Tuple[int, int]] = []
        self.history: deque = deque(maxlen=5)

        # --- UMBRALES ---
        self.WALK_THRESH: float = 3.0  # Bajado un poco para detectar caminatas lentas
        self.WALL_DIST: int = 40
        self.HOLE_DIST: int = 30  # Radio por defecto

        self._load_config()

    def _load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError("[X] Faltan coordenadas. Ejecuta calibración (Opción 1) primero.")

        with open(self.config_path, 'r') as f:
            data = json.load(f)
            self.limits = data['limits']
            self.holes = [tuple(h) for h in data['holes']]

            if "hole_radius" in data:
                self.HOLE_DIST = data["hole_radius"]

    def _get_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def check_location(self, cx: float, cy: float) -> str:
        """
        Determina la zona: HOLE, WALL o CENTER.
        NOTA: La prioridad se maneja mejor en apply_rules,
        aquí solo devolvemos la ubicación física.
        """
        # 1. ¿Está en un agujero?
        for hx, hy in self.holes:
            if self._get_distance(np.array([cx, cy]), np.array([hx, hy])) < self.HOLE_DIST:
                return 'HOLE'

        # 2. ¿Está en la pared? (Zona entre rect. interior y exterior)
        l = self.limits
        # Margen de seguridad para no considerar pared el centro absoluto
        # Si está fuera de los límites INTERIORES (hacia afuera), es pared
        if (cx < l['x_min'] or cx > l['x_max'] or
                cy < l['y_min'] or cy > l['y_max']):
            return 'WALL'

        # 3. Si no, es Centro
        return 'CENTER'

    def apply_rules(self, box: List[float], yolo_label: str) -> Tuple[str, float]:
        """
        Aplica reglas jerárquicas corregidas:
        1. Climbing (Si está en pared y postura vertical) -> GANA A TODO
        2. Walking (Si hay velocidad) -> GANA A REARING/DIPPING
        3. Hole/Dipping (Si está quieta en agujero)
        4. Rearing (Si está quieta en centro y levantada)
        """
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        location: str = self.check_location(cx, cy)

        # --- CÁLCULO DE VELOCIDAD ---
        current_pos = np.array([cx, cy])
        speed: float = 0.0
        if len(self.history) > 0:
            speed = self._get_distance(current_pos, self.history[-1])
        self.history.append(current_pos)

        is_moving: bool = speed > self.WALK_THRESH

        # =========================================================
        # REGLAS DE COMPORTAMIENTO (JERARQUÍA ESTRICTA)
        # =========================================================

        # REGLA 1: CLIMBING (Prioridad Máxima en Pared)
        # Si está en la pared y el modelo dice climbing o rearing, es CLIMBING.
        # Esto soluciona que te salga "head_dipping" en la pared.
        if location == 'WALL':
            if yolo_label in ["rat_climbing", "rat_rearing"]:
                return "rat_climbing", speed

        # REGLA 2: WALKING (Movimiento)
        # Si se mueve, está caminando. Esto evita que "caminar" se convierta en "rearing".
        # Excepción: Si el modelo está MUY seguro de que es climbing/rearing,
        # a veces caminan a dos patas, pero es raro. Asumimos walking.
        if is_moving:
            return "rat_walking", speed

        # REGLA 3: INTERACCIÓN CON AGUJEROS (Estático)
        # Si llegamos aquí, la rata está QUIETA (speed < thresh).
        if location == 'HOLE':
            # Cualquier interacción cerca del agujero quieta se considera dipping
            if yolo_label in ["rat_head_dipping", "rat_sniffing", "rat_horizontal"]:
                return "rat_head_dipping", speed

        # REGLA 4: COMPORTAMIENTO EN CENTRO (Estático)
        if location == 'CENTER':
            # Solución a tu problema: Solo es Rearing si el modelo dice Rearing/Climbing
            # Y además sabemos que NO se está moviendo (filtrado por Regla 2).
            if yolo_label in ["rat_rearing", "rat_climbing"]:
                return "rat_rearing", speed

            # Si dice head_dipping en el centro (sin agujero), es sniffing o immobility
            if yolo_label == "rat_head_dipping":
                return "rat_sniffing", speed

            return "rat_immobility", speed

        # REGLA 5: FALLBACK PARED
        if location == 'WALL':
            return "rat_sniffing", speed

        # Retorno por defecto
        return "rat_immobility", speed