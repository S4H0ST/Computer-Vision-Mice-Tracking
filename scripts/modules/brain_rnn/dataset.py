import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class RatDataset(Dataset):
    def __init__(self, csv_files, seq_length=30):
        self.seq_length = seq_length
        self.features = []
        self.labels = []

        # El vocabulario mezcla clases YOLO (rat_climbing, rat_grooming,
        # rat_head_dipping, rat_horizontal, rat_rearing) con labels que serán
        # derivados por lógica de post-procesado en RatDetector (rat_walking
        # cuando speed supera un umbral). PENDIENTE DE IMPLEMENTAR en
        # detector.py — el label_map ya está preparado para cuando se añada
        # esa lógica.
        self.label_map = {
            # Nombres largos (Dataset original)
            'rat_rearing': 0,
            'rat_grooming': 1,
            'rat_horizontal': 2,
            'rat_walking': 2,  # PENDIENTE: asignado por lógica de velocidad en RatDetector
            'rat_climbing': 3,
            'rat_head_dipping': 4,

            # Nombres cortos (Por si YOLO devuelve solo esto)
            'rearing': 0, 'grooming': 1, 'horizontal': 2,
            'walking': 2, 'climbing': 3, 'head_dipping': 4, 'dipping': 4
        }

        self._process_files(csv_files)

    def _process_files(self, files):
        for f in files:
            try:
                # 1. Cargar CSV
                df = pd.read_csv(f)

                # 2. DETECTAR NOMBRE DE LA COLUMNA DE ETIQUETA
                target_col = None
                if 'final_label' in df.columns:
                    target_col = 'final_label'  # La que genera tu código nuevo
                elif 'label' in df.columns:
                    target_col = 'label'  # Por compatibilidad antigua
                elif 'yolo_label' in df.columns:
                    target_col = 'yolo_label'  # Respaldo

                if target_col is None:
                    print(f"[!] Saltando {f}: No encuentro columna de etiqueta válida.")
                    continue

                # 3. Normalización (0.0 a 1.0)
                # Asumimos 640x480. Si tus videos son distintos, esto se ajustará solo si usas config
                df['w'] = (df['x2'] - df['x1']) / 640.0
                df['h'] = (df['y2'] - df['y1']) / 480.0
                df['cx'] = ((df['x1'] + df['x2']) / 2) / 640.0
                df['cy'] = ((df['y1'] + df['y2']) / 2) / 480.0

                # 4. Velocidad (Diferencia con frame anterior)
                # Multiplicamos por 100 para que el número no sea tan pequeño (ayuda a la red)
                df['speed'] = np.sqrt(df['cx'].diff() ** 2 + df['cy'].diff() ** 2).fillna(0) * 100

                # 5. Filtrar columnas útiles
                # Usamos 'target_col' que hemos detectado arriba
                clean_df = df[['cx', 'cy', 'w', 'h', 'speed', target_col]].dropna()

                # 6. Crear secuencias (Ventanas de tiempo)
                data_values = clean_df[['cx', 'cy', 'w', 'h', 'speed']].values
                label_values = clean_df[target_col].values

                sequences_created = 0
                for i in range(len(clean_df) - self.seq_length):
                    seq = data_values[i: i + self.seq_length]
                    target_text = label_values[i + self.seq_length - 1]  # Etiqueta del último frame

                    if target_text in self.label_map:
                        self.features.append(seq)
                        self.labels.append(self.label_map[target_text])
                        sequences_created += 1

                # print(f"   -> {f}: {sequences_created} secuencias extraídas.")

            except Exception as e:
                print(f"[!] Error inesperado leyendo {f}: {e}")

        # Convertir a tensores de PyTorch
        if len(self.features) > 0:
            self.features = torch.FloatTensor(np.array(self.features))
            self.labels = torch.LongTensor(np.array(self.labels))
        else:
            # Crear tensores vacíos para evitar crash
            self.features = torch.empty(0)
            self.labels = torch.empty(0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]