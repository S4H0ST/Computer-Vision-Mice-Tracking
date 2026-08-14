import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class RatDataset(Dataset):
    def __init__(self, csv_files, seq_length=30):
        self.seq_length = seq_length
        self.features = []
        self.labels = []

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
                df = pd.read_csv(f)

                target_col = None
                if 'final_label' in df.columns:
                    target_col = 'final_label'
                elif 'label' in df.columns:
                    target_col = 'label'
                elif 'yolo_label' in df.columns:
                    target_col = 'yolo_label'

                if target_col is None:
                    print(f"[!] Saltando {f}: No encuentro columna de etiqueta válida.")
                    continue

                df['w'] = (df['x2'] - df['x1']) / 640.0
                df['h'] = (df['y2'] - df['y1']) / 480.0
                df['cx'] = ((df['x1'] + df['x2']) / 2) / 640.0
                df['cy'] = ((df['y1'] + df['y2']) / 2) / 480.0
                df['speed'] = np.sqrt(df['cx'].diff() ** 2 + df['cy'].diff() ** 2).fillna(0) * 100

                clean_df = df[['cx', 'cy', 'w', 'h', 'speed', target_col]].dropna()

                data_values = clean_df[['cx', 'cy', 'w', 'h', 'speed']].values
                label_values = clean_df[target_col].values

                for i in range(len(clean_df) - self.seq_length):
                    seq = data_values[i: i + self.seq_length]
                    target_text = label_values[i + self.seq_length - 1]

                    if target_text in self.label_map:
                        self.features.append(seq)
                        self.labels.append(self.label_map[target_text])

            except Exception as e:
                print(f"[!] Error inesperado leyendo {f}: {e}")

        if len(self.features) > 0:
            self.features = torch.FloatTensor(np.array(self.features))
            self.labels = torch.LongTensor(np.array(self.labels))
        else:
            self.features = torch.empty(0)
            self.labels = torch.empty(0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
