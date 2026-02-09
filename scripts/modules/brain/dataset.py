import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class RatDataset(Dataset):
    def __init__(self, csv_files, seq_length=30):
        self.seq_length = seq_length
        self.features = []
        self.labels = []

        # MAPEO EXACTO (Debe coincidir con tu data.yaml)
        # 0:rearing, 1:grooming, 2:horizontal/walking, 3:climbing, 4:dipping
        self.label_map = {
            'rat_rearing': 0,
            'rat_grooming': 1,
            'rat_horizontal': 2,
            'rat_walking': 2,  # Unificamos walking y horizontal
            'rat_climbing': 3,
            'rat_head_dipping': 4
        }

        self._process_files(csv_files)

    def _process_files(self, files):
        for f in files:
            try:
                df = pd.read_csv(f)
                # Normalización (Asumiendo 640x480, ajustar si cambia la resolución)
                df['w'] = (df['x2'] - df['x1']) / 640.0
                df['h'] = (df['y2'] - df['y1']) / 480.0
                df['cx'] = ((df['x1'] + df['x2']) / 2) / 640.0
                df['cy'] = ((df['y1'] + df['y2']) / 2) / 480.0

                # Velocidad (delta de posición)
                df['speed'] = np.sqrt(df['cx'].diff() ** 2 + df['cy'].diff() ** 2).fillna(0) * 100

                # Limpiar
                clean_df = df[['cx', 'cy', 'w', 'h', 'speed', 'label']].dropna()

                # Crear ventanas deslizantes
                data_values = clean_df[['cx', 'cy', 'w', 'h', 'speed']].values
                label_values = clean_df['label'].values

                for i in range(len(clean_df) - self.seq_length):
                    seq = data_values[i: i + self.seq_length]
                    target = label_values[i + self.seq_length - 1]

                    if target in self.label_map:
                        self.features.append(seq)
                        self.labels.append(self.label_map[target])
            except Exception as e:
                print(f"[!] Error procesando {f}: {e}")

        self.features = torch.FloatTensor(np.array(self.features))
        self.labels = torch.LongTensor(np.array(self.labels))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]