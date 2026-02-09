import torch
import numpy as np
from collections import deque
from modules.brain.model import RatActionRNN


class ActionPredictor:
    def __init__(self, model_path='best_rnn.pth', seq_length=30):
        self.seq_length = seq_length
        self.buffer = deque(maxlen=seq_length)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cargar modelo
        self.model = RatActionRNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.active = True
            print("[OK] Cerebro RNN cargado correctamente.")
        except:
            print("[!] ADVERTENCIA: No se encontró 'best_rnn.pth'. La RNN no funcionará.")
            self.active = False

        # Nombres para decodificar la salida (orden del dataset)
        self.class_names = ['Rearing', 'Grooming', 'Walking', 'Climbing', 'Head_Dipping']

    def update_and_predict(self, box, img_w=640, img_h=480):
        if not self.active:
            return None

        # Desempaquetar y normalizar
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        # Calcular velocidad (simple)
        speed = 0.0
        if len(self.buffer) > 0:
            prev_cx, prev_cy = self.buffer[-1][0], self.buffer[-1][1]
            speed = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) * 100

        # Añadir al buffer
        self.buffer.append([cx, cy, w, h, speed])

        # Predecir solo si el buffer está lleno
        if len(self.buffer) == self.seq_length:
            tensor_in = torch.FloatTensor([list(self.buffer)]).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor_in)
                _, predicted = torch.max(outputs, 1)
                idx = predicted.item()
                return self.class_names[idx]

        return "Analyzing..."