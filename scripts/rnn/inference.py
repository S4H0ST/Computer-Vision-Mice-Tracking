import torch
import numpy as np
from collections import deque
from rnn.model import RatActionRNN
from config.config import paths


class ActionPredictor:
    def __init__(self, seq_length=30):
        self.seq_length = seq_length
        self.buffer = deque(maxlen=seq_length)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = RatActionRNN().to(self.device)
        model_path = paths.rnn_model

        try:
            if model_path.exists():
                self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
                self.model.eval()
                self.active = True
                print(f"[Brain] Cerebro cargado desde: {model_path.name}")
            else:
                print(f"[Brain] ADVERTENCIA: No existe {model_path}. La RNN no funcionará.")
                self.active = False
        except Exception as e:
            print(f"[Brain] Error cargando modelo: {e}")
            self.active = False

        self.class_names = ['rat_rearing', 'rat_grooming', 'rat_horizontal', 'rat_climbing', 'rat_head_dipping']

    def update_and_predict(self, box, img_w=640, img_h=480):
        if not self.active:
            return None

        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        speed = 0.0
        if len(self.buffer) > 0:
            prev_cx, prev_cy = self.buffer[-1][0], self.buffer[-1][1]
            speed = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) * 100

        self.buffer.append([cx, cy, w, h, speed])

        if len(self.buffer) == self.seq_length:
            tensor_in = torch.FloatTensor([list(self.buffer)]).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor_in)
                _, predicted = torch.max(outputs, 1)
                idx = predicted.item()
                if 0 <= idx < len(self.class_names):
                    return self.class_names[idx]

        return "Analyzing..."
