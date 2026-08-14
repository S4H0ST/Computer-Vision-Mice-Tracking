import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rnn.model import RatActionRNN
from rnn.dataset import RatDataset
from config.config import paths
import glob


class RNNTrainer:
    def __init__(self, batch_size=32, lr=0.001):
        self.search_pattern = str(paths.output_dir / "*.csv")

        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RatActionRNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=50):
        files = glob.glob(self.search_pattern)
        print(f"[Brain] Buscando CSVs en: {self.search_pattern}")

        if not files:
            print("[X] ERROR: No encontré archivos .csv en la carpeta 'outputs'.")
            print("    -> Ejecuta primero la Opción 3 para generar datos.")
            return

        dataset = RatDataset(files)
        if len(dataset) == 0:
            print("[X] ERROR: Los CSV encontrados están vacíos o corruptos.")
            return

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"[i] Entrenando con {len(dataset)} secuencias de movimiento...")

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for sequences, targets in loader:
                sequences, targets = sequences.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

        save_location = paths.rnn_model
        torch.save(self.model.state_dict(), save_location)
        print(f"[OK] Cerebro RNN guardado correctamente en: {save_location}")
