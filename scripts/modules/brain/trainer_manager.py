import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules.brain.model import RatActionRNN
from modules.brain.dataset import RatDataset
import glob


class RNNTrainer:
    def __init__(self, data_path='data/training_csvs/*.csv', batch_size=32, lr=0.001):
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RatActionRNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=50, save_path='best_rnn.pth'):
        files = glob.glob(self.data_path)
        if not files:
            print("[X] No hay CSVs para entrenar.")
            return

        dataset = RatDataset(files)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"[i] Iniciando entrenamiento con {len(dataset)} secuencias...")

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

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

        torch.save(self.model.state_dict(), save_path)
        print(f"[OK] Modelo RNN guardado en: {save_path}")