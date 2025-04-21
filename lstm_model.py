import pytorch_lightning as pl
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Zakładamy, że wejście ma rozmiar (B, 1, 28, 28)
        x = x.squeeze(1)  # usuwamy kanał (B, 28, 28)
        lstm_out, _ = self.lstm(x)  # output: (B, 28, hidden_size)
        out = lstm_out[:, -1, :]    # ostatni krok czasowy (B, hidden_size)
        out = self.relu(self.fc1(out))
        return self.fc2(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def evaluate_accuracy(self, dataloader):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"\nAccuracy on the test set: {acc:.4f}")
        return acc

    def visualize_results(self, dataloader, save_path=None):
        self.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                preds = self(x).argmax(dim=1)
                all_preds.extend(preds.cpu())
                all_labels.extend(y.cpu())

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - LSTM")
        if save_path:
            plt.savefig(save_path)
        plt.show()
