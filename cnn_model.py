import pytorch_lightning as pl
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class CNNModel(pl.LightningModule):
    def __init__(self, filters=[32, 64, 128], dropout=0.5, input_shape=(1, 28, 28)):
        super().__init__()
        self.save_hyperparameters()  # automatyczne logowanie hiperparametrów

        # Warstwy konwolucyjne
        self.conv_layers = nn.Sequential(
            # Warstwa konwolucyjna 1 -> uczy się prostych cech jak: linie, krawędzie, kąty
            nn.Conv2d(
                in_channels=input_shape[0],  # liczba kanałów wejściowych (obrazy grayscale)
                out_channels=filters[0],  # liczba filtrów uczonych przez model
                kernel_size=3,  # Rozmiar "okna" filtru 3x3
                padding=1  # dodanie 1 piksela dookoła, aby zachować oryginalny rozmiar
            ),
            nn.ReLU(),  # Funkcja aktywacji ReLU
            nn.MaxPool2d(2),  # Zmniejszenie rozmiaru przez pooling (2x2)

            # Warstwa konwolucyjna 2 -> wykorzystuje poprzednie cechy, by rozpoznawać np. kształty, kontury cyfr.
            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Warstwa konwolucyjna 3 -> buduje z poprzednich cech kompletne wzorce, np. „cyfra 8” albo „górna część 4”.
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Obliczanie rozmiaru wejścia do warstwy FC po przejściu przez konwolucje
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            out = self.conv_layers(dummy_input)
            flattened_size = out.view(1, -1).size(1)

        # Warstwy gęste (fully connected)
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10)  # wyjście: 10 klas
        )

        self.loss_fn = nn.CrossEntropyLoss() # funkcja straty do klasyfikacji wieloklasowej

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # spłaszczanie tensorów
        return self.fc_layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
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

    def visualize_results(self, dataloader, save_path=None):
        self.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self(x).argmax(dim=1)
                all_preds.extend(preds.cpu())
                all_labels.extend(y.cpu())

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - CNN Model")
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def evaluate_accuracy(self, dataloader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f'\nAccuracy on the test set: {acc:.4f}')
        return acc