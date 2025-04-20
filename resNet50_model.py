import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ResNet50Classifier(pl.LightningModule):
    def __init__(self, num_classes=10):
        """
        Klasa definiująca model ResNet50 dostosowany do klasyfikacji obrazów grayscale (np. MNIST)
        z wykorzystaniem PyTorch Lightning.
        """
        super().__init__()
        self.save_hyperparameters()  # automatyczne logowanie hiperparametrów (np. liczba klas)

        # Wczytanie pretrenowanego modelu ResNet50 z wagami z ImageNet
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Zmiana pierwszej warstwy wejściowej na obsługującą obrazy grayscale (1 kanał)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Zamiana ostatniej warstwy FC (fully connected) na dostosowaną do 10 klas
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Funkcja straty: klasyfikacja wieloklasowa
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Jeden krok treningowy: obliczenie straty i dokładności na batchu.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Jeden krok walidacyjny: obliczenie straty i dokładności na batchu walidacyjnym.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        """
        Konfiguracja optymalizatora.
        Używamy Adam'a z LR = 1e-4.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def visualize_results(self, dataloader, save_path=None):
        self.eval()  # przełączenie modelu w tryb ewaluacji
        all_preds, all_labels = [], []

        # Wyłączenie gradientów (ma na celu oszczędność zasobów)
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self(x).argmax(dim=1)
                all_preds.extend(preds.cpu())
                all_labels.extend(y.cpu())

        # Obliczenie i wyświetlenie macierzy pomyłek
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - ResNet50")

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
