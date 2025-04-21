from analyze_data import *  # Własne funkcje do analizy danych
from cnn_model import CNNModel  # Definicja modelu CNN
from resNet50_model import ResNet50Classifier  # Definicja modelu ResNet50
from hyperparameter_tuning import *  # Funkcje do strojenia hiperparametrów (rezerwacja na przyszłość)
from lstm_model import *  # Definicja modelu LSTM (rezerwacja na przyszłość)

import pandas as pd  # Do pracy z danymi (CSV, tabele)
import torch  # PyTorch – budowa i trenowanie modeli
from torch.utils.data import TensorDataset, DataLoader  # Tworzenie zbiorów danych i loaderów
import pytorch_lightning as pl  # Lightning – wygodny wrapper dla PyTorch
import os  # Obsługa plików i folderów
from pytorch_lightning.callbacks import ModelCheckpoint  # Zapisywanie najlepszego modelu
from pytorch_lightning.loggers import CSVLogger  # Logowanie postępów treningu do pliku CSV
import gc

def load_mnist_data(train_path="mnist_train.csv", test_path="mnist_test.csv", resize_to_96=False):
    """
    Wczytuje dane MNIST, przetwarza je i zwraca DataLoadery dla zbioru treningowego i walidacyjnego.

    Args:
    - train_path (str): Ścieżka do pliku CSV z danymi treningowymi.
    - test_path (str): Ścieżka do pliku CSV z danymi testowymi.
    - resize_to_224 (bool): Czy przeskalować obrazy do 96x96 (potrzebne dla ResNet50).

    Returns:
    - train_loader (DataLoader): Loader danych treningowych.
    - val_loader (DataLoader): Loader danych walidacyjnych.
    """

    # Wczytanie danych do DataFrame'ów
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Normalizacja pikseli i zmiana kształtu na (1, 28, 28)
    X_train = train_df.drop("label", axis=1).values.reshape(-1, 1, 28, 28) / 255.0
    y_train = train_df["label"].values
    X_test = test_df.drop("label", axis=1).values.reshape(-1, 1, 28, 28) / 255.0
    y_test = test_df["label"].values

    # Konwersja do tensorów PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Przeskalowanie do 64x64
    if resize_to_96:
        X_train = torch.nn.functional.interpolate(X_train, size=(96, 96), mode="bilinear")
        X_test = torch.nn.functional.interpolate(X_test, size=(96, 96), mode="bilinear")

    # Utworzenie obiektów TensorDataset
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    # Dobranie liczby workerów (równoległe ładowanie danych)
    num_workers = 2 # jezeli jest >0 to Ładowanie danych z dysku / pliku może być wolne, dane są ładowane równolegle z treningiem, GPU / CPU nie czekają na dane
                     # gdy 0 to jest mniejsze zużycie pamięc, Ładowanie danych w jednym wątku (kolejno).

    #Ustawiania dla CNN:
        #num_workers = 8, persistent_workers=True, batch_size=64
    #Ustawienie dla ResNet50:
        #num_workers=2, persistent_workers=False, batch_size=32 -> w celu optymalizacji pamięci RAM/CPU oraz aby przyśpieszyc trenowanie

    # DataLoadery dla treningu i walidacji
    train_loader = DataLoader(
        train_ds,  # Zbiór danych (TensorDataset z obrazkami i etykietami)
        batch_size=32,  # Liczba próbek w jednej partii (batch)
        shuffle=True,  # Tasuj dane co epokę (ważne dla treningu!)
        num_workers=num_workers, persistent_workers=True # Liczba procesów do ładowania danych (więcej = szybciej)
    )

    val_loader = DataLoader(
        test_ds, batch_size=32,
        num_workers=num_workers, persistent_workers=True
    )
    return train_loader, val_loader


# Funkcja trenująca model CNN
def train_cnn():
    """
    Trenuje model CNN na zbiorze MNIST
    """

    train_loader, val_loader = load_mnist_data(resize_to_96=False)

    # Inicjalizacja modelu CNN
    model = CNNModel()

    # Callback do zapisywania najlepszego modelu na podstawie val_loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/cnn",
        filename="cnn-best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )

    # Logger do zapisywania logów treningu do katalogu "logs" (CSV)
    logger = CSVLogger("logs", name="cnn")

    # Trener PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    model.evaluate_accuracy(val_loader)

    model.visualize_results(val_loader, save_path="cnn_conf_matrix.png")


def train_resnet50():
    """
    Trenuje model ResNet50 na zbiorze danych MNIST (obrazki grayscale) i zapisuje najlepszy model
    na podstawie straty walidacyjnej (var_loss)
    """

    # Wczytanie danych i przeskalowanie do 96x96
    train_loader, val_loader = load_mnist_data(resize_to_96=True)

    # Inicjalizacja modelu ResNet50 z 10 klasami wyjściowymi
    model = ResNet50Classifier(num_classes=10)

    # Callback zapisujący najlepszy model według najmniejszej straty walidacyjnej (val_loss)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",             # monitorowana metryka
        dirpath="checkpoints/resnet50", # katalog docelowy dla checkpointów
        filename="resnet50-best-{epoch:02d}-{val_loss:.2f}", # szablon nazwy pliku
        save_top_k=1,                   # zapisuj tylko najlepszy model
        mode="min"                      # minimalizujemy stratę walidacyjną
    )

    # Logger zapisujący metryki do pliku CSV
    logger = CSVLogger("logs", name="resnet50")

    # Inicjalizacja trenera Lightning
    trainer = pl.Trainer(
        max_epochs=10,          # liczba epok
        accelerator="auto",     # automatyczny wybór: GPU (jeśli dostępne) lub CPU
        callbacks=[checkpoint_callback],  # callback do zapisu modelu
        logger=logger           # logger do CSV
    )

    trainer.fit(model, train_loader, val_loader)

    model.evaluate_accuracy(val_loader)

    model.visualize_results(val_loader, save_path="resnet50_conf_matrix.png")


def train_lstm():
    """
    Trenuje model LSTM na zbiorze MNIST traktując obrazy jako sekwencje 28x28.
    """

    # Dla LSTM nie skalujemy danych, zostają 28x28
    train_loader, val_loader = load_mnist_data(resize_to_96=False)

    model = LSTMClassifier()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/lstm",
        filename="lstm-best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )

    logger = CSVLogger("logs", name="lstm")

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    model.evaluate_accuracy(val_loader)
    model.visualize_results(val_loader, save_path="lstm_conf_matrix.png")


def plot_training_curves(log_dir):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(log_dir)

    # Filtrowanie tylko z wypełnionymi wartościami
    df = df.dropna(subset=["epoch"])

    # Grupowanie po epoce i obliczanie średniej (jeśli potrzeba)
    epochs = df["epoch"].astype(int)
    train_acc = df[df["train_acc"].notnull()].groupby("epoch")["train_acc"].mean()
    val_acc = df[df["val_acc"].notnull()].groupby("epoch")["val_acc"].mean()
    train_loss = df[df["train_loss"].notnull()].groupby("epoch")["train_loss"].mean()
    val_loss = df[df["val_loss"].notnull()].groupby("epoch")["val_loss"].mean()

    # Accuracy plot
    plt.figure(figsize=(10, 4))
    plt.plot(train_acc.index, train_acc.values, label="Train Accuracy")
    plt.plot(val_acc.index, val_acc.values, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Loss plot
    plt.figure(figsize=(10, 4))
    plt.plot(train_loss.index, train_loss.values, label="Train Loss")
    plt.plot(val_loss.index, val_loss.values, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Wczytanie danych z plików CSV
    train_data = pd.read_csv('mnist_train.csv')
    test_data = pd.read_csv('mnist_test.csv')

    # Analiza danych (np. liczność klas, rozkład)
    #analyze_data(train_data, test_data)

    # Trening CNN - 28x28 pixeli
    #train_cnn()
    #plot_training_curves("logs/cnn/version_0/metrics.csv")

    #Hyperparameter tuning dla CNN
    #Sa tu eksparymenty dla różnej liczby filtor z różnym dropoutem i z mniejszymi batch_size oraz epokami.
    #run_all_experiments()
    # Test z różną liczbą filtrów
    #plot_training_curves("logs/cnn_small/version_0/metrics.csv")
    #plot_training_curves("logs/cnn_medium/version_0/metrics.csv")
    #plot_training_curves("logs/cnn_large/version_0/metrics.csv")

    # Test z różną liczbą epok
    #plot_training_curves("logs/cnn_epochs_5/version_0/metrics.csv")
    #plot_training_curves("logs/cnn_epochs_15/version_0/metrics.csv")

    # Test z mniejszym batch_size i większym dropoutem
    #plot_training_curves("logs/cnn_small_batch_dropout/version_0/metrics.csv")

    # Trening ResNet50 - 96x96 pixeli
    #train_resnet50()
    #plot_training_curves("logs/resnet50/version_0/metrics.csv")

    # Trening LSTM - 28 kroków po 28 pixeli (linia po linii)
    train_lstm()
    plot_training_curves("logs/lstm/version_0/metrics.csv")
