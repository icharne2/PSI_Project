import pytorch_lightning as pl
from cnn_model import CNNModel
from main import load_mnist_data
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


def run_experiment(name, filters, epochs, batch_size=64, dropout=0.5):
    print(f"\n>>> Running experiment: {name}")

    train_loader, val_loader = load_mnist_data(resize_to_96=False, add_noise=True, noise_std=0.3)

    model = CNNModel(filters=filters, dropout=dropout)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{name}",
        filename=f"{name}-best-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        mode="min"
    )

    logger = CSVLogger("logs", name=name)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)
    model.evaluate_accuracy(val_loader)
    model.visualize_results(val_loader, save_path=f"{name}_conf_matrix_WhiteNoise_0.3.png")


def run_all_experiments():
    # Test z różną liczbą filtrów
    run_experiment("cnn_small", filters=[16, 32, 64], epochs=10)
    run_experiment("cnn_medium", filters=[32, 64, 128], epochs=10)
    run_experiment("cnn_large", filters=[64, 128, 256], epochs=10)

    # Test z różną liczbą epok
    run_experiment("cnn_epochs_5", filters=[32, 64, 128], epochs=5)
    run_experiment("cnn_epochs_15", filters=[32, 64, 128], epochs=15)

    # Test z mniejszym batch_size i większym dropoutem
    run_experiment("cnn_small_batch_dropout", filters=[32, 64, 128], epochs=10, batch_size=32, dropout=0.6)
