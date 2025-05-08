from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os


def get_transform(model_class):
    """Wybiera odpowiednią transformację w zależności od typu modelu"""
    if model_class == "ResNet50Classifier":
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:  # CNN, LSTM, inne modele 28x28
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


def predict_single_image(model, image_path):
    model.eval()
    transform = get_transform(model.__class__.__name__)
    try:
        img = Image.open(image_path)
        img_tensor = transform(img).unsqueeze(0).to(model.device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()

        print(f"Model przewidział cyfrę: {pred}")
        return pred
    except Exception as e:
        print(f"Błąd przetwarzania obrazu {image_path}: {e}")
        return None


def predict_images_batch(model, image_paths, true_labels=None):
    model.eval()
    transform = get_transform(model.__class__.__name__)

    preds = []
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    for i, path in enumerate(valid_paths):
        try:
            img = Image.open(path)
            img_tensor = transform(img).unsqueeze(0).to(model.device)

            with torch.no_grad():
                output = model(img_tensor)
                pred = output.argmax(dim=1).item()
                preds.append(pred)
                print(f"[{i+1}] Obraz: {path} => Predykcja: {pred}")
        except Exception as e:
            print(f"Błąd przetwarzania obrazu {path}: {e}")
            preds.append(None)

    if true_labels is not None:
        cm = confusion_matrix(true_labels, preds, labels=range(10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Batch Prediction")
        plt.show()

    return preds
