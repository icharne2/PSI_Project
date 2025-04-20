# Project description

This project was written using **PyTorch Lightning** and aims to **compare different deep learning models** for the task of **handwritten digit recognition (MNIST)**.

It includes:
- a custom Convolutional Neural Network (CNN)
- transfer learning using the **ResNet50** architecture
- a sequential **LSTM model**
- functions for **hyperparameter tuning**
- data analysis and result visualization
- model training, validation, and performance evaluation

## 📁 Project structure

| File | Description |
|------|-------------|
| `main.py` | Main script for training and evaluation |
| `cnn_model.py` | Custom CNN architecture |
| `resNet50_model.py` | Modified ResNet50 for grayscale input |
| `lstm_model.py` | LSTM model structure (to be developed) |
| `hyperparameter_tuning.py` | Tools for tuning model parameters |
| `analyze_data.py` | Data analysis and visualizations |
| `*.png` | Saved figures: confusion matrices, label distributions, example images |


## 🧠 Technology

- PyTorch
- PyTorch Lightning
- Matplotlib / Seaborn
- Pandas / Scikit-learn
