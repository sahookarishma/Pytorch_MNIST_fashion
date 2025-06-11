# Pytorch_MNIST_fashion
# 🧠 PyTorch Deep Learning Experiments

This repository contains a collection of PyTorch-based notebooks demonstrating core concepts in neural networks, autograd mechanics, and practical model training on datasets like Fashion-MNIST and breast cancer classification. These notebooks are structured to help beginners and intermediate learners understand the mechanics and best practices in model building using PyTorch.

---

## 📁 Contents

| Notebook Name                                | Description |
|---------------------------------------------|-------------|
| `Pytorch_autograd.ipynb`                     | Basic demonstration of PyTorch's `autograd` module, including how gradients are computed and visualized. |
| `PYTORCH_NN_module.ipynb`                    | An introduction to building neural networks using `nn.Module` and understanding layers, forward pass, and training loop. |
| `PYTORCH_fashion_mnist_ANN.ipynb`            | A simple artificial neural network (ANN) trained on Fashion-MNIST dataset using CPU. |
| `PYTORCH_fashion_mnist_ANN_usingGPU.ipynb`   | The same Fashion-MNIST ANN, this time with GPU acceleration using CUDA (if available). |
| `PYTORCH_fashion_mnist_Optimization.ipynb`   | Comparison of different optimization algorithms (e.g., SGD, Adam) in training Fashion-MNIST models. |
| `Pytorch_training_breastcancer.ipynb`        | A custom ANN built to classify breast cancer data using PyTorch, covering preprocessing, training, and evaluation. |

---

## 🛠️ Requirements

Install dependencies using pip:

```bash
pip install torch torchvision matplotlib scikit-learn pandas
🧪 Notebook Overview
🔹 Pytorch_autograd.ipynb
Introduction to PyTorch autograd

Understanding automatic differentiation

🔹 PYTORCH_NN_module.ipynb
Building neural networks using nn.Module

Forward propagation and model customization

🔹 PYTORCH_fashion_mnist_ANN.ipynb
Feedforward ANN on Fashion MNIST dataset

CPU-based training and evaluation

🔹 PYTORCH_fashion_mnist_ANN_usingGPU.ipynb
GPU-accelerated version of Fashion MNIST model

Model training and performance monitoring on CUDA

🔹 PYTORCH_fashion_mnist_Optimization.ipynb
Comparison of different optimizers (SGD, Adam, RMSProp)

Training loss tracking and visualization

🔹 Pytorch_training_breastcancer.ipynb
Application of PyTorch ANN to breast cancer classification

Data preprocessing and performance evaluation using accuracy, loss
