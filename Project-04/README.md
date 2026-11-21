
# GPU-Accelerated Deep Residual Intrusion Detection System

## Project Overview

This project develops a Machine Learning based Intrusion Detection System using PyTorch and CUDA.

The system analyzes network traffic records from the CICIDS2017 dataset and classifies them into seven categories.

## Attack Categories

1. BENIGN
2. Botnet
3. BruteForce
4. DoS/DDoS
5. Infiltration
6. PortScan
7. Web Attack

## Technologies

- Python 3.11
- PyTorch
- CUDA
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Git and GitHub

## Main Objectives

- Preprocess network traffic data.
- Build a Deep Residual Neural Network.
- Train the model on CPU and GPU.
- Compare training and inference performance.
- Measure GPU speedup.
- Evaluate classification accuracy.
- Save the trained model and performance charts.

## Project Structure

```text
Project-04/
│
├── src/
│   └── final_ids/
│       └── final_ids.py
│
├── data/
│   └── cicids2017/
│
├── models/
│
├── docs/
│
└── README.md
```

## Team

Parallel and Distributed Computing Lab  
Team Members: 3

## Dataset

CICIDS2017 - Canadian Institute for Cybersecurity.
