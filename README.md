# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Project Overview

This project builds and evaluates a feed-forward neural network for a binary classification task: predicting customer churn. The notebook uses TensorFlow/Keras.

Dataset: `customer_churn_nn.csv`  
Target variable: `churn` (`1` = churned, `0` = retained)

## Repository Structure

```text
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.csv
    ├── model_comparison_table.png
    └── evaluation_outputs.png
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Place `customer_churn_nn.csv` in the same directory as the notebook before running it.

## Assignment Coverage

### Task 1: Dataset Understanding

The notebook reports the dataset shape, data types, missing values, statistical summary, and target distribution.

### Task 2: Data Preprocessing

The notebook drops the identifier column, encodes categorical columns, performs a stratified train/test split, and scales the numerical feature matrix. Scaling is fitted on the training set only to avoid test-set leakage.

### Task 3: Neural Network Model Building

The model uses dense input, hidden, and output layers. Hidden layers use activation functions such as ReLU or tanh, and the binary target is modeled with a sigmoid output layer, binary cross-entropy loss, and the Adam optimizer.

### Task 4: Training and Evaluation

The notebook reports training and testing loss/accuracy, ROC-AUC, a classification report, and a confusion matrix. Because the dataset is highly imbalanced, ROC-AUC, recall, precision, and F1 are interpreted alongside accuracy.

### Task 5: Hyperparameter Experimentation

Five configurations are tested by changing hidden layers, number of neurons, learning rate, batch size, and activation function. Results are saved in `results/model_comparison_table.csv` and `results/model_comparison_table.png`.

### Task 6: Final Reflection

The notebook explains weights and biases, activation functions, learning-rate behavior, and underfitting/overfitting observations.

## Result Summary

The saved experiment table shows that different metrics favor different configurations. Config 4 produced the strongest ROC-AUC, Config 1 produced the strongest churn F1 score, and Config 2 produced the strongest churn recall. Accuracy is high across several models because churn is rare, so minority-class metrics are more informative than accuracy alone.
