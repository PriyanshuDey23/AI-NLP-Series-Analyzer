# Why we are using it
# Class weights are used to balance the loss function (error function) when dealing with imbalanced datasets.

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate
import torch

# Loading the Accuracy Metric
metric = evaluate.load('accuracy')

# Function to compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Function to compute class weights
def get_class_weights(df):
    class_labels = np.array(sorted(df['label'].unique()))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=df['label'].values
    )
    return torch.tensor(class_weights, dtype=torch.float, requires_grad=False)