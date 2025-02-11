# Why we are using it
# Class weights are used to balance the loss function(error function) when dealing with imbalanced datasets.

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate
import torch

# Loading the Accuracy Metric
metric = evaluate.load('accuracy')

# This function computes the accuracy metric for a given set of predictions and labels.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Make sure logits is of correct shape and take argmax for predictions
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# This function computes the class weights for a given dataset.
def get_class_weights(df):
    # Ensure the labels are numeric (int type) and get class weights
    class_weights = compute_class_weight(
        "balanced",
        classes=sorted(df['label'].unique().tolist()),  # Sort to match class order
        y=df['label'].tolist()
    )
    
    # Convert class weights to a PyTorch tensor if needed for model loss function
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    return class_weights_tensor