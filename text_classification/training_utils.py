# Why we are using it
# Class weights are used to balance the loss function(error function) when dealing with imbalanced datasets.

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate

# Loading the Accuracy Metric
metric = evaluate.load('accuracy')

# This function computes the accuracy metric for a given set of predictions and labels.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



# This function computes the class weights for a given dataset.
def get_class_weights(df):
    class_weights = compute_class_weight("balanced",
                         classes = sorted(df['label'].unique().tolist()),
                         y = df['label'].tolist()
                         )
    return class_weights