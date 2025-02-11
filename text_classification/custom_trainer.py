import torch
from torch import nn
from transformers import Trainer 


# The CustomTrainer class is a custom implementation of the Trainer class from the Hugging Face Transformers library. 
# This class is designed to handle custom loss computation and device management.

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        # Forward Pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits = logits.float()
        
        # Ensure logits and labels are on the same device
        device = self.device if hasattr(self, 'device') else torch.device("cpu")
        logits = logits.to(device)
        labels = labels.to(device)
        
        # Compute Custom Loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float).to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights
    
    def set_device(self, device):
        self.device = device