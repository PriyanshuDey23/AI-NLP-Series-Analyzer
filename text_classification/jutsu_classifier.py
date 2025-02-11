import pandas as pd
import torch
import huggingface_hub
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          pipeline)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
import gc
from cleaner import Cleaner  # Ensure proper import path
from training_utils import get_class_weights, compute_metrics  # Ensure these exist
from custom_trainer import CustomTrainer  # Ensure correct import path
from huggingface_hub import model_info

# Function to check if a model exists on Hugging Face Hub
def repo_exists(model_path):
    try:
        model_info(model_path)
        return True
    except:
        return False

# Classification
class JutsuClassifier():
    def __init__(self,
                 model_path,
                 data_path=None,
                 text_column_name='text',
                 label_column_name='jutsu',
                 model_name="distilbert/distilbert-base-uncased",
                 test_size=0.2,
                 num_labels=3,
                 huggingface_token=None):
        
        # Self variables
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Assign Hugging Face token
        self.huggingface_token = huggingface_token
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
        
        self.tokenizer = self.load_tokenizer()

        # Check if the model exists on Hugging Face Hub
        if not repo_exists(self.model_path):
            # Ensure data path is provided for training
            if data_path is None:
                raise ValueError("Data path is required to train the model, since the model path does not exist in Hugging Face Hub")
            
            # Load and preprocess data
            train_data, test_data = self.load_data(self.data_path)
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()
            
            # Compute class weights
            all_data = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data)
            
            # Train the model
            self.train_model(train_data, test_data, class_weights)

        # Load trained model
        self.model = self.load_model(self.model_path)
    
    # Load model from Hugging Face
    def load_model(self, model_path):
        return pipeline('text-classification', model=model_path, return_all_scores=True)
    
    # Train the model
    def train_model(self, train_data, test_data, class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels,
            id2label=self.label_dict,  # Ensuring label dictionary exists
            label2id={v: k for k, v in self.label_dict.items()}  # Fixing missing label2id
        )
        
        # Convert class weights to tensor and move to correct device
        class_weights = torch.tensor(class_weights).to(self.device)
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir=self.model_path,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            push_to_hub=True,
        )
        
        # Custom trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        trainer.set_device(self.device)
        trainer.set_class_weights(class_weights)
        trainer.train()
        
        # Free memory
        del trainer, model
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
    
    # Simplify jutsu types
    def simplify_jutsu(self, jutsu):
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"
        return "Unknown"  # Fix: Ensure default return value
    
    # Preprocess function for tokenization
    def preprocess_function(self, tokenizer, examples):
        return tokenizer(examples['text_cleaned'], truncation=True)
    
    # Load and preprocess dataset
    def load_data(self, data_path):
        df = pd.read_json(data_path, lines=True)
        
        df['jutsu_type_simplified'] = df['jutsu_type'].apply(self.simplify_jutsu)
        df['text'] = df['jutsu_name'] + ". " + df['jutsu_description']
        df[self.label_column_name] = df['jutsu_type_simplified']
        df = df[['text', self.label_column_name]].dropna()
        
        # Clean text
        cleaner = Cleaner()
        df['text_cleaned'] = df[self.text_column_name].apply(cleaner.clean)
        
        # Encode labels
        le = preprocessing.LabelEncoder()
        le.fit(df[self.label_column_name].tolist())
        
        self.label_dict = {index: label for index, label in enumerate(le.classes_)}  # Ensure label dictionary exists
        df['label'] = le.transform(df[self.label_column_name].tolist())
        
        # Split dataset
        df_train, df_test = train_test_split(df, test_size=self.test_size, stratify=df['label'])
        
        # Convert to Hugging Face dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)
        
        # Tokenize dataset
        tokenized_train = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                            batched=True,
                                            remove_columns=['text', self.label_column_name])
        tokenized_test = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                          batched=True,
                                          remove_columns=['text', self.label_column_name])
        
        return tokenized_train, tokenized_test
    
    # Load tokenizer
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_path if repo_exists(self.model_path) else self.model_name)
    
    # Postprocess model output
    def postprocess(self, model_output):
        return [max(pred, key=lambda x: x['score'])['label'] for pred in model_output]
    
    # Inference function
    def classify_jutsu(self, text):
        model_output = self.model(text)
        return self.postprocess(model_output)
