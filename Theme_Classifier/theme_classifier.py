from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
import numpy as np
import os
import sys
import pathlib
import os
import pandas as pd
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from utils import load_subtitles_dataset

# Folder Path
folder_path = pathlib.Path(__file__).parent.resolve()  # Get the Parent Directory Path
sys.path.append(os.path.join(folder_path, '../'))  # Append the Parent Directory to the System Path


# Class
class ThemeClassifier:
    def __init__(self, theme_list):  # Theme List
        self.model_name = "facebook/bart-large-mnli"  # Model
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)  # Load the model

    # Load the model
    def load_model(self, device):
        theme_classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )
        return theme_classifier

    # Combine all three
    def get_themes_inference(self, script):
        script_sentences = sent_tokenize(script)  # sent_tokenize(Sentence Tokenize) Convert the script to sentences

        # Batch Sentence (Convert the sentences to a batch of sentences)
        sentence_batch_size = 20
        script_batches = []
        for index in range(0, len(script_sentences), sentence_batch_size):
            sent = " ".join(script_sentences[index:index + sentence_batch_size])
            script_batches.append(sent)

        # Run Model
        # Emotions will be detected based on the batched data
        theme_output = self.theme_classifier(
            script_batches,
            self.theme_list,  # Emotions list
            multi_label=True
        )

        # Wrangle Output
        # Now we will get the emotions for all sentences
        # Then we will find the mean under each category by combining all the scores and save it under the category
        themes = {}
        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)

        themes = {key: np.mean(np.array(value)) for key, value in themes.items()}

        return themes

    def get_themes(self, dataset_path, save_path=None):
        # Read Save Output if Exists
        if save_path is not None:
            # Ensure save_path is a file and not a directory
            if os.path.isdir(save_path):
                raise ValueError(f"Expected a file path, but got a directory: {save_path}")
            
            if os.path.exists(save_path):
                df = pd.read_csv(save_path)
                return df

        # Load Dataset
        df = load_subtitles_dataset(dataset_path)  # From utils

        # Run Inference
        output_themes = df['script'].apply(self.get_themes_inference)  # On all the scripts apply get_themes_inference

        # Convert to dataframe
        themes_df = pd.DataFrame(output_themes.tolist())
        df[themes_df.columns] = themes_df

        # Save output
        if save_path is not None:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)

        return df
