import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import joblib
import torch
import tqdm
from transformers import AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader

from src.models.model import HierarchicalProductClassifier
from src.data.dataset import ProductDataset
from src.data.data_preprocessing import preprocess_text

def evaluate_on_unseen_data(unseen_csv_path, model_path, le_paths, model_name='bert-base-uncased'):

    # Load unseen data
    unseen_df = pd.read_csv(unseen_csv_path)
    
    # Ensure 'text' column exists
    if 'Title' in unseen_df.columns and 'Text' in unseen_df.columns:
        unseen_df['text'] = unseen_df['Title'] + ' ' + unseen_df['Text']
    elif 'text' not in unseen_df.columns:
        raise ValueError("DataFrame must contain either 'text' column or both 'Title' and 'Text' columns")

    # Preprocess text
    unseen_df['text'] = unseen_df['text'].apply(preprocess_text)
    
    # Load label encoders
    label_encoders = [joblib.load(path) for path in le_paths]
    n_levels = len(label_encoders)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes_list = [len(le.classes_) for le in label_encoders]
    model = HierarchicalProductClassifier(n_classes_list, model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataset and dataloader
    unseen_dataset = ProductDataset(
        texts=unseen_df.text.to_numpy(),
        labels=np.zeros((len(unseen_df), n_levels)),  # Dynamic Dummy labels
        tokenizer=tokenizer,
        max_len=128
    )
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=16)
    
    # Evaluate
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(unseen_dataloader, desc="Evaluating on unseen data"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Predict for all levels
            batch_preds = []
            for i in range(n_levels):
                _, preds = torch.max(outputs[i], dim=1)
                batch_preds.append(preds)
            
            # Store predictions
            for i in range(len(batch_preds[0])):
                pred_dict = {'text': batch['text'][i]}
                for level in range(n_levels):
                    pred_dict[f'pred_cat{level+1}'] = label_encoders[level].inverse_transform([batch_preds[level][i].item()])[0]
                predictions.append(pred_dict)
    
    # Create DataFrame from predictions
    predictions_df = pd.DataFrame(predictions)
    
    pred_columns = [f'pred_cat{level+1}' for level in range(n_levels)]
    
    # Merge predictions with original data
    result_df = pd.concat([unseen_df, predictions_df[pred_columns]], axis=1)
    
    # Save predictions to CSV
    result_df.to_csv('unseen_data_predictions.csv', index=False)
    print("Predictions saved to 'unseen_data_predictions.csv'")

# Usage example
if __name__ == '__main__':
    unseen_csv_path = '/path/to/your/unseen_data.csv'  # Update this path
    model_path='best_model.pth'
    le_paths=['le1.joblib', 'le2.joblib', 'le3.joblib']
    evaluate_on_unseen_data(unseen_csv_path, model_path, le_paths)