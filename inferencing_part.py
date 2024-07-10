import pandas as pd
import joblib
import torch
import tqdm
from transformers import BertTokenizer
import numpy as np
from torch.utils.data import DataLoader

from model import HierarchicalProductClassifier
from dataset import ProductDataset
from data_preprocessing import preprocess_text

def evaluate_on_unseen_data(unseen_csv_path, model_path, le1_path, le2_path, le3_path):

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
    le1 = joblib.load(le1_path)
    le2 = joblib.load(le2_path)
    le3 = joblib.load(le3_path)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalProductClassifier(len(le1.classes_), len(le2.classes_), len(le3.classes_))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset and dataloader
    unseen_dataset = ProductDataset(
        texts=unseen_df.text.to_numpy(),
        labels=np.zeros((len(unseen_df), 3)),  # Dummy labels
        tokenizer=tokenizer,
        max_len=128
    )
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=16)
    
    # Evaluate
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(unseen_dataloader, desc="Evaluating on unseen data"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs1, outputs2, outputs3 = model(input_ids=input_ids, attention_mask=attention_mask)
            
            _, preds1 = torch.max(outputs1, dim=1)
            _, preds2 = torch.max(outputs2, dim=1)
            _, preds3 = torch.max(outputs3, dim=1)
            
            # Store predictions
            for i in range(len(preds1)):
                predictions.append({
                    'text': batch['text'][i],
                    'pred_cat1': le1.inverse_transform([preds1[i].item()])[0],
                    'pred_cat2': le2.inverse_transform([preds2[i].item()])[0],
                    'pred_cat3': le3.inverse_transform([preds3[i].item()])[0]
                })
    
    # Create DataFrame from predictions
    predictions_df = pd.DataFrame(predictions)
    
    # Merge predictions with original data
    result_df = pd.concat([unseen_df, predictions_df[['pred_cat1', 'pred_cat2', 'pred_cat3']]], axis=1)
    
    # Save predictions to CSV
    result_df.to_csv('unseen_data_predictions.csv', index=False)
    print("Predictions saved to 'unseen_data_predictions.csv'")

# Usage example
if __name__ == '__main__':
    unseen_csv_path = '/path/to/your/unseen_data.csv'  # Update this path
    model_path='best_model.pth', 
    le1_path='le1.joblib', 
    le2_path='le2.joblib', 
    le3_path='le3.joblib'
    evaluate_on_unseen_data(unseen_csv_path, model_path, le1_path, le2_path, le3_path)