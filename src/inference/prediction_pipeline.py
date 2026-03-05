import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import joblib
import torch
from transformers import AutoTokenizer

from src.models.model import HierarchicalProductClassifier
from src.data.data_preprocessing import preprocess_text

def prediction_pipeline(text1, model_name='bert-base-uncased', model_path='best_model.pth'):
    # Preprocessing of Text
    text = preprocess_text(text1)

    # Load label encoders dynamically
    label_encoders = []
    i = 1
    while os.path.exists(f'le{i}.joblib'):
        le = joblib.load(f'le{i}.joblib')
        label_encoders.append(le)
        i += 1
        
    if not label_encoders:
        raise Exception("No label encoders found (le*.joblib). Have you trained the model?")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes_list = [len(le.classes_) for le in label_encoders]
    model = HierarchicalProductClassifier(n_classes_list, model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Preprocess and predict
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    predictions = []
    for i in range(len(label_encoders)):
        _, preds = torch.max(outputs[i], dim=1)
        cat = label_encoders[i].inverse_transform(preds.cpu().numpy())[0]
        predictions.append(cat)
    
    return predictions

def main():
    # Prediction example  - to predict it with just a single sentence
    sample_text = "Simple Solution Washable Male Wrap  My dog is disabled and this wrap is a Godsend. Without it I wouldn't be able to keep my dogs diapers in place. This product has made a near impossible situation workable."

    predicted_categories = prediction_pipeline(sample_text)
    
    print("Predicted categories:")
    for i, cat in enumerate(predicted_categories):
        print(f"  Cat{i+1}: {cat}")

# Main execution
if __name__ == '__main__':
    main()