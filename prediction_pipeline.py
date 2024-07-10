import pandas as pd
import joblib
import torch
from transformers import BertTokenizer

from model import HierarchicalProductClassifier
from data_preprocessing import preprocess_text

def prediction_pipeline(text1, model_path='best_model.pth'):

    # Preprocessing of Text
    text = preprocess_text(text1)

    # Load label encoders
    le1 = joblib.load('le1.joblib')
    le2 = joblib.load('le2.joblib')
    le3 = joblib.load('le3.joblib')
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalProductClassifier(len(le1.classes_), len(le2.classes_), len(le3.classes_))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
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
        outputs1, outputs2, outputs3 = model(input_ids=input_ids, attention_mask=attention_mask)
    
    _, preds1 = torch.max(outputs1, dim=1)
    _, preds2 = torch.max(outputs2, dim=1)
    _, preds3 = torch.max(outputs3, dim=1)
    
    cat1 = le1.inverse_transform(preds1.cpu().numpy())
    cat2 = le2.inverse_transform(preds2.cpu().numpy())
    cat3 = le3.inverse_transform(preds3.cpu().numpy())
    
    return cat1[0], cat2[0], cat3[0]

def main():
    # Prediction example  - to predict it with just a single sentence
    sample_text = "Simple Solution Washable Male Wrap  My dog is disabled and this wrap is a Godsend. Without it I wouldn't be able to keep my dogs diapers in place. This product has made a near impossible situation workable."

    cat1, cat2, cat3 = prediction_pipeline(sample_text)
    print(f"Predicted categories: Cat1: {cat1}, Cat2: {cat2}, Cat3: {cat3}")


# Main execution
if __name__ == '__main__':
    main()