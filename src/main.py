import os
import sys
# Update sys.path to run main.py natively from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from src.data.data_preprocessing import preprocess_pipeline
from src.models.model import HierarchicalProductClassifier
from src.data.dataset import ProductDataset
from src.training.trainer import CustomTrainer
from src.evaluation.evaluation import evaluate_best_model
from src.utils.logger import logger

def main():
    logger.info("Initializing NLP Training Pipeline...")
    # Model config
    model_name = 'bert-base-uncased'
    category_cols = ['Cat1', 'Cat2', 'Cat3']  # Update this list to add/remove hierarchy levels

    # Data preprocessing
    csv_path = '/home/cdui/netflix_poc/chaitanya/Self_Supervised_Learning_Prod/data.csv'
    df, label_encoders = preprocess_pipeline(csv_path, category_cols=category_cols)

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Define encoded label columns for the dataset
    encoded_cols = [f'Cat{i+1}_encoded' for i in range(len(category_cols))]

    # Create datasets and dataloaders
    train_dataset = ProductDataset(
        texts=train_df.text.to_numpy(),
        labels=train_df[encoded_cols].values,
        tokenizer=tokenizer,
        max_len=128
    )
    
    val_dataset = ProductDataset(
        texts=val_df.text.to_numpy(),
        labels=val_df[encoded_cols].values,
        tokenizer=tokenizer,
        max_len=128
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model dynamically handling any number of levels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes_list = [len(le.classes_) for le in label_encoders]
    model = HierarchicalProductClassifier(n_classes_list, model_name=model_name)
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * 15  # 15 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Create and run custom trainer
    trainer = CustomTrainer(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=15, n_levels=len(n_classes_list))
    trainer.train()
    
    # Evaluate the best model
    avg_loss, all_metrics = evaluate_best_model(val_dataloader, label_encoders, model, device)

    print("\nBest Model Performance on Validation Data:")
    print(f"Validation Loss: {avg_loss:.4f}")
    
    for i, metrics in enumerate(all_metrics):
        print(f"\nCategory {i+1}:")
        print(f"  Accuracy: {metrics[0]:.2f}%")
        print(f"  Precision: {metrics[1]:.4f}")
        print(f"  Recall: {metrics[2]:.4f}")
        print(f"  F1 Score: {metrics[3]:.4f}")

if __name__ == '__main__':
    main()