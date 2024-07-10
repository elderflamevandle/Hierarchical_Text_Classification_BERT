import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess_pipeline
from model import HierarchicalProductClassifier
from dataset import ProductDataset
from trainer import CustomTrainer
from evaluation import evaluate_best_model

def main():
    # Data preprocessing
    csv_path = '/home/cdui/netflix_poc/chaitanya/Self_Supervised_Learning_Prod/data.csv'
    df, le1, le2, le3 = preprocess_pipeline(csv_path)

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets and dataloaders
    train_dataset = ProductDataset(
        texts=train_df.text.to_numpy(),
        labels=train_df[['Cat1_encoded', 'Cat2_encoded', 'Cat3_encoded']].values,
        tokenizer=tokenizer,
        max_len=128
    )
    
    val_dataset = ProductDataset(
        texts=val_df.text.to_numpy(),
        labels=val_df[['Cat1_encoded', 'Cat2_encoded', 'Cat3_encoded']].values,
        tokenizer=tokenizer,
        max_len=128
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalProductClassifier(len(le1.classes_), len(le2.classes_), len(le3.classes_))
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
    trainer = CustomTrainer(model, train_dataloader, val_dataloader, optimizer, scheduler, device)
    trainer.train()
    
    # Evaluate the best model
    avg_loss, cat1_metrics, cat2_metrics, cat3_metrics = evaluate_best_model(val_dataloader, le1, le2, le3, model, device)

    print("\nBest Model Performance on Validation Data:")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"\nCategory 1:")
    print(f"  Accuracy: {cat1_metrics[0]:.2f}%")
    print(f"  Precision: {cat1_metrics[1]:.4f}")
    print(f"  Recall: {cat1_metrics[2]:.4f}")
    print(f"  F1 Score: {cat1_metrics[3]:.4f}")
    print(f"\nCategory 2:")
    print(f"  Accuracy: {cat2_metrics[0]:.2f}%")
    print(f"  Precision: {cat2_metrics[1]:.4f}")
    print(f"  Recall: {cat2_metrics[2]:.4f}")
    print(f"  F1 Score: {cat2_metrics[3]:.4f}")
    print(f"\nCategory 3:")
    print(f"  Accuracy: {cat3_metrics[0]:.2f}%")
    print(f"  Precision: {cat3_metrics[1]:.4f}")
    print(f"  Recall: {cat3_metrics[2]:.4f}")
    print(f"  F1 Score: {cat3_metrics[3]:.4f}")

if __name__ == '__main__':
    main()