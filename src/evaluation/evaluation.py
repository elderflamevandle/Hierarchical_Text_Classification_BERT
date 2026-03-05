import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

def evaluate_best_model(val_dataloader, label_encoders, model, device, model_path='best_model.pth'):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    n_levels = len(label_encoders)
    loss_fns = [nn.CrossEntropyLoss().to(device) for _ in range(n_levels)]

    total_loss = 0
    all_preds = [[] for _ in range(n_levels)]
    all_labels = [[] for _ in range(n_levels)]
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating best model"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = 0
            for i in range(n_levels):
                l = loss_fns[i](outputs[i], labels[:, i])
                loss += l
                
                _, preds = torch.max(outputs[i], dim=1)
                all_preds[i].extend(preds.cpu().numpy())
                all_labels[i].extend(labels[:, i].cpu().numpy())
                
            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)

    all_metrics = []
    for i in range(n_levels):
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels[i], all_preds[i], average='weighted')
        acc = 100 * sum(l == p for l, p in zip(all_labels[i], all_preds[i])) / len(all_labels[i])
        all_metrics.append((acc, precision, recall, f1))

    return avg_loss, all_metrics