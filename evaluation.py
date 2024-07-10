import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

def evaluate_best_model(val_dataloader, le1, le2, le3, model, device, model_path='best_model.pth'):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    loss_fn1 = nn.CrossEntropyLoss().to(device)
    loss_fn2 = nn.CrossEntropyLoss().to(device)
    loss_fn3 = nn.CrossEntropyLoss().to(device)

    total_loss = 0
    all_preds1, all_preds2, all_preds3 = [], [], []
    all_labels1, all_labels2, all_labels3 = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating best model"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs1, outputs2, outputs3 = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss1 = loss_fn1(outputs1, labels[:, 0])
            loss2 = loss_fn2(outputs2, labels[:, 1])
            loss3 = loss_fn3(outputs3, labels[:, 2])
            loss = loss1 + loss2 + loss3
            
            total_loss += loss.item()

            _, preds1 = torch.max(outputs1, dim=1)
            _, preds2 = torch.max(outputs2, dim=1)
            _, preds3 = torch.max(outputs3, dim=1)
            
            all_preds1.extend(preds1.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_preds3.extend(preds3.cpu().numpy())
            all_labels1.extend(labels[:, 0].cpu().numpy())
            all_labels2.extend(labels[:, 1].cpu().numpy())
            all_labels3.extend(labels[:, 2].cpu().numpy())

    avg_loss = total_loss / len(val_dataloader)

    precision1, recall1, f1_1, _ = precision_recall_fscore_support(all_labels1, all_preds1, average='weighted')
    precision2, recall2, f1_2, _ = precision_recall_fscore_support(all_labels2, all_preds2, average='weighted')
    precision3, recall3, f1_3, _ = precision_recall_fscore_support(all_labels3, all_preds3, average='weighted')

    acc1 = 100 * sum(l == p for l, p in zip(all_labels1, all_preds1)) / len(all_labels1)
    acc2 = 100 * sum(l == p for l, p in zip(all_labels2, all_preds2)) / len(all_labels2)
    acc3 = 100 * sum(l == p for l, p in zip(all_labels3, all_preds3)) / len(all_labels3)

    return avg_loss, (acc1, precision1, recall1, f1_1), (acc2, precision2, recall2, f1_2), (acc3, precision3, recall3, f1_3)