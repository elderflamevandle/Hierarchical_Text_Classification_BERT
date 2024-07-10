import torch
from torch import nn
from tqdm import tqdm

class CustomTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=20):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        
        self.loss_fn1 = nn.CrossEntropyLoss().to(device)
        self.loss_fn2 = nn.CrossEntropyLoss().to(device)
        self.loss_fn3 = nn.CrossEntropyLoss().to(device)
        
        self.best_val_loss = float('inf')
        self.patience = 3
        self.counter = 0
        
        self.best_cat1_loss = float('inf')
        self.cat1_patience = 3
        self.cat1_counter = 0
        self.train_cat1 = True

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_loss1, total_loss2, total_loss3 = 0, 0, 0
        correct1, correct2, correct3 = 0, 0, 0
        total1, total2, total3 = 0, 0, 0
        
        for batch in tqdm(self.train_dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs1, outputs2, outputs3 = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss1 = self.loss_fn1(outputs1, labels[:, 0]) if self.train_cat1 else 0
            loss2 = self.loss_fn2(outputs2, labels[:, 1])
            loss3 = self.loss_fn3(outputs3, labels[:, 2])
            
            loss = loss1 + loss2 + loss3
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_loss1 += loss1.item() if self.train_cat1 else 0
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()

            # Calculate accuracies
            pred1 = outputs1.argmax(dim=1, keepdim=True)
            pred2 = outputs2.argmax(dim=1, keepdim=True)
            pred3 = outputs3.argmax(dim=1, keepdim=True)
            
            correct1 += pred1.eq(labels[:, 0].view_as(pred1)).sum().item()
            correct2 += pred2.eq(labels[:, 1].view_as(pred2)).sum().item()
            correct3 += pred3.eq(labels[:, 2].view_as(pred3)).sum().item()
            
            total1 += labels[:, 0].size(0)
            total2 += labels[:, 1].size(0)
            total3 += labels[:, 2].size(0)
        
        avg_loss = total_loss / len(self.train_dataloader)
        avg_loss1 = total_loss1 / len(self.train_dataloader) if self.train_cat1 else 0
        avg_loss2 = total_loss2 / len(self.train_dataloader)
        avg_loss3 = total_loss3 / len(self.train_dataloader)
        
        acc1 = 100. * correct1 / total1
        acc2 = 100. * correct2 / total2
        acc3 = 100. * correct3 / total3
        
        return avg_loss, avg_loss1, avg_loss2, avg_loss3, acc1, acc2, acc3

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_loss1, total_loss2, total_loss3 = 0, 0, 0
        correct1, correct2, correct3 = 0, 0, 0
        total1, total2, total3 = 0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs1, outputs2, outputs3 = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                loss1 = self.loss_fn1(outputs1, labels[:, 0])
                loss2 = self.loss_fn2(outputs2, labels[:, 1])
                loss3 = self.loss_fn3(outputs3, labels[:, 2])
                
                loss = loss1 + loss2 + loss3
                
                total_loss += loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_loss3 += loss3.item()

                # Calculate accuracies
                pred1 = outputs1.argmax(dim=1, keepdim=True)
                pred2 = outputs2.argmax(dim=1, keepdim=True)
                pred3 = outputs3.argmax(dim=1, keepdim=True)
                
                correct1 += pred1.eq(labels[:, 0].view_as(pred1)).sum().item()
                correct2 += pred2.eq(labels[:, 1].view_as(pred2)).sum().item()
                correct3 += pred3.eq(labels[:, 2].view_as(pred3)).sum().item()
                
                total1 += labels[:, 0].size(0)
                total2 += labels[:, 1].size(0)
                total3 += labels[:, 2].size(0)
        
        avg_loss = total_loss / len(self.val_dataloader)
        avg_loss1 = total_loss1 / len(self.val_dataloader)
        avg_loss2 = total_loss2 / len(self.val_dataloader)
        avg_loss3 = total_loss3 / len(self.val_dataloader)
        
        acc1 = 100. * correct1 / total1
        acc2 = 100. * correct2 / total2
        acc3 = 100. * correct3 / total3
        
        return avg_loss, avg_loss1, avg_loss2, avg_loss3, acc1, acc2, acc3

    def train(self):
        for epoch in range(self.num_epochs):
            print("\n")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_loss1, train_loss2, train_loss3, train_acc1, train_acc2, train_acc3 = self.train_epoch()
            val_loss, val_loss1, val_loss2, val_loss3, val_acc1, val_acc2, val_acc3 = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Cat1 Loss: {train_loss1:.4f}, Cat2 Loss: {train_loss2:.4f}, Cat3 Loss: {train_loss3:.4f}")
            print(f"Val Cat1 Loss: {val_loss1:.4f}, Val Cat2 Loss: {val_loss2:.4f}, Val Cat3 Loss: {val_loss3:.4f}")
            print(f"Train Accuracies: Cat1: {train_acc1:.2f}%, Cat2: {train_acc2:.2f}%, Cat3: {train_acc3:.2f}%")
            print(f"Val Accuracies: Cat1: {val_acc1:.2f}%, Cat2: {val_acc2:.2f}%, Cat3: {val_acc3:.2f}%")
            
            # Early stopping based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping")
                    break
            
            # Stop training Cat1 if its loss is not reducing
            if self.train_cat1:
                if train_loss1 < self.best_cat1_loss:
                    self.best_cat1_loss = train_loss1
                    self.cat1_counter = 0
                else:
                    self.cat1_counter += 1
                    if self.cat1_counter >= self.cat1_patience:
                        print("Stopping training for Cat1")
                        self.train_cat1 = False
