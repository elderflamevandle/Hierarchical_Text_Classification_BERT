import torch
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

class CustomTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=20, n_levels=3):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.n_levels = n_levels
        
        self.loss_fns = [nn.CrossEntropyLoss().to(device) for _ in range(n_levels)]
        
        self.best_val_loss = float('inf')
        self.patience = 3
        self.counter = 0

        # AMP Scaler for mixed precision
        self.scaler = GradScaler()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        level_losses = [0] * self.n_levels
        level_corrects = [0] * self.n_levels
        level_totals = [0] * self.n_levels
        
        for batch in tqdm(self.train_dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                loss = 0
                batch_level_losses = []
                for i in range(self.n_levels):
                    l = self.loss_fns[i](outputs[i], labels[:, i])
                    loss += l
                    batch_level_losses.append(l)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            
            for i in range(self.n_levels):
                level_losses[i] += batch_level_losses[i].item()
                preds = outputs[i].argmax(dim=1, keepdim=True)
                level_corrects[i] += preds.eq(labels[:, i].view_as(preds)).sum().item()
                level_totals[i] += labels[:, i].size(0)
        
        avg_loss = total_loss / len(self.train_dataloader)
        avg_level_losses = [l / len(self.train_dataloader) for l in level_losses]
        accuracies = [100. * c / t for c, t in zip(level_corrects, level_totals)]
        
        return avg_loss, avg_level_losses, accuracies

    def validate(self):
        self.model.eval()
        total_loss = 0
        level_losses = [0] * self.n_levels
        level_corrects = [0] * self.n_levels
        level_totals = [0] * self.n_levels
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    loss = 0
                    batch_level_losses = []
                    for i in range(self.n_levels):
                        l = self.loss_fns[i](outputs[i], labels[:, i])
                        loss += l
                        batch_level_losses.append(l)
                
                total_loss += loss.item()
                
                for i in range(self.n_levels):
                    level_losses[i] += batch_level_losses[i].item()
                    preds = outputs[i].argmax(dim=1, keepdim=True)
                    level_corrects[i] += preds.eq(labels[:, i].view_as(preds)).sum().item()
                    level_totals[i] += labels[:, i].size(0)
        
        avg_loss = total_loss / len(self.val_dataloader)
        avg_level_losses = [l / len(self.val_dataloader) for l in level_losses]
        accuracies = [100. * c / t for c, t in zip(level_corrects, level_totals)]
        
        return avg_loss, avg_level_losses, accuracies

    def train(self):
        for epoch in range(self.num_epochs):
            print("\n")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_level_losses, train_accs = self.train_epoch()
            val_loss, val_level_losses, val_accs = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            train_loss_str = ", ".join([f"Cat{i+1}: {l:.4f}" for i, l in enumerate(train_level_losses)])
            val_loss_str = ", ".join([f"Cat{i+1}: {l:.4f}" for i, l in enumerate(val_level_losses)])
            print(f"Train Level Losses: {train_loss_str}")
            print(f"Val Level Losses: {val_loss_str}")
            
            train_acc_str = ", ".join([f"Cat{i+1}: {a:.2f}%" for i, a in enumerate(train_accs)])
            val_acc_str = ", ".join([f"Cat{i+1}: {a:.2f}%" for i, a in enumerate(val_accs)])
            print(f"Train Accuracies: {train_acc_str}")
            print(f"Val Accuracies: {val_acc_str}")
            
            # Early stopping based on global validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("-> Saved best model!")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break
