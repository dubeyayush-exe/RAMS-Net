
# ============================================================================
# 4. TRAINING PIPELINE
# ============================================================================

class DisasterResponseTrainer:
    """Training pipeline for the multimodal disaster response model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        # Loss functions
        self.severity_criterion = nn.CrossEntropyLoss()
        self.evacuation_criterion = nn.CrossEntropyLoss()
        self.priority_criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate losses
            loss_severity = self.severity_criterion(
                outputs['severity'], batch['labels'][:, 0]
            )
            loss_evacuation = self.evacuation_criterion(
                outputs['evacuation'], batch['labels'][:, 1]
            )
            loss_priority = self.priority_criterion(
                outputs['priority'], batch['labels'][:, 2]
            )
            
            # Combined loss with weights
            loss = 0.4 * loss_severity + 0.3 * loss_evacuation + 0.3 * loss_priority
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        correct = {'severity': 0, 'evacuation': 0, 'priority': 0}
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch)
                
                # Calculate accuracy for each task
                severity_pred = outputs['severity'].argmax(dim=1)
                evacuation_pred = outputs['evacuation'].argmax(dim=1)
                priority_pred = outputs['priority'].argmax(dim=1)
                
                correct['severity'] += (severity_pred == batch['labels'][:, 0]).sum().item()
                correct['evacuation'] += (evacuation_pred == batch['labels'][:, 1]).sum().item()
                correct['priority'] += (priority_pred == batch['labels'][:, 2]).sum().item()
                
                total += batch['labels'].size(0)
        
        return {
            'severity_acc': correct['severity'] / total,
            'evacuation_acc': correct['evacuation'] / total,
            'priority_acc': correct['priority'] / total
        }
