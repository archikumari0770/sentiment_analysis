
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from config import config
from models.fusion_model import MultimodalFusionModel
from utils.data_loader import get_data_loader

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        visual = batch['visual'].to(device)
        audio = batch['audio'].squeeze(1).to(device)  # Remove extra dimension
        labels = batch['label'].to(device)
        
        # Skip if labels are invalid (dummy data)
        if (labels == -1).any():
            continue
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(visual, audio)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            visual = batch['visual'].to(device)
            audio = batch['audio'].squeeze(1).to(device)
            labels = batch['label'].to(device)
            
            if (labels == -1).any():
                continue
            
            outputs = model(visual, audio)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

def main():
    # Create directories
    config.create_directories()
    
    # Setup device
    device = config.DEVICE
    print(f'Using device: {device}')
    
    # Create data loaders
    train_loader = get_data_loader(
        video_dir=config.RAW_VIDEO_DIR,
        annotation_file=os.path.join(config.DATA_DIR, 'train_annotations.csv'),
        mode='train',
        shuffle=True
    )
    
    val_loader = get_data_loader(
        video_dir=config.RAW_VIDEO_DIR,
        annotation_file=os.path.join(config.DATA_DIR, 'val_annotations.csv'),
        mode='val',
        shuffle=False
    )
    
    # Create model
    model = MultimodalFusionModel(
        num_classes=config.NUM_CLASSES,
        fusion_type='attention'  # Options: 'concat', 'add', 'attention'
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(config.BASE_DIR, 'runs'))
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f'\\n--- Epoch {epoch}/{config.NUM_EPOCHS} ---')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Logging
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth'))
            print(f'✓ Saved best model with validation accuracy: {val_acc:.2f}%')
    
    writer.close()
    print('\\nTraining complete!')

if __name__ == '__main__':
    main()

