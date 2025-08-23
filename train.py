import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
import os

from models import BrainAneurysmCNN, AneurysmLoss
from datasets import BrainAneurysmDataset, create_datasets

from pathlib import Path
import json
from tqdm import tqdm
from metrics import compute_accuracy_metrics, compute_recall_metrics

import wandb  # 🎯 Weights & Biases for logging

def get_device():
    """Detect the best available device: MPS > CUDA > CPU"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using MacBook Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device

def train_step(model, batch, criterion, optimizer, device):
    """Single training step"""
    model.train()
    
    # Move data to device
    images = batch['image'].to(device)
    aneurysm_present = batch['aneurysm_present'].to(device)
    aneurysm_locations = batch['aneurysm_locations'].to(device)
    
    # Forward pass
    optimizer.zero_grad()
    predictions = model(aneurysm_locations)

    # Prepare targets
    targets = {
        'aneurysm_present': aneurysm_present,
        'aneurysm_locations': aneurysm_locations
    }
    
    # Compute loss
    loss_dict = criterion(predictions, targets)
    
    # Backward pass
    loss_dict['total_loss'].backward()
    optimizer.step()
    
    return loss_dict, predictions

def validate_step(model, batch, criterion, device):
    """Single validation step"""
    model.eval()
    
    with torch.no_grad():
        # Move data to device
        images = batch['image'].to(device)
        aneurysm_present = batch['aneurysm_present'].to(device)
        aneurysm_locations = batch['aneurysm_locations'].to(device)
        
        # Forward pass
        predictions = model(images)
        
        # Prepare targets
        targets = {
            'aneurysm_present': aneurysm_present,
            'aneurysm_locations': aneurysm_locations
        }
        
        # Compute loss
        loss_dict = criterion(predictions, targets)
    
    return loss_dict, predictions

def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    num_epochs=10,
    log_every=10,
    save_every=5
):
    """
    Complete training loop with step-by-step loss logging
    """
    print(f"🚀 Starting training on {device}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_presence_losses': [],
        'train_location_losses': [],
        'val_presence_losses': [],
        'val_location_losses': []
    }
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Training phase
        model.train()
        train_losses = []
        train_presence_losses = []
        train_location_losses = []
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch in train_pbar:
            global_step += 1
            
            # Training step
            loss_dict, predictions = train_step(model, batch, criterion, optimizer, device)
            
            # Log losses
            total_loss = loss_dict['total_loss'].item()
            presence_loss = loss_dict['presence_loss'].item()
            location_loss = loss_dict['location_loss'].item()
            
            train_losses.append(total_loss)
            train_presence_losses.append(presence_loss)
            train_location_losses.append(location_loss)
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{total_loss:.4f}',
                'Pres': f'{presence_loss:.4f}',
                'Loc': f'{location_loss:.4f}'
            })
            
            # Log every N steps
            if global_step % log_every == 0:
                print(f"\nStep {global_step}: Loss={total_loss:.4f}, Presence={presence_loss:.4f}, Location={location_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        val_presence_losses = []
        val_location_losses = []
        
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        
        for batch in val_pbar:
            loss_dict, predictions = validate_step(model, batch, criterion, device)
            
            val_losses.append(loss_dict['total_loss'].item())
            val_presence_losses.append(loss_dict['presence_loss'].item())
            val_location_losses.append(loss_dict['location_loss'].item())
            
            val_pbar.set_postfix({
                'Val Loss': f'{loss_dict["total_loss"].item():.4f}'
            })
        
        # Epoch summary
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_presence = np.mean(train_presence_losses)
        avg_train_location = np.mean(train_location_losses)
        avg_val_presence = np.mean(val_presence_losses)
        avg_val_location = np.mean(val_location_losses)
        
        print(f"\nEPOCH {epoch+1} SUMMARY:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train - Presence: {avg_train_presence:.4f}, Location: {avg_train_location:.4f}")
        print(f"Val   - Presence: {avg_val_presence:.4f}, Location: {avg_val_location:.4f}")
        
        # Save history
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['train_presence_losses'].append(avg_train_presence)
        history['train_location_losses'].append(avg_train_location)
        history['val_presence_losses'].append(avg_val_presence)
        history['val_location_losses'].append(avg_val_location)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'history': history
            }, 'best_aneurysm_model.pth')
            print(f"💾 Saved best model (Val Loss: {avg_val_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"💾 Saved checkpoint for epoch {epoch+1}")
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return history

def main():
    """Main training function"""
    # Get device
    device = get_device()
    
    # Create datasets
    print("📁 Loading datasets...")
    
    resize_transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    
    train_dataset = BrainAneurysmDataset(
        data_dir='./data',
        split='train',
        classification_csv='./train.csv',
        labels_csv='./train_labels.csv',
        normalization='local_minmax',  # Use the normalization we discussed
        return_coordinates=False,
        transform=resize_transform
    )
    
    val_dataset = BrainAneurysmDataset(
        data_dir='./data',
        split='validation',
        classification_csv='./validation.csv',
        labels_csv='./validation_labels.csv',
        normalization='local_minmax',
        return_coordinates=False,
        transform=resize_transform
    )
    
    # Create data loaders
    batch_size = 16 if device.type == 'mps' else 32  # Smaller batch for MPS
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0 if device.type == 'mps' else 4  # MPS doesn't work well with multiprocessing
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0 if device.type == 'mps' else 4
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    
    # Create model
    print("🧠 Creating model...")
    model = BrainAneurysmCNN(num_classes=13, dropout_rate=0.3)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Get class weights for imbalanced dataset
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"Using class weights for imbalanced data: {class_weights}")
    
    # Create loss function
    criterion = AneurysmLoss(
        location_weights=class_weights,
        presence_weight=1.0,
        location_weight=2.0  # Weight location loss higher since it's more specific
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001, 
        weight_decay=0.01
    )
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=20,
        log_every=50,  # Log every 50 steps
        save_every=5
    )
    
    return model, history

if __name__ == "__main__":
    model, history = main()