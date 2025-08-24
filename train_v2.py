import wandb
import torch
from datasets import BrainAneurysmDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from models import BrainAneurysmCNN, BrainAneurysmCoordCNN, AneurysmLoss, NoOpLRScheduler, BrainAneurysmEfficientNet, DeepBrainAneurysmCNN, DeepBrainAneurysmCoordCNN
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from metrics import setup_metrics, extract_metrics_results
aneurysm_columns = [
    'Left_Infraclinoid_Internal_Carotid_Artery',
    'Right_Infraclinoid_Internal_Carotid_Artery',
    'Left_Supraclinoid_Internal_Carotid_Artery',
    'Right_Supraclinoid_Internal_Carotid_Artery',
    'Left_Middle_Cerebral_Artery',
    'Right_Middle_Cerebral_Artery',
    'Anterior_Communicating_Artery',
    'Left_Anterior_Cerebral_Artery',
    'Right_Anterior_Cerebral_Artery',
    'Left_Posterior_Communicating_Artery',
    'Right_Posterior_Communicating_Artery',
    'Basilar_Tip',
    'Other_Posterior_Circulation'
]


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

def train_step(model, batch, criterion, optimizer, device, with_coordinates=True):
    """Single training step"""
    model.train()
    
    # Move to device
    images = batch['image'].to(device)
    aneurysm_present = batch['aneurysm_present'].to(device)
    aneurysm_locations = batch['aneurysm_locations'].to(device)
    
    # Forward pass
    optimizer.zero_grad()
    predictions = model(images)
    
    targets = {
        'aneurysm_present': aneurysm_present,
        'aneurysm_locations': aneurysm_locations,
    }
    if with_coordinates:
        targets['coordinate_targets'] = batch['coordinate_targets'].to(device)
    
    loss_dict = criterion(predictions, targets)
    loss_dict['total_loss'].backward()
    optimizer.step()
    
    return loss_dict, predictions, targets

def validate_step(model, batch, criterion, device, with_coordinates=True):
    """Single validation step"""
    model.eval()
    
    with torch.no_grad():
        images = batch['image'].to(device)
        aneurysm_present = batch['aneurysm_present'].to(device)
        aneurysm_locations = batch['aneurysm_locations'].to(device)
        
        predictions = model(images)
        
        targets = {
            'aneurysm_present': aneurysm_present,
            'aneurysm_locations': aneurysm_locations,
        }

        if with_coordinates:
            targets['coordinate_targets'] = batch['coordinate_targets'].to(device) 
        
        loss_dict = criterion(predictions, targets)
    
    return loss_dict, predictions, targets

def evaluate_model(model, dataloader, criterion, device, metrics_tuple, with_coordinates=True):
    """Comprehensive model evaluation"""
    model.eval()
    
    total_loss = 0
    total_presence_loss = 0
    total_location_loss = 0
    total_coordinate_loss = 0
    binary_metrics, multilabel_metrics, per_class_metrics = metrics_tuple

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            loss_dict, predictions, targets = validate_step(model, batch, criterion, device, with_coordinates=with_coordinates)
            
            total_loss += loss_dict['total_loss'].item()
            total_presence_loss += loss_dict['presence_loss'].item()
            total_location_loss += loss_dict['location_loss'].item()

            # Convert logits to probabilities
            presence_probs = torch.sigmoid(predictions['aneurysm_present'].squeeze())
            location_probs = torch.sigmoid(predictions['locations'])
            
            # Update binary metrics
            binary_metrics.update(presence_probs, targets['aneurysm_present'].int())
            
            # Update multilabel metrics
            multilabel_metrics.update(location_probs, targets['aneurysm_locations'].int())
            per_class_metrics.update(location_probs, targets['aneurysm_locations'].int())

            if with_coordinates:
                total_coordinate_loss += loss_dict['coordinate_loss'].item()
            
    all_metrics = {
        'loss': total_loss / len(dataloader),
        'presence_loss': total_presence_loss / len(dataloader),
        'location_loss': total_location_loss / len(dataloader),
    }
    if with_coordinates:
        all_metrics['coordinate_loss'] = total_coordinate_loss / len(dataloader)
    
    # Compute final metrics
    binary_results = binary_metrics.compute()
    multilabel_results = multilabel_metrics.compute()
    per_class_results = per_class_metrics.compute()

    # Reset metrics for next evaluation
    binary_metrics.reset()
    multilabel_metrics.reset()
    per_class_metrics.reset() 
    
    return all_metrics, (binary_results, multilabel_results, per_class_results)

def train_brain_aneurysm_model(config):
    """Main training function with wandb logging"""
    
    # Initialize wandb
    wandb.init(
        project="brain-aneurysm-detection",
        config=config,
        name=f"{config['model_name']}_{config['experiment_name']}"
    )

    device = get_device()
    
    # Create datasets
    print("📁 Loading datasets...")
    train_dataset = BrainAneurysmDataset(
        data_dir=config['data_dir'],
        split='train',
        classification_csv=config['train_csv'],
        labels_csv=config['train_labels_csv'],
        normalization=config['normalization'],
        target_size=config['target_size'],
        return_coordinates=config['with_coordinates'],
    )
    
    val_dataset = BrainAneurysmDataset(
        data_dir=config['data_dir'],
        split='validation',
        classification_csv=config['val_csv'],
        labels_csv=config['val_labels_csv'],
        normalization=config['normalization'],
        target_size=config['target_size'],
        return_coordinates=config['with_coordinates'],
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    
    print(f"Train samples 💪: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    # When implementing a new model add your configuration here
    # This will enable the model throw and error if model is not built for coordinate predictions
    model = None
    if config['model_name'] == 'CustomBrainAneurysmCNN':
        if config['with_coordinates']:
            ModelClass = BrainAneurysmCoordCNN
        else:
            ModelClass = BrainAneurysmCNN
        model = ModelClass(
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
    if config['model_name'] == 'BrainAneurysmEfficientNet':
        model = BrainAneurysmEfficientNet(
            num_classes=config['num_classes'],
            pretrained=config['custom_parameters']['efficientnet']['pretrained'],
            version=config['custom_parameters']['efficientnet']['version'],
            retrain=config['custom_parameters']['efficientnet']['retrain'],
            with_coordinates=config['with_coordinates'],
            dropout_rate=config['dropout_rate']
        )
    if config['model_name'] == 'DeepBrainAneurysmCNN':
        if config['with_coordinates']:
            ModelClass = DeepBrainAneurysmCoordCNN
        else:
            ModelClass = DeepBrainAneurysmCNN
        model = ModelClass(
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
    if not model:
        raise Exception("No model found! Add model to training loop")
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model created with {total_params:,} parameters")
    
    # Log model to wandb
    wandb.watch(model, log_freq=100)

    # Create loss function
    criterion = None
    if config['balance_classes']:
        class_weights = train_dataset.get_class_weights().to(device)
        criterion = AneurysmLoss(
            location_weights=class_weights,
            presence_weight=config['presence_weight'],
            location_weight=config['location_weight'],
            coordinate_weight=config['coordinate_weight'],
            with_coordinates=config['with_coordinates'],
        )
    else:
        criterion = AneurysmLoss(
            presence_weight=config['presence_weight'],
            location_weight=config['location_weight'],
            coordinate_weight=config['coordinate_weight'],
            with_coordinates=config['with_coordinates'],
        )
    if not criterion:
        raise Exception("No loss function found! Add loss function to training loop")

    # Create optimizer
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    
    # Learning rate scheduler
    if config['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    else:
        # Custom scheduler to do nothing
        scheduler = NoOpLRScheduler(optimizer)
    
    # Training loop
    best_val_loss = float('inf')

    # Get matrics collections
    custom_metrics = setup_metrics(device)
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        train_losses = []
        train_presence_losses = []
        train_location_losses = []
        train_coordinate_losses = []
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(train_pbar):
            loss_dict, predictions, targets = train_step(model, batch, criterion, optimizer, device, with_coordinates=config['with_coordinates'])
            
            train_losses.append(loss_dict['total_loss'].item())
            train_presence_losses.append(loss_dict['presence_loss'].item())
            train_location_losses.append(loss_dict['location_loss'].item())
            if config['with_coordinates']:
                train_coordinate_losses.append(loss_dict['coordinate_loss'].item())
            
            postfix = {
                'Loss': f'{loss_dict["total_loss"].item():.4f}',
                'Pres': f'{loss_dict["presence_loss"].item():.4f}',
                'Loc': f'{loss_dict["location_loss"].item():.4f}',
            } 
            if config['with_coordinates']:
                postfix['Coord'] = f'{loss_dict["coordinate_loss"].item():.4f}' 
            train_pbar.set_postfix(postfix)
            
            # Log training step
            if batch_idx % config['log_every'] == 0:
                wandb_log = {
                    "train/step_loss": loss_dict['total_loss'].item(),
                    "train/step_presence_loss": loss_dict['presence_loss'].item(),
                    "train/step_location_loss": loss_dict['location_loss'].item(),
                    "epoch": epoch,
                    "step": epoch * len(train_loader) + batch_idx
                }

                if config['with_coordinates']:
                    wandb_log ["train/step_coordinate_loss"] = loss_dict['coordinate_loss'].item(),
                wandb.log(wandb_log)
        
        # Calculate training averages
        avg_train_loss = np.mean(train_losses)
        avg_train_presence = np.mean(train_presence_losses)
        avg_train_location = np.mean(train_location_losses)

        if config['with_coordinates']:
            avg_train_coordinate = np.mean(train_coordinate_losses)
        
        print("Running validation...")
        validation_loss_metrics, other_metrics_packed = evaluate_model(model, val_loader, criterion, device, custom_metrics, config['with_coordinates'])
        other_metrics = extract_metrics_results(*other_metrics_packed)
        # Learning rate scheduling
        scheduler.step(validation_loss_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEPOCH {epoch+1} SUMMARY:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {validation_loss_metrics['loss']:.4f}")
        print(f"Binary Accuracy (Aneurysm detection): {other_metrics['binary_accuracy']:.4f}")
        print(f"Binary Recall (Aneurysm detection): {other_metrics['binary_recall']:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Log to wandb
        wandb_data = {
            # Training metrics
            "train/epoch_loss": avg_train_loss,
            "train/epoch_presence_loss": avg_train_presence,
            "train/epoch_location_loss": avg_train_location,
            
            # Validation metrics
            "val/loss": validation_loss_metrics['loss'],
            "val/presence_loss": validation_loss_metrics['presence_loss'],
            "val/location_loss": validation_loss_metrics['location_loss'],
            
            # Training info
            "epoch": epoch,
            "learning_rate": current_lr
        }

        if config['with_coordinates']:
            wandb_data["train/epoch_coordinate_loss"] = avg_train_coordinate
            wandb_data["val/coordinate_loss"] = validation_loss_metrics['coordinate_loss']

        for key, value in other_metrics.items():
            if not key.startswith('per_class'):
                wandb_data[f"val/{key}"] = value
            elif key.endswith('accuracy'):
                for i, class_value in enumerate(value):
                    wandb_data[f"val/class_{aneurysm_columns[i]}_accuracy"] = class_value
            elif key.endswith('recall'):
                for i, class_value in enumerate(value):
                    wandb_data[f"val/class_{aneurysm_columns[i]}_recall"] = class_value
            elif key.endswith('precision'):
                for i, class_value in enumerate(value):
                    wandb_data[f"val/class_{aneurysm_columns[i]}_precision"] = class_value
            elif key.endswith('f1'):
                for i, class_value in enumerate(value):
                    wandb_data[f"val/class_{aneurysm_columns[i]}_f1"] = class_value
        wandb.log(wandb_data)
        
        # Save best model
        if validation_loss_metrics['loss'] < best_val_loss:
            best_val_loss = validation_loss_metrics['loss']
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': validation_loss_metrics['loss'],
                'config': config
            }, f'best_model_{config["instance"]}.pth')

            # Save to wandb
            wandb.save(f'best_model_{config["instance"]}.pth')
            print(f"💾 Saved best model (Val Loss: {validation_loss_metrics['loss']:.4f})")
            
            # Update wandb summary
            wandb.run.summary["best_val_loss"] = validation_loss_metrics['loss']
            wandb.run.summary["best_val_accuracy"] = other_metrics['binary_accuracy']
            wandb.run.summary["best_epoch"] = epoch
        
        # Save checkpoint
        # if (epoch + 1) % config['save_every'] == 0:
        #     checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'config': config
        #     }, checkpoint_path)
        #     wandb.save(checkpoint_path)
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final summary
    wandb.run.summary["total_epochs"] = config['num_epochs']
    wandb.run.summary["total_parameters"] = total_params
    
    wandb.finish()
    
    return model

def main():
    """Main function to run training"""
    
    # Configuration
    config = {
        'instance': '0',
        # Data paths
        'data_dir': './data',
        'train_csv': './train.csv',
        'val_csv': './validation.csv',
        'train_labels_csv': './train_labels.csv',
        'val_labels_csv': './validation_labels.csv',

        # Coordinates mode:
        'with_coordinates': True,
        
        # Model configuration
        'model_name': 'DeepBrainAneurysmCNN',  # 'CustomBrainAneurysmCNN' or 'BrainAneurysmEfficientNet' or 'DeepBrainAneurysmCNN'
        'experiment_name': 'nocap',
        # 'pretrained': True,
        'num_classes': 13,
        'dropout_rate': 0.5,
        'custom_parameters': {
            'efficientnet': {
                'version': 'b2',
                'pretrained': True,
                'retrain': True,
            }
        },

        # Data configuration
        'target_size': (512, 512),
        'normalization': 'percentile',
        'batch_size': 16,
        'num_workers': 4,
        
        # Training configuration
        'num_epochs': 1,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'optimizer': 'adamw',  # 'adamw' or 'adam' or 'sgd'
        'ReduceLROnPlateau': True, # set this to false to remove or modify it in code

        # Loss weights
        'presence_weight': 1.0,
        'location_weight': 1.5,
        'coordinate_weight': 1,  # 3.0, # None if you don't use coordinates
        'balance_classes': True, # if or not if to use the custom per class weights

        # Logging
        'log_every': 25,
    }
    
    print("🚀 Starting Brain Aneurysm Detection Training")
    print(f"Configuration: {config}")
    
    # Train model
    model = train_brain_aneurysm_model(config)
    
    print("✅ Training completed successfully!")
    
    return model

if __name__ == "__main__":
    # Run training
    model = main()