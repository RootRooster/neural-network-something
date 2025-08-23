import torch.nn as nn
from torch.optim import Optimizer

# EPOCH 20 SUMMARY:
# Train Loss: 3.0255 | Val Loss: 2.9627
# Train - Presence: 0.5457, Location: 1.2399
# Val   - Presence: 0.5129, Location: 1.2249
# 💾 Saved checkpoint for epoch 20
# 
# 🎉 Training completed!
# Best validation loss: 2.9409
#
#    criterion = AneurysmLoss(
#        location_weights=class_weights,
#        presence_weight=1.0,
#        location_weight=2.0  # Weight location loss higher since it's more specific
#    )
#    
#    # Create optimizer
#    optimizer = optim.AdamW(
#        model.parameters(), 
#        lr=0.001, 
#        weight_decay=0.01
#    )
class BrainAneurysmCNN(nn.Module):
    """
    CNN model for brain aneurysm detection and location classification.
    
    Architecture designed for medical imaging with:
    - Residual connections for better gradient flow
    - Batch normalization for stable training
    - Dropout for regularization
    - Multi-output head for multi-label classification
    """
    
    def __init__(self, num_classes=13, dropout_rate=0.3):
        super(BrainAneurysmCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Ensure consistent output size
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # Multi-output heads
        self.aneurysm_present_head = nn.Linear(512, 1)  # Binary: aneurysm present/absent
        self.location_heads = nn.Linear(512, num_classes)  # Multi-label: specific locations
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared features
        features = self.classifier(x)
        
        # Multi-output predictions
        aneurysm_present = self.aneurysm_present_head(features)
        location_logits = self.location_heads(features)
        result = {
            'aneurysm_present': aneurysm_present,
            'locations': location_logits
        }
        return result

class BrainAneurysmCoordCNN(BrainAneurysmCNN):
    def __init__(self, num_classes=13, dropout_rate=0.3):
        super().__init__(num_classes, dropout_rate)
        ## TODO IMPLEMENT THIS

class AneurysmLoss(nn.Module):
    """
    Combined loss for aneurysm detection:
    - Binary cross entropy for overall aneurysm presence
    - Weighted binary cross entropy for location classification
    """
    
    def __init__(self, location_weights=None, presence_weight=1.0, location_weight=1.0, coordinate_weight=2.0, with_coordinates=False):
        super(AneurysmLoss, self).__init__()
        self.presence_weight = presence_weight
        self.location_weight = location_weight
        self.coordinate_weight = coordinate_weight
        self.with_coordinates = with_coordinates

        # Binary cross entropy for aneurysm presence
        self.bce_loss = nn.BCEWithLogitsLoss()

        if self.with_coordinates:
            self.coordinate_loss = nn.SmoothL1Loss(reduction='none')
        
        # Weighted binary cross entropy for locations
        if location_weights is not None:
            self.location_bce = nn.BCEWithLogitsLoss(pos_weight=location_weights)
        else:
            self.location_bce = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        # Aneurysm presence loss
        presence_loss = self.bce_loss(
            predictions['aneurysm_present'].squeeze(),
            targets['aneurysm_present']
        )
        
        # Location classification loss
        location_loss = self.location_bce(
            predictions['locations'],
            targets['aneurysm_locations']
        )
        
        # Combined loss
        total_loss = (self.presence_weight * presence_loss + 
                     self.location_weight * location_loss)
        
        results = {
            'total_loss': total_loss,
            'presence_loss': presence_loss,
            'location_loss': location_loss
        }

        if self.with_coordinates:
            if 'coordinates' not in predictions or 'coordinates_targets' not in targets:
                raise ValueError("Predictions and targets must include 'coordinates' and 'coordinates_targets' when with_coordinates is True.")
            coord_targets = targets['coordinates_targets']
            coord_preds = predictions['coordinates']
            coord_mask = targets['aneurysm_locations']

            coord_loss_per_point = self.coordinate_loss(coord_preds, coord_targets)
            # average the loss for the x and y coordinate
            coord_loss_per_class = coord_loss_per_point.mean(dim=2) # [batch size, num_classes]

            # This operation penalizes only the coordinates whose classes are detected
            masked_coord_loss = coord_loss_per_class * coord_mask

            # Calculate num of coordinate predictions 
            valid_coords = coord_mask.sum()

            # Normalize the data so it is independent on the number of predictions
            coordinate_loss = masked_coord_loss.sum() / (valid_coords + 1e-8)
            total_loss += self.coordinate_weight * coordinate_loss
            results['total_loss'] = total_loss 
            results['coordinates'] = coord_preds
        return results



class NoOpLRScheduler:
    def __init__(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

    def step(self):
        # Do nothing
        pass