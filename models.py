import torch.nn as nn
from torch.optim import Optimizer
from torchvision import models

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for better gradient flow
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class DeepBrainAneurysmCNN(nn.Module):
    """
    Deeper CNN model for brain aneurysm detection with residual connections.
    
    Architecture features:
    - 8 convolutional stages (1.5x deeper than original)
    - Residual connections for better gradient flow
    - Batch normalization and dropout for regularization
    - Progressive channel expansion: 1->32->64->128->256->512->768->1024->1024
    """
    
    def __init__(self, num_classes=13, dropout_rate=0.3):
        super(DeepBrainAneurysmCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1: 32 -> 64 channels
        self.stage1 = nn.Sequential(
            self._make_residual_block(32, 64, stride=1),
            self._make_residual_block(64, 64, stride=1)
        )
        
        # Stage 2: 64 -> 128 channels
        self.stage2 = nn.Sequential(
            self._make_residual_block(64, 128, stride=2),
            self._make_residual_block(128, 128, stride=1)
        )
        
        # Stage 3: 128 -> 256 channels
        self.stage3 = nn.Sequential(
            self._make_residual_block(128, 256, stride=2),
            self._make_residual_block(256, 256, stride=1)
        )
        
        # Stage 4: 256 -> 512 channels
        self.stage4 = nn.Sequential(
            self._make_residual_block(256, 512, stride=2),
            self._make_residual_block(512, 512, stride=1)
        )
        
        # Stage 5: 512 -> 768 channels (new deeper layer)
        self.stage5 = nn.Sequential(
            self._make_residual_block(512, 768, stride=2),
            self._make_residual_block(768, 768, stride=1)
        )
        
        # Stage 6: 768 -> 1024 channels (new deeper layer)
        self.stage6 = nn.Sequential(
            self._make_residual_block(768, 1024, stride=2),
            self._make_residual_block(1024, 1024, stride=1)
        )
        
        # Stage 7: Additional depth with same channels (new deeper layer)
        self.stage7 = nn.Sequential(
            self._make_residual_block(1024, 1024, stride=1),
            self._make_residual_block(1024, 1024, stride=1)
        )
        
        # Global Average Pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head with more capacity
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # Multi-output heads
        self.aneurysm_present_head = nn.Linear(512, 1)
        self.location_heads = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """Create a residual block with optional downsampling"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        return ResidualBlock(in_channels, out_channels, stride, downsample)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU networks"""
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
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Progressive feature extraction through residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        
        # Global pooling and flatten
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Shared features
        features = self.classifier(x)
        
        # Multi-output predictions
        aneurysm_present = self.aneurysm_present_head(features)
        location_logits = self.location_heads(features)
        
        return {
            'aneurysm_present': aneurysm_present,
            'locations': location_logits
        }

class DeepBrainAneurysmCoordCNN(DeepBrainAneurysmCNN):
    """
    Deeper CNN with coordinate prediction capability
    """
    def __init__(self, num_classes=13, dropout_rate=0.3):
        super().__init__(num_classes, dropout_rate)
        
        # Replace the classifier with shared features
        self.shared_features = self.classifier
        del self.classifier
        
        # Enhanced output heads
        self.aneurysm_present_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
        
        self.location_classification_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Enhanced coordinate prediction branch
        self.coordinate_features = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        self.coordinate_regression_head = nn.Sequential(
            nn.Linear(512, num_classes * 2),  # 2 coordinates (x, y) per class
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        # Feature extraction through all stages
        x = self.initial_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        
        # Global pooling and flatten
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Shared feature extraction
        shared_features = self.shared_features(x)
        
        # Multi-task predictions
        aneurysm_present = self.aneurysm_present_head(shared_features)
        location_logits = self.location_classification_head(shared_features)
        
        # Coordinate prediction
        coord_features = self.coordinate_features(shared_features)
        coordinate_preds = self.coordinate_regression_head(coord_features)
        
        batch_size = coordinate_preds.shape[0]
        coordinate_preds = coordinate_preds.view(batch_size, self.num_classes, 2)
        
        return {
            'aneurysm_present': aneurysm_present,
            'locations': location_logits,
            'coordinates': coordinate_preds
        }


class BrainAneurysmEfficientNet(nn.Module):
    """
    CNN model for brain aneurysm detection and location classification.
    
    Based on EfficientNet
    """
    def __init__(self, num_classes=13, pretrained=True, retrain=True, version='b0', with_coordinates=True, dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes
        efficientnet_models = {
            'b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            'b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            'b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            'b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            'b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            'b5': (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
            'b6': (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
            'b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
            # You can also add EfficientNet V2 variants:
            'v2_s': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT),
            'v2_m': (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.DEFAULT),
            'v2_l': (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights.DEFAULT),
        }
        
        if version not in efficientnet_models:
            raise ValueError(f"Unsupported EfficientNet version: {version}. "
                           f"Supported versions: {list(efficientnet_models.keys())}")
        
        model_constructor, default_weights = efficientnet_models[version]
        
        # Load the base model
        base_model = model_constructor(weights=default_weights if pretrained else None)
        
        original_first_conv = base_model.features[0][0]
        original_weights = original_first_conv.weight.data
        
        out_channels = original_first_conv.out_channels
        
        new_first_conv = nn.Conv2d(
            in_channels=1, # this gets changed
            out_channels=out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias is not None
        )
        
        # Average the weights across input channels: [out_channels, 3, 3, 3] -> [out_channels, 1, 3, 3]
        new_first_conv.weight.data = original_weights.mean(dim=1, keepdim=True)
        base_model.features[0][0] = new_first_conv        
        
        self.backbone = base_model.features
        in_features = base_model.classifier[1].in_features
        
        if pretrained and not retrain:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.aneurysm_present_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1)
        ) 

        self.location_heads = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        self.with_coordinates = with_coordinates
        if with_coordinates:
            self.coordinate_regression_head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes * 2),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.mean([2,3])

        aneurysm_present = self.aneurysm_present_head(features)
        location_logits = self.location_heads(features)

        out = {
            'aneurysm_present': aneurysm_present,
            'locations': location_logits,
        }
        
        if self.with_coordinates:
            coordinate_preds = self.coordinate_regression_head(features)
            batch_size = coordinate_preds.shape[0]
            coordinate_preds = coordinate_preds.view(batch_size, self.num_classes, 2)
            out['coordinates'] = coordinate_preds
        return out
        
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
        self.shared_features = self.classifier
        del self.classifier
        
        self.aneurysm_present_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1)
        )
        
        self.location_classification_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        self.coordinate_features = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        self.coordinate_regression_head = nn.Sequential(
            nn.Linear(512, num_classes * 2),  # 2 coordinates (x, y) per class
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        shared_features = self.shared_features(x)
        
        aneurysm_present = self.aneurysm_present_head(shared_features)
        location_logits = self.location_classification_head(shared_features)
        
        coord_features = self.coordinate_features(shared_features)
        coordinate_preds = self.coordinate_regression_head(coord_features)
        
        batch_size = coordinate_preds.shape[0]
        coordinate_preds = coordinate_preds.view(batch_size, self.num_classes, 2)
        return {
            'aneurysm_present': aneurysm_present,
            'locations': location_logits,
            'coordinates': coordinate_preds
        }
        
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
            if 'coordinates' not in predictions or 'coordinate_targets' not in targets:
                raise ValueError("Predictions and targets must include 'coordinates' and 'coordinate_targets' when with_coordinates is True.")
            coord_targets = targets['coordinate_targets']
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
            results['coordinate_loss'] = coordinate_loss
        return results

class NoOpLRScheduler:
    def __init__(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

    def step(self):
        # Do nothing
        pass