import ast
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import json
from typing import Optional, Tuple, Dict, Any
from torchvision import transforms

class BrainAneurysmDataset(Dataset):
    """
    PyTorch Dataset for brain aneurysm detection and classification.
    
    This dataset handles DICOM pixel arrays stored as .npy files and their corresponding
    aneurysm classification labels and coordinate annotations.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str, # train validation
        classification_csv: str, # Path to the classification CSV file
        labels_csv: str, # Path to the labels CSV file (with coordinates)
        transform: Optional[callable] = None,
        normalization: str = 'none',  # 'none', 'minmax', 'zscore', 'hounsfield', 'local_minmax'
        intensity_range: Optional[Tuple[float, float]] = None,
        num_channels: Optional[int] = 1,
        return_coordinates: bool = False,
        target_size: Optional[Tuple[int, int]] = (512, 512)
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the data directory containing train/validation/test folders
            split: Either 'train', 'validation'
            classification_csv: Path to the classification CSV file
            labels_csv: Path to the labels CSV file (with coordinates)
            transform: Optional transform to be applied to the images
            normalization: Type of normalization to apply
                - 'none': No normalization
                - 'minmax': Scale to [0, 1] using global or specified range
                - 'local_minmax': Scale to [0, 1] using local (per-image) min/max
                - 'zscore': Z-score normalization (mean=0, std=1)
                - 'hounsfield': Clip to typical brain HU range and normalize
                - 'percentile': Use 1st and 99th percentiles for robust normalization
            intensity_range: If provided, use this range for minmax normalization
            num_channels: Number of channels in the input images (0 or 1)
            return_coordinates: Whether to return coordinates for keypoints and their labels
            target_size: Desired size for image resizing (H, W) - manditory resize !!!
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.normalization = normalization
        self.intensity_range = intensity_range
        self.num_channels = num_channels
        self.return_coordinates = return_coordinates
        self.target_size = target_size

        # Define aneurysm location columns
        self.aneurysm_columns = [
            'Left Infraclinoid Internal Carotid Artery',
            'Right Infraclinoid Internal Carotid Artery',
            'Left Supraclinoid Internal Carotid Artery',
            'Right Supraclinoid Internal Carotid Artery',
            'Left Middle Cerebral Artery',
            'Right Middle Cerebral Artery',
            'Anterior Communicating Artery',
            'Left Anterior Cerebral Artery',
            'Right Anterior Cerebral Artery',
            'Left Posterior Communicating Artery',
            'Right Posterior Communicating Artery',
            'Basilar Tip',
            'Other Posterior Circulation'
        ]

        # Create mapping from aneurysm type names to indices
        self.coord_type_to_index = {col: i for i, col in enumerate(self.aneurysm_columns)} 
        
        # Load CSV files
        self.classification_df = pd.read_csv(classification_csv)
        self.labels_df = pd.read_csv(labels_csv) if labels_csv and os.path.exists(labels_csv) else None
        
        # Create a mapping from (SeriesInstanceUID, SOPInstanceUID) to index
        self.classification_df['unique_id'] = (
            self.classification_df['SeriesInstanceUID'].astype(str) + '_' + 
            self.classification_df['SOPInstanceUID'].astype(str)
        )
        
        if self.labels_df is not None:
            self.labels_df['unique_id'] = (
                self.labels_df['SeriesInstanceUID'].astype(str) + '_' + 
                self.labels_df['SOPInstanceUID'].astype(str)
            )
        
        if normalization in ['zscore', 'minmax'] and intensity_range is None:
            print("Computing global dataset statistics...")
            self._compute_global_stats()
        
        # Filter valid samples (check if corresponding .npy files exist)
        self.valid_samples = []
        self._validate_samples()
        
        print(f"Loaded {len(self.valid_samples)} valid samples for {split} split")

    def _compute_global_stats(self):
        """Compute global mean, std, min, max across all images."""
        all_values = []
        
        # Sample a subset for efficiency (or use all if dataset is small)
        sample_indices = np.random.choice(
            len(self.valid_samples), 
            min(100, len(self.valid_samples)), 
            replace=False
        )
        
        for idx in sample_indices:
            row_idx = self.valid_samples[idx]
            row = self.classification_df.iloc[row_idx]
            
            series_uid = str(row['SeriesInstanceUID'])
            sop_uid = str(row['SOPInstanceUID'])
            
            npy_path = os.path.join(
                self.data_dir, self.split, series_uid, f"{sop_uid}.npy"
            )
            
            image = np.load(npy_path).astype(np.float32)
            all_values.extend(image.flatten())
        
        all_values = np.array(all_values)
        
        self.global_mean = np.mean(all_values)
        self.global_std = np.std(all_values)
        self.global_min = np.min(all_values)
        self.global_max = np.max(all_values)
        self.percentile_1 = np.percentile(all_values, 1)
        self.percentile_99 = np.percentile(all_values, 99)
        
        print(f"Global stats - Mean: {self.global_mean:.2f}, Std: {self.global_std:.2f}")
        print(f"Range: [{self.global_min:.2f}, {self.global_max:.2f}]")
        print(f"1-99 percentile: [{self.percentile_1:.2f}, {self.percentile_99:.2f}]")
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply the specified normalization to the image."""
        if self.normalization == 'none':
            return image
        
        elif self.normalization == 'minmax' or self.normalization == 'local_minmax':
            if self.intensity_range:
                min_val, max_val = self.intensity_range
            elif self.normalization == 'minmax':
                min_val, max_val = self.global_min, self.global_max
            else:
                min_val, max_val = np.min(image), np.max(image)

            image = np.clip(image, min_val, max_val)
            image = (image - min_val) / (max_val - min_val + 1e-8)
            
        elif self.normalization == 'zscore': # Aka standard score normalization
            image = (image - self.global_mean) / (self.global_std + 1e-8)
            
        elif self.normalization == 'hounsfield': # Not for the provided data - but why not
            # Typical brain tissue HU range: -100 to +100
            image = np.clip(image, -100, 100)
            image = (image + 100) / 200  # Scale to [0, 1]
            
        elif self.normalization == 'percentile':
            # Robust normalization using percentiles
            image = np.clip(image, self.percentile_1, self.percentile_99)
            image = (image - self.percentile_1) / (self.percentile_99 - self.percentile_1 + 1e-8)

        return image
    
    def _validate_samples(self):
        """Validate that .npy files exist for all samples in the CSV."""
        for idx, row in self.classification_df.iterrows():
            series_uid = str(row['SeriesInstanceUID'])
            sop_uid = str(row['SOPInstanceUID'])
            
            # Construct file path
            npy_path = os.path.join(
                self.data_dir, 
                self.split, 
                series_uid, 
                f"{sop_uid}.npy"
            )
            
            if os.path.exists(npy_path):
                self.valid_samples.append(idx)
            else:
                print(f"Warning: Missing file {npy_path}")
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
            - 'image': torch.Tensor of shape (H, W) or (C, H, W) if transformed
            - 'aneurysm_present': Binary label for aneurysm presence
            - 'aneurysm_locations': Multi-label tensor for aneurysm locations
        """
        # Get the actual row index if some images didn't load
        row_idx = self.valid_samples[idx]
        row = self.classification_df.iloc[row_idx]
        
        # Load image data
        series_uid = str(row['SeriesInstanceUID'])
        sop_uid = str(row['SOPInstanceUID'])
        
        npy_path = os.path.join(
            self.data_dir, 
            self.split, 
            series_uid, 
            f"{sop_uid}.npy"
        )
        
        # Load the numpy array
        image = np.load(npy_path)
        
        # Convert to float32 and normalize if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # normalize depending on self.normalization
        image = self._normalize_image(image)

        # Coordinates keypoint detection
        coordinates = torch.zeros((len(self.aneurysm_columns), 2), dtype=torch.float32)

        if self.labels_df is not None and self.return_coordinates:
            unique_id = f"{series_uid}_{sop_uid}"
            label_rows = self.labels_df[self.labels_df['unique_id'] == unique_id]
            
            # compute the scale ratio
            height_scale = self.target_size[0] / image.shape[0]
            width_scale = self.target_size[1] / image.shape[1]

            for l_row in label_rows.itertuples():
                # Use a new variable 'coord_data' to avoid collision
                coord_data_str = l_row.Coordinates
                class_name = l_row._4 # Assuming the 4th column is the class name string
    
                if isinstance(coord_data_str, str):
                    # Safely evaluate the coordinate string
                    coord_dict = ast.literal_eval(coord_data_str)
        
                    # Scale coordinates
                    x_scaled = coord_dict['x'] * width_scale
                    y_scaled = coord_dict['y'] * height_scale
        
                    # Get the index for the specific aneurysm location
                    target_idx = self.coord_type_to_index[class_name]

                    # Populate the ORIGINAL 'coordinates' tensor correctly
                    coordinates[target_idx, 0] = x_scaled / self.target_size[1] # Normalize x
                    coordinates[target_idx, 1] = y_scaled / self.target_size[0] # Normalize y

        # Add channel dimension (converting (H, W) to (1, H, W))
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        else:
            resize_transform = transforms.Resize(self.target_size)
            image = resize_transform(image)
        
        # Prepare labels
        aneurysm_present = torch.tensor(row['Aneurysm Present'], dtype=torch.float32)
        
        # Multi-label classification for aneurysm locations
        aneurysm_locations = torch.tensor([
            row[col] for col in self.aneurysm_columns
        ], dtype=torch.float32)

        result = {
            'image': image,
            'aneurysm_present': aneurysm_present,
            'aneurysm_locations': aneurysm_locations,
        }

        if self.return_coordinates:
            result['coordinate_targets'] = coordinates
        
        return result
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        
        Returns:
            Tensor of shape (13,) with weights for each aneurysm location
        """
        # Calculate positive/negative ratios for each aneurysm location
        weights = []
        for col in self.aneurysm_columns:
            pos_count = self.classification_df[col].sum()
            neg_count = len(self.classification_df) - pos_count
            
            if pos_count > 0:
                weight = neg_count / pos_count
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_aneurysm_statistics(self) -> Dict[str, Any]:
        """Get statistics about aneurysm distribution in the dataset."""
        stats = {}
        
        # Overall aneurysm presence
        aneurysm_present_count = self.classification_df['Aneurysm Present'].sum()
        stats['total_samples'] = len(self.classification_df)
        stats['aneurysm_present'] = int(aneurysm_present_count)
        stats['aneurysm_absent'] = len(self.classification_df) - int(aneurysm_present_count)
        
        # Individual location statistics
        location_stats = {}
        for col in self.aneurysm_columns:
            location_stats[col] = int(self.classification_df[col].sum())
        
        stats['location_distribution'] = location_stats
        
        return stats


# Helper functions
def create_datasets(
    data_dir: str,
    train_csv: str,
    val_csv: str,
    train_labels_csv: str,
    val_labels_csv: str,
    transform_train: Optional[callable] = None,
    transform_val: Optional[callable] = None
) -> Tuple[BrainAneurysmDataset, BrainAneurysmDataset]:
    """
    Create train and validation datasets.
    
    Args:
        data_dir: Path to data directory
        train_csv: Path to training classification CSV
        val_csv: Path to validation classification CSV
        train_labels_csv: Path to training labels CSV
        val_labels_csv: Path to validation labels CSV
        transform_train: Transform for training data
        transform_val: Transform for validation data
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_dataset = BrainAneurysmDataset(
        data_dir=data_dir,
        split='train',
        classification_csv=train_csv,
        labels_csv=train_labels_csv,
        transform=transform_train,
    )
    
    val_dataset = BrainAneurysmDataset(
        data_dir=data_dir,
        split='validation',
        classification_csv=val_csv,
        labels_csv=val_labels_csv,
        transform=transform_val,
    )
    
    return train_dataset, val_dataset

def analyze_intensity_distribution(dataset, num_samples=50):
    """Analyze intensity distribution across samples."""
    all_stats = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image'].squeeze().numpy() if isinstance(sample['image'], torch.Tensor) else sample['image']
        
        stats = {
            'min': image.min(),
            'max': image.max(),
            'mean': image.mean(),
            'std': image.std()
        }
        all_stats.append(stats)
    
    # Print summary
    mins = [s['min'] for s in all_stats]
    maxs = [s['max'] for s in all_stats]
    means = [s['mean'] for s in all_stats]
    stds = [s['std'] for s in all_stats]
    
    print(f"Min values range: {min(mins):.2f} to {max(mins):.2f}")
    print(f"Max values range: {min(maxs):.2f} to {max(maxs):.2f}")
    print(f"Mean values range: {min(means):.2f} to {max(means):.2f}")
    print(f"Std values range: {min(stds):.2f} to {max(stds):.2f}")
    
    return all_stats

def analyze_image_dimensions(dataset, num_samples=100):
    """
    Analyze the dimensions of images in the dataset.
    
    Args:
        dataset: BrainAneurysmDataset instance
        num_samples: Number of samples to analyze (None for all)
    
    Returns:
        Dictionary with dimension statistics
    """
    import numpy as np
    from collections import Counter, defaultdict
    
    print("🔍 Analyzing image dimensions...")
    
    dimensions = []
    shapes_counter = Counter()
    aspect_ratios = []
    areas = []
    
    # Determine how many samples to analyze
    total_samples = len(dataset)
    if num_samples is None:
        num_samples = total_samples
    else:
        num_samples = min(num_samples, total_samples)
    
    # Sample indices
    if num_samples < total_samples:
        indices = np.random.choice(total_samples, num_samples, replace=False)
    else:
        indices = range(total_samples)
    
    print(f"Analyzing {num_samples} samples...")
    
    for i, idx in enumerate(indices):
        try:
            # Get raw numpy array (before any transforms)
            row_idx = dataset.valid_samples[idx]
            row = dataset.classification_df.iloc[row_idx]
            
            series_uid = str(row['SeriesInstanceUID'])
            sop_uid = str(row['SOPInstanceUID'])
            
            npy_path = os.path.join(
                dataset.data_dir, 
                dataset.split, 
                series_uid, 
                f"{sop_uid}.npy"
            )
            
            # Load raw image
            image = np.load(npy_path)
            shape = image.shape
            
            # Store information
            dimensions.append(shape)
            shapes_counter[shape] += 1
            
            # Calculate derived metrics
            if len(shape) == 2:  # H, W
                h, w = shape
                aspect_ratios.append(w / h)
                areas.append(h * w)
            elif len(shape) == 3:  # Could be C, H, W or H, W, C
                if shape[0] < 10:  # Assume C, H, W
                    c, h, w = shape
                else:  # Assume H, W, C
                    h, w, c = shape
                aspect_ratios.append(w / h)
                areas.append(h * w)
            
            if i % 20 == 0:
                print(f"Processed {i+1}/{num_samples} samples...")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Analyze results
    if not dimensions:
        print("❌ No valid dimensions found!")
        return None
    
    # Convert to numpy for easier analysis
    dimensions = np.array(dimensions, dtype=object)
    
    # Extract heights and widths
    heights = []
    widths = []
    channels = []
    
    for shape in dimensions:
        if len(shape) == 2:
            h, w = shape
            heights.append(h)
            widths.append(w)
            channels.append(1)
        elif len(shape) == 3:
            if shape[0] < 10:  # C, H, W
                c, h, w = shape
                channels.append(c)
            else:  # H, W, C
                h, w, c = shape
                channels.append(c)
            heights.append(h)
            widths.append(w)
    
    heights = np.array(heights)
    widths = np.array(widths)
    channels = np.array(channels)
    aspect_ratios = np.array(aspect_ratios)
    areas = np.array(areas)
    
    # Statistics
    stats = {
        'total_analyzed': len(dimensions),
        'unique_shapes': len(shapes_counter),
        'height_stats': {
            'min': int(heights.min()),
            'max': int(heights.max()),
            'mean': float(heights.mean()),
            'median': float(np.median(heights)),
            'std': float(heights.std())
        },
        'width_stats': {
            'min': int(widths.min()),
            'max': int(widths.max()),
            'mean': float(widths.mean()),
            'median': float(np.median(widths)),
            'std': float(widths.std())
        },
        'aspect_ratio_stats': {
            'min': float(aspect_ratios.min()),
            'max': float(aspect_ratios.max()),
            'mean': float(aspect_ratios.mean()),
            'median': float(np.median(aspect_ratios)),
            'std': float(aspect_ratios.std())
        },
        'area_stats': {
            'min': int(areas.min()),
            'max': int(areas.max()),
            'mean': float(areas.mean()),
            'median': float(np.median(areas)),
        },
        'channel_stats': {
            'min': int(channels.min()),
            'max': int(channels.max()),
            'most_common': int(np.bincount(channels).argmax())
        },
        'most_common_shapes': shapes_counter.most_common(10),
        'shape_distribution': dict(shapes_counter)
    }
    
    # Print detailed report
    print("\n" + "="*60)
    print("📊 IMAGE DIMENSION ANALYSIS REPORT")
    print("="*60)
    
    print(f"📋 OVERVIEW:")
    print(f"  • Total samples analyzed: {stats['total_analyzed']:,}")
    print(f"  • Unique shapes found: {stats['unique_shapes']}")
    
    print(f"\n📏 HEIGHT STATISTICS:")
    print(f"  • Range: {stats['height_stats']['min']} - {stats['height_stats']['max']} pixels")
    print(f"  • Mean: {stats['height_stats']['mean']:.1f} ± {stats['height_stats']['std']:.1f}")
    print(f"  • Median: {stats['height_stats']['median']:.1f}")
    
    print(f"\n📐 WIDTH STATISTICS:")
    print(f"  • Range: {stats['width_stats']['min']} - {stats['width_stats']['max']} pixels")
    print(f"  • Mean: {stats['width_stats']['mean']:.1f} ± {stats['width_stats']['std']:.1f}")
    print(f"  • Median: {stats['width_stats']['median']:.1f}")
    
    print(f"\n📊 ASPECT RATIO (W/H):")
    print(f"  • Range: {stats['aspect_ratio_stats']['min']:.3f} - {stats['aspect_ratio_stats']['max']:.3f}")
    print(f"  • Mean: {stats['aspect_ratio_stats']['mean']:.3f}")
    
    print(f"\n🖼️  MOST COMMON SHAPES:")
    for shape, count in stats['most_common_shapes']:
        percentage = (count / stats['total_analyzed']) * 100
        print(f"  • {shape}: {count} samples ({percentage:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if stats['unique_shapes'] == 1:
        print("  ✅ All images have the same dimensions - no resizing needed!")
    elif stats['unique_shapes'] <= 5:
        print("  ⚠️  Few different sizes - consider standardizing to most common size")
        most_common_shape = stats['most_common_shapes'][0][0]
        print(f"     Recommended target size: {most_common_shape}")
    else:
        print("  🚨 Many different sizes detected - resizing is HIGHLY RECOMMENDED")
        
        # Suggest reasonable target size
        median_h = int(stats['height_stats']['median'])
        median_w = int(stats['width_stats']['median'])
        
        # Round to nearest power of 2 or common sizes
        common_sizes = [256, 512, 768, 1024]
        target_h = min(common_sizes, key=lambda x: abs(x - median_h))
        target_w = min(common_sizes, key=lambda x: abs(x - median_w))
        
        print(f"     Suggested target size: ({target_h}, {target_w})")
        print(f"     Based on median: ({median_h}, {median_w})")
    
    # Check for potential issues
    height_var = stats['height_stats']['std'] / stats['height_stats']['mean']
    width_var = stats['width_stats']['std'] / stats['width_stats']['mean']
    
    if height_var > 0.2 or width_var > 0.2:
        print("\n🚨 HIGH VARIABILITY DETECTED:")
        print("   • CNN architecture will need resizing for consistent input")
        print("   • Coordinate normalization needs per-image dimensions")
        
    return stats

# Example of how to use the dataset
if __name__ == "__main__":
    # Example usage
    data_dir = "./data"
    
    train_dataset = BrainAneurysmDataset(
        data_dir=data_dir,
        split='train',
        classification_csv='train.csv',
        labels_csv='train_labels.csv'
    )
    
    # Print dataset statistics
    print("Dataset Statistics:")
    stats = train_dataset.get_aneurysm_statistics()
    print(f"Total samples: {stats['total_samples']}")
    print(f"Aneurysm present: {stats['aneurysm_present']}")
    print(f"Aneurysm absent: {stats['aneurysm_absent']}'\n")
    
    # Get a sample
    sample = train_dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Aneurysm present: {sample['aneurysm_present']}")
    print(f"Aneurysm locations: {sample['aneurysm_locations']}\n")
    print("Raw image values distribution:")
    stats = analyze_intensity_distribution(train_dataset)
    print(f"\nClass weights:")
    for col, weight in zip(train_dataset.aneurysm_columns, train_dataset.get_class_weights()):
        print(f"{col}: {weight}")

    print()
    analyze_image_dimensions(train_dataset)
        
