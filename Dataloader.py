import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import re
from collections import defaultdict
from pathlib import Path
from load_feat_pd import load_feat  # Ensure this module is in your PYTHONPATH
import pandas as pd
from sklearn.model_selection import train_test_split


class FeatureDataset(Dataset):
    def __init__(self, base_folder_path: str, feature_names: List[str], log_value: bool = False):
        """
        Initializes the dataset by loading and merging multiple features.

        Args:
            base_folder_path (str): Base path where feature folders are located.
            feature_names (List[str]): List of feature names to load (e.g., ['energy', 'f0']).
            log_value (bool): Whether to apply logarithmic transformation to feature values.
        """
        base_folder_path_unnorm = base_folder_path.replace('_normalized', '')

        self.base_folder_path = base_folder_path
        self.feature_names = feature_names
        self.log_value = log_value

        # Load data for each feature
        self.feature_data = {}
        for feature in self.feature_names:
            self.base_folder_path = base_folder_path_unnorm if feature == 'energy' else base_folder_path
            print(f"Loading feature: {feature}")
            data = load_feat(self.base_folder_path, feature_name=feature, log_value=self.log_value)
            self.feature_data[feature] = data
            print(f"Loaded {len(data)} samples for feature '{feature}'\n")

        # Merge data across features
        self.merged_data = self._merge_features()

    def _merge_features(self) -> List[Dict]:
        """
        Merges data from different features based on the 'filename' key.

        Returns:
            List[Dict]: A list of merged sample dictionaries containing all features.
        """
        # Create a mapping from filename to data for each feature
        feature_maps = {}
        for feature_name, data in self.feature_data.items():
            feature_map = {}
            for sample in data:
                filename = sample.get('filename')
                if filename:
                    feature_map[filename] = sample
                else:
                    print(f"Warning: Sample without 'filename' in feature '{feature_name}'. Skipping.")
            feature_maps[feature_name] = feature_map
            print(f"Feature '{feature_name}' has {len(feature_map)} unique filenames.")

        # Find the intersection of filenames across all features
        all_filenames = set.intersection(*(set(fm.keys()) for fm in feature_maps.values()))
        print(f"\nTotal common filenames across all features: {len(all_filenames)}\n")

        # Merge samples that are present in all feature sets
        merged_samples = []
        for filename in all_filenames:
            merged_sample = {}
            # Retrieve metadata from the first feature (assuming metadata is consistent across features)
            first_feature = self.feature_names[0]
            sample_data = feature_maps[first_feature][filename]

            # Copy metadata fields
            for key, value in sample_data.items():
                if key != 'value' and key != 'filename':
                    merged_sample[key] = value
                elif key == 'filename':
                    merged_sample[key] = value  # Keep 'filename'

            # Add features with feature names as keys
            for feature_name in self.feature_names:
                feature_sample = feature_maps[feature_name][filename]
                feature_value = feature_sample.get('value')
                if feature_value is not None:
                    merged_sample[feature_name] = feature_value
                else:
                    print(f"Warning: 'value' missing for filename '{filename}' in feature '{feature_name}'.")
                    # Optionally handle missing feature values here

            merged_samples.append(merged_sample)

        print(f"Total merged samples with all features: {len(merged_samples)}")
        return merged_samples

    def __len__(self):
        return len(self.merged_data)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict: A dictionary containing all features and metadata for the sample.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        return self.merged_data[idx]


class SubsetFeatureDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, feature_names: List[str]):
        """
        Initializes the subset dataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the subset of data.
            feature_names (List[str]): List of feature names included in the dataset.
        """
        self.data = dataframe.reset_index(drop=True)
        self.feature_names = feature_names
        self.feats2level = {
        'jitter': 'utt',
        'shimmer': 'utt',
        'rp': 'utt',
        'f0': 'frame',
        'energy': 'frame'
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: A tensor containing all processed features for the sample.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        # Initialize a list to hold processed feature values
        processed_features = []
        
        # Iterate over each feature
        for feature in self.feature_names:
            level = self.feats2level.get(feature, 'utt')  # Default to 'utt' if not specified
            feature_data = self.data.iloc[idx][feature]
            
            if level == 'frame':
                # Ensure feature_data is a NumPy array for consistency
                if isinstance(feature_data, list):
                    feature_data = np.array(feature_data)
                
                # Calculate mean and standard deviation
                mean = np.mean(feature_data)
                std = np.std(feature_data)
                
                # Append mean and std to the processed_features list
                processed_features.extend([mean, std])
            else:
                # For 'utt' level features
                if isinstance(feature_data, np.ndarray):
                    # If the feature is an array, flatten it and extend the list
                    processed_features.extend(feature_data.flatten())
                else:
                    # If the feature is a scalar, append it directly
                    processed_features.append(feature_data)
        
        # Convert the processed features list to a PyTorch tensor of type float32
        concate_features = torch.tensor(processed_features, dtype=torch.float32)
        
        return concate_features
    

def train_test_split_by_subject(
    dataset: Dataset,
    feature_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[SubsetFeatureDataset, SubsetFeatureDataset]:
    """
    Splits the dataset into train and test sets based on subject_id, ensuring no overlapping subjects
    and maintaining gender balance in the test set.

    Args:
        dataset (Dataset): The merged dataset (FeatureDataset instance).
        feature_names (List[str]): List of feature names included in the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[SubsetFeatureDataset, SubsetFeatureDataset]: Train and test datasets.
    """
    # Convert merged data to DataFrame
    merged_data = dataset.merged_data  # Assuming 'merged_data' is a list of dicts
    df = pd.DataFrame(merged_data)
    
    # Extract unique subject_ids and their genders
    subjects_df = df[['subject_id', 'gender']].drop_duplicates()

    # Check for missing values in 'gender'
    subjects_df = subjects_df.dropna(subset=['gender'])

    # Ensure 'gender' is categorical
    subjects_df['gender'] = subjects_df['gender'].astype(str)

    # Perform stratified split based on gender
    train_subjects, test_subjects = train_test_split(
        subjects_df,
        test_size=test_size,
        stratify=subjects_df['gender'],
        random_state=random_state
    )

    # Extract lists of subject_ids
    train_subject_ids = set(train_subjects['subject_id'])
    test_subject_ids = set(test_subjects['subject_id'])

    print(f"Total subjects: {len(subjects_df)}")
    print(f"Training subjects: {len(train_subject_ids)}")
    print(f"Testing subjects: {len(test_subject_ids)}")
    
  
    

    # Assign samples to train and test sets based on subject_id
    train_df = df[df['subject_id'].isin(train_subject_ids)].reset_index(drop=True)
    test_df = df[df['subject_id'].isin(test_subject_ids)].reset_index(drop=True)

    print(f"Total training samples: {len(train_df)}")
    print(f"Total testing samples: {len(test_df)}")
    

    # Verify gender balance in test set
    test_gender_counts = test_df['gender'].value_counts(normalize=True)
    print("\nGender distribution in test set:")
    print(test_gender_counts)
    
    # Verify group balance in test set
    test_group_counts = test_df['group_id'].value_counts(normalize=True)
    print("\nGroup distribution in test set:")
    print(test_group_counts)
    

    # Create subset datasets
    train_dataset = SubsetFeatureDataset(train_df, feature_names)
    test_dataset = SubsetFeatureDataset(test_df, feature_names)

    return train_dataset, test_dataset



        
if __name__ == '__main__':

    feats2level = {
        'jitter': 'utt',
        'shimmer': 'utt',
        'rp': 'utt',
        'f0': 'frame',
        'energy': 'frame'
    }

    np.set_printoptions(precision=2)
    features_to_load = ['jitter', 'shimmer', 'rp', 'f0', 'energy']
    # features_to_load = ['energy', 'rp']
    # for feat in allfeats:
    
    base_folder = '/data/storage025/Turntaking/wavs_single_channel_normalized_nosil'

    # Initialize the merged dataset
    merged_dataset = FeatureDataset(
        base_folder_path=base_folder,
        feature_names=features_to_load,
    )



    # Perform train-test split
    train_dataset, test_dataset = train_test_split_by_subject(
        merged_dataset,
        feature_names=features_to_load,
        test_size=0.2,        # 20% for testing
        random_state=523       # For reproducibility
    )

    print(f"\nTraining Dataset: {len(train_dataset)} samples")
    print(f"Testing Dataset: {len(test_dataset)} samples")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,      # Adjust based on your memory constraints
        shuffle=True,       # Shuffle for training
        num_workers=4       # Adjust based on your CPU cores
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,      # Adjust as needed
        shuffle=False,      # No need to shuffle for testing
        num_workers=4       # Adjust based on your CPU cores
    )

    # Example: Iterate over the training DataLoader
    for batch_idx, batch in enumerate(train_loader):
        # Access features and metadata
        print(batch)
        print(len(batch))
        # ... and so on for other metadata and features

        # Example: Print batch information
        print(f"Train Batch {batch_idx + 1}:")
 

        # Break after the first batch for demonstration
        break

