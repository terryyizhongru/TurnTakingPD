import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import pickle
import json

import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.base import BaseEstimator, clone

from load_feat_pd import load_feat  # Ensure this module is in your PYTHONPATH


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
            # self.base_folder_path = base_folder_path_unnorm if feature == 'energy' else base_folder_path
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
    def __init__(self, dataframe: pd.DataFrame, feature_names: List[str], feats2level: dict):
        """
        Initializes the subset dataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the subset of data.
            feature_names (List[str]): List of feature names included in the dataset.
        """
        self.data = dataframe.reset_index(drop=True)
        self.feature_names = feature_names
        self.feats2level = feats2level
        self.label_mapping = {
            '21': 0,
            '22': 1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves the sample and its label at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing concatenated features and the label.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        # Initialize a list to hold processed feature values
        processed_features = []
        
        # Iterate over each feature
        for feature in self.feature_names:
            level = self.feats2level.get(feature, 'frame')  # Default to 'frame' if not specified
            feature_data = self.data.iloc[idx][feature]
            
            if level == 'frame':
                # Ensure feature_data is a NumPy array for consistency
                if isinstance(feature_data, list):
                    feature_data = np.array(feature_data)
                elif isinstance(feature_data, np.ndarray):
                    pass
                else:
                    raise ValueError(f"Unsupported type for frame-level feature '{feature}': {type(feature_data)}")
                
                # Calculate mean and standard deviation
                mean = np.mean(feature_data)
                std = np.std(feature_data)
                
                # Append mean and std to the processed_features list
                processed_features.extend([mean, std])
            elif level == '3d':
                # Ensure feature_data is a NumPy array for consistency
                if isinstance(feature_data, list):
                    feature_data = np.array(feature_data)
                elif isinstance(feature_data, np.ndarray):
                    pass
                else:
                    raise ValueError(f"Unsupported type for 3D feature '{feature}': {type(feature_data)}")
        
                # Append mean and std to the processed_features list
                for dim in range(feature_data.shape[0]):
                    mean = np.mean(feature_data[dim])
                    std = np.std(feature_data[dim])
                    processed_features.extend([mean, std])
            else:
                # For 'utt' level features
                if isinstance(feature_data, np.ndarray):
                    if feature_data.size == 1:
                        # Single value, use it directly
                        value = feature_data.item()
                        processed_features.append(value)
                    else:
                        # Multiple values, flatten and extend
                        processed_features.extend(feature_data.flatten().tolist())
                else:
                    # Assume it's a scalar
                    processed_features.append(feature_data)
        
        # Convert the processed features list to a PyTorch tensor of type float32
        concate_features = torch.tensor(processed_features, dtype=torch.float32)
        
        # Retrieve and map the label
        group_id = self.data.iloc[idx]['group_id']
        if group_id not in self.label_mapping:
            raise ValueError(f"Unexpected group_id '{group_id}'. Expected '21' or '22'.")
        label = self.label_mapping[group_id]
        
        return concate_features, label

    def get_numpy(
        self,
        label_column: Optional[str] = None
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Converts the dataset to NumPy arrays for scikit-learn training.

        Args:
            label_column (str, optional): The name of the metadata column to use as labels.
                                          If None, only features are returned.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Features and labels as NumPy arrays.
                                                     If label_column is None, returns (X, None).
        """
        X = []
        y = [] if label_column else None

        for idx in range(len(self)):
            features, label = self.__getitem__(idx)
            X.append(features.numpy())
            if label_column:
                # Map the label using the predefined mapping
                group_id = self.data.iloc[idx][label_column]
                if group_id not in self.label_mapping:
                    raise ValueError(f"Unexpected group_id '{group_id}'. Expected '21' or '22'.")
                y.append(self.label_mapping[group_id])
        
        X = np.stack(X)  # Shape: (num_samples, num_features)
        y = np.array(y) if label_column else None

        return X, y

def train_test_split_by_subject(
    dataset: Dataset,
    feature_names: List[str],
    feats2level: dict,
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
    train_dataset = SubsetFeatureDataset(train_df, feature_names, feats2level)
    test_dataset = SubsetFeatureDataset(test_df, feature_names, feats2level)

    return train_dataset, test_dataset



def make_serializable(obj):
    """
    Recursively converts non-serializable objects into serializable formats.
    
    Args:
        obj: The object to serialize.
    
    Returns:
        A serializable version of the object.
    """
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, (dict)):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif callable(obj):
        return str(obj)  # Convert functions/methods to their string representation
    else:
        return obj


def nested_k_fold_cross_validation(
    dataset: Dataset,
    feature_names: List[str],
    feats2level: dict,
    model: BaseEstimator,
    outer_k: int = 5,
    inner_k: int = 3,
    random_state: int = 42
) -> Dict[str, List]:
    """
    Performs nested k-fold cross-validation using the existing train_test_split_by_subject function
    for the outer loop and GroupKFold for the inner loop.

    Args:
        dataset (Dataset): The merged dataset (FeatureDataset instance).
        feature_names (List[str]): List of feature names included in the dataset.
        model (BaseEstimator): The machine learning model to train (must follow scikit-learn's estimator API).
        outer_k (int): Number of outer folds.
        inner_k (int): Number of inner folds.
        random_state (int): Base random seed for reproducibility.

    Returns:
        Dict[str, List]: A dictionary containing performance metrics for each outer fold.
    """
    # Convert merged data to DataFrame
    merged_data = dataset.merged_data  # Assuming 'merged_data' is a list of dicts
    df = pd.DataFrame(merged_data)

    # Extract unique subject_ids and their genders
    subjects_df = df[['subject_id', 'gender']].drop_duplicates()

    # Drop subjects with missing gender
    subjects_df = subjects_df.dropna(subset=['gender'])

    # Ensure 'gender' is string type
    subjects_df['gender'] = subjects_df['gender'].astype(str)

    # Initialize StratifiedKFold for outer loop based on gender
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=random_state)

    # Prepare data for outer loop
    X_subjects_outer = subjects_df['subject_id']
    y_subjects_outer = subjects_df['gender']

    # Initialize a dictionary to store metrics
    metrics = {
        'accuracy': [],
        'roc_auc': [],
        'sensitivity': [],
        'specificity': [],
        'confusion_matrix': [],
        'classification_report': []
    }

    # Outer Loop
    for outer_fold, (train_subjects_idx, test_subjects_idx) in enumerate(outer_cv.split(X_subjects_outer, y_subjects_outer), 1):
        print(f"\n=== Outer Fold {outer_fold} ===")

        # Extract outer train and test subject_ids
        outer_train_subjects = subjects_df.iloc[train_subjects_idx]
        outer_test_subjects = subjects_df.iloc[test_subjects_idx]

        outer_train_ids = set(outer_train_subjects['subject_id'])
        outer_test_ids = set(outer_test_subjects['subject_id'])

        # Assign samples to outer train and test sets based on subject_id
        outer_train_df = df[df['subject_id'].isin(outer_train_ids)].reset_index(drop=True)
        outer_test_df = df[df['subject_id'].isin(outer_test_ids)].reset_index(drop=True)

        # Create SubsetFeatureDataset instances
        outer_train_dataset = SubsetFeatureDataset(outer_train_df, feature_names, feats2level)
        outer_test_dataset = SubsetFeatureDataset(outer_test_df, feature_names, feats2level)

        print(f"Training subjects: {len(outer_train_ids)}")
        print(f"Testing subjects: {len(outer_test_ids)}")
        print(f"Total training samples: {len(outer_train_df)}")
        print(f"Total testing samples: {len(outer_test_df)}")

        # Extract features and labels for outer train and test sets
        X_outer_train, y_outer_train = outer_train_dataset.get_numpy(label_column='group_id')
        X_outer_test, y_outer_test = outer_test_dataset.get_numpy(label_column='group_id')

        # Initialize GroupKFold for inner loop based on subject_id
        inner_cv = GroupKFold(n_splits=inner_k)

        # Extract groups for inner loop
        groups_outer_train = outer_train_df['subject_id'].values

        best_score = -np.inf
        best_model = None

        # Inner Loop: Hyperparameter Tuning or Model Validation
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train, groups=groups_outer_train), 1):
            print(f"  --- Inner Fold {inner_fold} ---")

            X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
            y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]  # Corrected here

            # Clone the model to ensure independence
            model_clone = clone(model)

            # Train the model on inner training set
            model_clone.fit(X_inner_train, y_inner_train)

            # Evaluate on inner validation set
            score = model_clone.score(X_inner_val, y_inner_val)
            print(f"    Inner Fold {inner_fold} Score: {score:.4f}")

            # Update best model if current model is better
            if score > best_score:
                best_score = score
                best_model = clone(model_clone)

        print(f"  Best Inner Fold Score: {best_score:.4f}")

        # Train the best model on the entire outer training set
        best_model.fit(X_outer_train, y_outer_train)

        # Predict on the outer test set
        y_pred = best_model.predict(X_outer_test)
        if hasattr(best_model, "predict_proba"):
            y_pred_proba = best_model.predict_proba(X_outer_test)[:, 1]
        else:
            # If model does not support predict_proba, use decision function or default probabilities
            y_pred_proba = best_model.decision_function(X_outer_test)
            # Ensure y_pred_proba is positive if necessary
            if np.any(y_pred_proba < 0):
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())

        # Calculate Accuracy
        accuracy = accuracy_score(y_outer_test, y_pred)
        metrics['accuracy'].append(accuracy)
        print(f"  Outer Fold {outer_fold} Accuracy: {accuracy:.4f}")

        # Calculate ROC-AUC
        try:
            roc_auc = roc_auc_score(y_outer_test, y_pred_proba)
        except ValueError as e:
            print(f"    ROC-AUC Calculation Error: {e}")
            roc_auc = np.nan
        metrics['roc_auc'].append(roc_auc)
        print(f"  Outer Fold {outer_fold} ROC-AUC: {roc_auc:.4f}")

        # Generate Confusion Matrix
        conf_matrix = confusion_matrix(y_outer_test, y_pred)
        metrics['confusion_matrix'].append(conf_matrix.tolist())  # Convert ndarray to list
        print(f"  Outer Fold {outer_fold} Confusion Matrix:\n{conf_matrix}")

        # Generate Classification Report
        class_report = classification_report(y_outer_test, y_pred, output_dict=True)
        # Convert any numpy types within the dict to native Python types
        class_report_serializable = make_serializable(class_report)
        metrics['classification_report'].append(class_report_serializable)
        print(f"  Outer Fold {outer_fold} Classification Report:\n{classification_report(y_outer_test, y_pred)}")

        # Calculate Sensitivity and Specificity
        if conf_matrix.shape == (2, 2):
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'].append(sensitivity)
            metrics['specificity'].append(specificity)
            print(f"  Outer Fold {outer_fold} Sensitivity (Recall): {sensitivity:.4f}")
            print(f"  Outer Fold {outer_fold} Specificity: {specificity:.4f}")
        else:
            print("  Confusion matrix is not binary. Skipping Sensitivity and Specificity calculation.")
            metrics['sensitivity'].append(None)
            metrics['specificity'].append(None)

    # Convert all metrics to serializable formats
    serializable_metrics = make_serializable(metrics)

    # Attempt to serialize and catch potential errors
    try:
        with open('nested_cv_metrics.json', 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        print("\nNested K-Fold Cross-Validation completed and metrics saved to 'nested_cv_metrics.json'.")
    except TypeError as e:
        print(f"\nSerialization Error: {e}")
        # Optionally, inspect the metrics to identify problematic entries
        for key, value in metrics.items():
            for idx, item in enumerate(value):
                try:
                    json.dumps(item)
                except TypeError:
                    print(f"Non-serializable object found in '{key}' at index {idx}: {item}")


def make_serializable(obj):
    """
    递归地将不可序列化的对象转换为可序列化格式。
    
    Args:
        obj: 需要序列化的对象。
    
    Returns:
        可序列化的对象。
    """
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif callable(obj):
        return str(obj)  # 将函数或方法转换为字符串表示
    else:
        return obj

def average_metrics(json_path: str) -> Dict[str, any]:
    """
    计算嵌套交叉验证保存的metrics的平均值和标准差。
    
    Args:
        json_path (str): 存储metrics的JSON文件路径。
    
    Returns:
        Dict[str, any]: 包含各个指标平均值和标准差的字典。
    """
    # 读取JSON文件
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    # 初始化用于存储平均值和标准差的字典
    averages = {}
    stds = {}
    
    # 简单数值指标
    simple_metrics = ['accuracy', 'roc_auc', 'sensitivity', 'specificity']
    
    for metric in simple_metrics:
        values = metrics.get(metric, [])
        # 转换为numpy数组，忽略None或NaN
        values = np.array([v for v in values if v is not None and not np.isnan(v)])
        if len(values) > 0:
            averages[metric] = np.mean(values)
            stds[metric] = np.std(values)
        else:
            averages[metric] = None
            stds[metric] = None
    
    # 处理混淆矩阵，计算平均混淆矩阵
    if 'confusion_matrix' in metrics and len(metrics['confusion_matrix']) > 0:
        conf_matrices = [np.array(cm) for cm in metrics['confusion_matrix']]
        avg_conf_matrix = np.mean(conf_matrices, axis=0)
        averages['confusion_matrix'] = avg_conf_matrix.tolist()
    else:
        averages['confusion_matrix'] = None
    
    # 处理分类报告，聚合每个类别的指标
    if 'classification_report' in metrics and len(metrics['classification_report']) > 0:
        # 假设是二分类，包含'0', '1', 'accuracy', 'macro avg', 'weighted avg'
        report_keys = metrics['classification_report'][0].keys()
        aggregated_report = {}
        for key in report_keys:
            if key == 'accuracy':
                # 'accuracy' 是一个单一的浮点数
                values = [m[key] for m in metrics['classification_report'] if key in m and isinstance(m[key], (int, float))]
                if len(values) > 0:
                    aggregated_report[key] = {
                        'accuracy': np.mean(values)
                    }
            elif key in ['macro avg', 'weighted avg']:
                # 这些是包含 'precision', 'recall', 'f1-score', 'support' 的字典
                values = [m[key] for m in metrics['classification_report'] if key in m and isinstance(m[key], dict)]
                if len(values) > 0:
                    aggregated_report[key] = {
                        'precision': np.mean([v['precision'] for v in values if 'precision' in v]),
                        'recall': np.mean([v['recall'] for v in values if 'recall' in v]),
                        'f1-score': np.mean([v['f1-score'] for v in values if 'f1-score' in v]),
                        'support': int(np.mean([v['support'] for v in values if 'support' in v]))
                    }
            else:
                # 每个类别的指标
                values = [m[key] for m in metrics['classification_report'] if key in m and isinstance(m[key], dict)]
                if len(values) > 0:
                    aggregated_report[key] = {
                        'precision': np.mean([v['precision'] for v in values if 'precision' in v]),
                        'recall': np.mean([v['recall'] for v in values if 'recall' in v]),
                        'f1-score': np.mean([v['f1-score'] for v in values if 'f1-score' in v]),
                        'support': int(np.mean([v['support'] for v in values if 'support' in v]))
                    }
        averages['classification_report'] = aggregated_report
    else:
        averages['classification_report'] = None
    
    # 打印平均值和标准差
    print("=== Metrics Averages ===")
    for metric in simple_metrics:
        if averages[metric] is not None:
            print(f"{metric.capitalize()}: {averages[metric]:.4f} ± {stds[metric]:.4f}")
        else:
            print(f"{metric.capitalize()}: N/A")
    
    if averages['confusion_matrix'] is not None:
        print("\nAverage Confusion Matrix:")
        print(np.array(averages['confusion_matrix']))
    else:
        print("\nAverage Confusion Matrix: N/A")
    
    if averages['classification_report'] is not None:
        print("\nAggregated Classification Report:")
        for key, value in averages['classification_report'].items():
            print(f"{key}:")
            for metric_name, metric_value in value.items():
                if metric_value is not None:
                    print(f"  {metric_name}: {metric_value:.4f}")
                else:
                    print(f"  {metric_name}: N/A")
    else:
        print("\nAggregated Classification Report: N/A")
    
    return averages


if __name__ == '__main__':
    
    import sys
    np.set_printoptions(precision=2)

    config_filepath = sys.argv[1]

    with open(config_filepath, "r") as json_file:
        config_data = json.load(json_file)

    feats2level = config_data["feats2level"]
    allfeats_batch1 = config_data["allfeats_batch1"]
    allfeats_batch2 = config_data["allfeats_batch2"]
    features_to_load = config_data["features_to_load"]
    merged_data_pkl = config_data["merged_data_pkl"]
    base_folder = config_data["base_folder"]
    # for feat in allfeats:
    

    # Initialize the merged dataset
    merged_dataset = FeatureDataset(
        base_folder_path=base_folder,
        feature_names=features_to_load,
    )

    # save merged dataset as pcikle
    with open(merged_data_pkl, 'wb') as f:
        pickle.dump(merged_dataset, f)