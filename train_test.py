from dataloader import *
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

if __name__ == '__main__':

    config_filepath = "config.json"

    with open(config_filepath, "r") as json_file:
        config_data = json.load(json_file)

    feats2level = config_data["feats2level"]
    allfeats_batch1 = config_data["allfeats_batch1"]
    allfeats_batch2 = config_data["allfeats_batch2"]
    features_to_load = config_data["features_to_load"]
    merged_data_pkl = config_data["merged_data_pkl"]
    base_folder = config_data["base_folder"]
    
     # Load merged dataset
    pickle_in = open(merged_data_pkl, "rb")
    merged_dataset = pickle.load(pickle_in)
    
    # contrast = merged_dataset.merged_data[0]['contrast']
    # # contrast = np.array(contrast)
    # print(type(contrast))
    # print(contrast.shape)

    # print(len(contrast))
    # print(contrast[0].shape)
    
        
    

    # Perform train-test split
    
    train_dataset, test_dataset = train_test_split_by_subject(
        merged_dataset,
        feature_names=features_to_load[:5],
        feats2level = feats2level,
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
    
    # Get NumPy arrays for scikit-learn (classification based on 'group_id')
    X_train, y_train = train_dataset.get_numpy(label_column='group_id')
    X_test, y_test = test_dataset.get_numpy(label_column='group_id')

    print("\nShape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)

    print("\nFirst 5 training labels:", y_train[:5])
    print("First 5 testing labels:", y_test[:5])



    


# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,          # Number of trees
    max_depth=None,            # Maximum depth of each tree
    min_samples_split=2,       # Minimum samples to split a node
    min_samples_leaf=1,        # Minimum samples at a leaf node
    random_state=13,           # Seed for reproducibility
    class_weight='balanced'    # Adjust weights inversely proportional to class frequencies
)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Generate Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# Generate Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate Sensitivity and Specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

