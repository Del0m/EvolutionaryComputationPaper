"""
dataloader.py

NSL-KDD dataset loader for use with the standard GA and Hybrid GA-SA
feature selection experiments.

Dataset download from Kaggle:
https://www.kaggle.com/datasets/hassan06/nslkdd

Expected dataset files (in same directory as dataloader):
    - KDDTrain+.txt
    - KDDTest+.txt
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

_DIR = os.path.dirname(os.path.abspath(__file__))


# NSL-KDD col defs ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label",        # attack/normal class label
    "difficulty"    # NSL-KDD difficulty score (dropped before training)
]

# Categorical columns that need label encoding
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Total number of real/usable features (excludes label and difficulty columns from learning)
NUM_FEATURES = 41


# Core loader ---------------------------------------------------------------------------

def load_nslkdd(
    train_path=os.path.join(_DIR, "KDDTrain+.txt"),
    test_path=os.path.join(_DIR, "KDDTest+.txt"),
    binary_labels=True,
    scale=False,
    validation_split=0.2,
    random_state=42
):
    """
    Load and preprocess NSL-KDD dataset

    Dataloading Parameters
        train_path : str
            Path to KDDTrain+.txt (filenames in same dir)

        test_path : str
            Path to KDDTest+.txt (filenames in same dir)

        binary_labels : bool
            If True: collapse all attack types into single 'attack' label to
            produce binary classification problem: normal=0/attack=1

            If False: keep original multi-class attack labels encoded
            as integers.

        scale : bool
            If True: apply Min-Max normalization to all feature columns.

        validation_split : float
            Fraction of training set to hold out as a validation set.
            Set to 0.0 to void validation set.

        random_state : int
            Random seed for reproducible train/validation splits.

    Returns
        dict with keys:
            X_train      : pd.DataFrame  (training features)
            y_train      : pd.Series     (training labels)
            X_val        : pd.DataFrame  (validation features, empty if validation_split=0)
            y_val        : pd.Series     (validation labels,   empty if validation_split=0)
            X_test       : pd.DataFrame  (test features)
            y_test       : pd.Series     (test labels)
            feature_names: list[str]     (list of the 41 feature column names)
            label_encoder: LabelEncoder  (fitted encoder for decoding predictions)
            scaler       : MinMaxScaler or None
    """

    # Load raw files
    train_df = pd.read_csv(train_path, header=None, names=FEATURE_NAMES)
    test_df  = pd.read_csv(test_path,  header=None, names=FEATURE_NAMES)

    # Drop the difficulty score column (not a feature to learn)
    train_df.drop(columns=["difficulty"], inplace=True)
    test_df.drop(columns=["difficulty"],  inplace=True)

    # Encode categorical features
    encoders = {}
    for col in CATEGORICAL_COLS:
        label_enc = LabelEncoder()
        # Fit on combined data so test set categories are always covered
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        label_enc.fit(combined)
        train_df[col] = label_enc.transform(train_df[col])
        test_df[col]  = label_enc.transform(test_df[col])
        encoders[col] = label_enc

    # Encode labels
    label_encoder = LabelEncoder()

    if binary_labels:
            # Map everything that is not 'normal' to 'attack'
            train_df["label"] = train_df["label"].apply(
                lambda x: "normal" if x.strip() == "normal" else "attack"
            )
            test_df["label"] = test_df["label"].apply(
                lambda x: "normal" if x.strip() == "normal" else "attack"
            )

            # Hardcode normal=0, attack=1 instead of relying on LabelEncoder sort
            binary_map = {"normal": 0, "attack": 1}
            train_df["label"] = train_df["label"].map(binary_map)
            test_df["label"]  = test_df["label"].map(binary_map)

            # Manually set label_encoder classes
            label_encoder.classes_ = np.array(["normal", "attack"])

    else:
        combined_labels = pd.concat([train_df["label"], test_df["label"]], axis=0)
        label_encoder.fit(combined_labels)
        train_df["label"] = label_encoder.transform(train_df["label"])
        test_df["label"]  = label_encoder.transform(test_df["label"])

    # Split features and labels
    feature_cols = [c for c in train_df.columns if c != "label"]

    X_train_full = train_df[feature_cols].reset_index(drop=True)
    y_train_full = train_df["label"].reset_index(drop=True)
    X_test       = test_df[feature_cols].reset_index(drop=True)
    y_test       = test_df["label"].reset_index(drop=True)

    # Optional validation split
    if validation_split > 0.0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=validation_split,
            random_state=random_state,
            stratify=y_train_full
        )
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_val   = X_val.reset_index(drop=True)
        y_val   = y_val.reset_index(drop=True)
    else:
        X_train = X_train_full
        y_train = y_train_full
        X_val   = pd.DataFrame()
        y_val   = pd.Series(dtype=int)

    # Optional Min-Max scaling
    scaler = None
    if scale:
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_cols
        )
        if not X_val.empty:
            X_val = pd.DataFrame(
                scaler.transform(X_val),
                columns=feature_cols
            )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_cols
        )

    return {
        "X_train":       X_train,
        "y_train":       y_train,
        "X_val":         X_val,
        "y_val":         y_val,
        "X_test":        X_test,
        "y_test":        y_test,
        "feature_names": feature_cols,
        "label_encoder": label_encoder,
        "scaler":        scaler
    }

# Convenience helper used by both GAs ---------------------------------------------------------------------------

def apply_feature_mask(X, mask):
    """
    Filter a feature DataFrame down to the columns selected by a binary mask.
    (returns only columns with a 1)

    Parameters
        X : pd.DataFrame
            Full feature set with NUM_FEATURES (41) columns.
        mask : list or array-like of int (0 or 1), length NUM_FEATURES
            Binary chromosome from the GA. 1 = include feature, 0 = exclude.

    Returns
        pd.DataFrame
            Subset of X containing only the selected feature columns.

    Raises
        ValueError
            If the mask selects zero features (would produce an empty DataFrame).
    """
    selected_cols = [col for col, bit in zip(X.columns, mask) if bit == 1]
    if len(selected_cols) == 0:
        raise ValueError(
            "Feature mask selects zero features. "
            "Ensure at least one bit is set to 1 in the chromosome."
        )
    return X[selected_cols]

# Quick sanity check running dataloader ---------------------------------------------------------------------------

"""
This runs the sanity check twice, once in each mode.
For every integer label it prints the actual string name it maps to with the count and percentage. 

"""

if __name__ == "__main__":
    for mode, binary in [("Binary (normal vs attack)", True), ("Multi-class (all attack types)", False)]:

        print(f"\nMode: {mode}")
        print(f"{'_'*60}")
        data = load_nslkdd(binary_labels=binary)
        total = len(data['X_train']) + len(data['X_val'])

        print(f"Training samples   : {len(data['X_train'])}")
        print(f"Validation samples : {len(data['X_val'])} ({len(data['X_val']) / (total) * 100:.1f}% split)")
        print(f"Test samples       : {len(data['X_test'])}")
        print(f"Number of features : {len(data['feature_names'])}")
        print(f"\nLabel classes      : {list(data['label_encoder'].classes_)}")

        print(f"\nTraining label distribution:")
        counts = data['y_train'].value_counts().sort_index()
        for encoded_int, count in counts.items():
            class_name = data['label_encoder'].inverse_transform([encoded_int])[0]
            pct = count / len(data['y_train']) * 100
            print(f"  {encoded_int:>3}  |{class_name:<20}|  {count:>6} samples  ({pct:.3f}%)")
