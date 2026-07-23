"""
=========================================================
Employee Attrition Prediction Using ANN
Training Script
=========================================================
"""

import os
import random
import warnings
import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.optimizers import Adam

from config import *

warnings.filterwarnings("ignore")

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

# ==========================================================
# RANDOM SEED
# ==========================================================

random.seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

tf.random.set_seed(RANDOM_STATE)

# ==========================================================
# CREATE DIRECTORY
# ==========================================================

os.makedirs(

    MODEL_DIR,

    exist_ok=True

)

os.makedirs(

    REPORT_DIR,

    exist_ok=True

)

os.makedirs(

    EDA_DIR,

    exist_ok=True

)

# ==========================================================
# LOAD PROCESSED DATASET
# ==========================================================

def load_processed_dataset():

    logging.info(
        "Loading processed dataset..."
    )

    if not os.path.exists(
        PROCESSED_DATASET_PATH
    ):

        raise FileNotFoundError(

            "processed_dataset.csv tidak ditemukan."

        )

    df = pd.read_csv(
        PROCESSED_DATASET_PATH
    )

    logging.info(
        f"Dataset Shape : {df.shape}"
    )

    return df

# ==========================================================
# SPLIT DATASET
# ==========================================================

def split_dataset(df):

    logging.info(
        "Splitting dataset..."
    )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    logging.info(

        f"Train : {X_train.shape}"

    )

    logging.info(

        f"Test  : {X_test.shape}"

    )

    return (

        X_train,

        X_test,

        y_train,

        y_test

    )

# ==========================================================
# BUILD ANN
# ==========================================================

def build_model(input_dim):

    logging.info(
        "Building ANN..."
    )

    model = Sequential()

    model.add(

        Dense(

            INPUT_LAYER,

            activation="relu",

            input_shape=(input_dim,)

        )

    )

    model.add(

        BatchNormalization()

    )

    model.add(

        Dropout(

            DROPOUT_RATE

        )

    )

    model.add(

        Dense(

            HIDDEN_LAYER_1,

            activation="relu"

        )

    )

    model.add(

        BatchNormalization()

    )

    model.add(

        Dropout(

            DROPOUT_RATE

        )

    )

    model.add(

        Dense(

            HIDDEN_LAYER_2,

            activation="relu"

        )

    )

    model.add(

        Dense(

            OUTPUT_LAYER,

            activation="sigmoid"

        )

    )

    model.compile(

        optimizer=Adam(

            learning_rate=LEARNING_RATE

        ),

        loss="binary_crossentropy",

        metrics=[

            "accuracy"

        ]

    )

    model.summary()

    return model

# ==========================================================
# CALLBACKS
# ==========================================================

def create_callbacks():

    logging.info(
        "Creating callbacks..."
    )

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=EARLY_STOPPING_PATIENCE,

        restore_best_weights=True,

        verbose=1

    )

    reduce_lr = ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=REDUCE_LR_PATIENCE,

        min_lr=1e-6,

        verbose=1

    )

    checkpoint = ModelCheckpoint(

        filepath=ANN_MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1

    )

    return [

        early_stopping,

        reduce_lr,

        checkpoint

    ]


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(

    model,

    X_train,

    y_train

):

    logging.info(
        "Training model..."
    )

    callbacks = create_callbacks()

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {
        0: class_weights[0],
        1: class_weights[1]
    }

    print("\nClass Weight :", class_weights)

    history = model.fit(

        X_train,

        y_train,

        validation_split=VALIDATION_SPLIT,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        class_weight=class_weights,

        verbose=1

    )

    logging.info(
        "Training completed."
    )

    return history


# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

def save_final_model(model):

    model.save(

        ANN_MODEL_PATH

    )

    logging.info(

        f"Model saved : {ANN_MODEL_PATH}"

    )


# ==========================================================
# SAVE TRAINING ACCURACY
# ==========================================================

def save_accuracy_plot(history):

    plt.figure(

        figsize=(8,5)

    )

    plt.plot(

        history.history["accuracy"],

        label="Training"

    )

    plt.plot(

        history.history["val_accuracy"],

        label="Validation"

    )

    plt.title(

        "Training Accuracy"

    )

    plt.xlabel(

        "Epoch"

    )

    plt.ylabel(

        "Accuracy"

    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            EDA_DIR,

            "training_accuracy.png"

        )

    )

    plt.close()

    logging.info(

        "training_accuracy.png saved."

    )


# ==========================================================
# SAVE TRAINING LOSS
# ==========================================================

def save_loss_plot(history):

    plt.figure(

        figsize=(8,5)

    )

    plt.plot(

        history.history["loss"],

        label="Training"

    )

    plt.plot(

        history.history["val_loss"],

        label="Validation"

    )

    plt.title(

        "Training Loss"

    )

    plt.xlabel(

        "Epoch"

    )

    plt.ylabel(

        "Loss"

    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            EDA_DIR,

            "training_loss.png"

        )

    )

    plt.close()

    logging.info(

        "training_loss.png saved."

    )


# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

def save_training_history(history):

    history_df = pd.DataFrame(

        history.history

    )

    history_df.to_csv(

        os.path.join(

            REPORT_DIR,

            "training_history.csv"

        ),

        index=False

    )

    logging.info(

        "training_history.csv saved."

    )

# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(

    model,

    X_test,

    y_test

):

    logging.info(
        "Evaluating model..."
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    probability = model.predict(
        X_test,
        verbose=0
    )

    probability = probability.flatten()

    prediction = prediction.flatten()

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    accuracy = accuracy_score(

        y_test,

        prediction

    )

    precision = precision_score(

        y_test,

        prediction,

        zero_division=0

    )

    recall = recall_score(

        y_test,

        prediction,

        zero_division=0

    )

    f1 = f1_score(

        y_test,

        prediction,

        zero_division=0

    )

    roc = roc_auc_score(

        y_test,

        probability

    )

    report = classification_report(

        y_test,

        prediction,

        digits=4,

        zero_division=0

    )

    matrix = confusion_matrix(

        y_test,

        prediction

    )

    logging.info(

        f"Accuracy : {accuracy:.4f}"

    )

    logging.info(

        f"Precision : {precision:.4f}"

    )

    logging.info(

        f"Recall : {recall:.4f}"

    )

    logging.info(

        f"F1 Score : {f1:.4f}"

    )

    logging.info(

        f"ROC AUC : {roc:.4f}"

    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc": roc,

        "report": report,

        "matrix": matrix,

        "prediction": prediction,

        "probability": probability,

        "y_true": y_test,

    }


# ==========================================================
# SAVE EVALUATION REPORT
# ==========================================================

def save_evaluation(metrics):

    with open(

        EVALUATION_REPORT,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("="*70 + "\n")

        file.write("ANN EVALUATION REPORT\n")

        file.write("="*70 + "\n\n")

        file.write(

            f"Accuracy  : {metrics['accuracy']:.4f}\n"

        )

        file.write(

            f"Precision : {metrics['precision']:.4f}\n"

        )

        file.write(

            f"Recall    : {metrics['recall']:.4f}\n"

        )

        file.write(

            f"F1 Score  : {metrics['f1']:.4f}\n"

        )

        file.write(

            f"ROC AUC   : {metrics['roc']:.4f}\n"

        )

        file.write("\n")

        file.write("="*70 + "\n")

        file.write("CLASSIFICATION REPORT\n")

        file.write("="*70 + "\n\n")

        file.write(

            metrics["report"]

        )

    logging.info(

        "evaluation.txt saved."

    )


# ==========================================================
# SAVE CONFUSION MATRIX
# ==========================================================

def save_confusion_matrix(metrics):

    disp = ConfusionMatrixDisplay(

        confusion_matrix=metrics["matrix"],

        display_labels=[

            "Stay",

            "Leave"

        ]

    )

    fig, ax = plt.subplots(

        figsize=(6,6)

    )

    disp.plot(

        ax=ax,

        colorbar=False

    )

    plt.title(

        "Confusion Matrix"

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            EDA_DIR,

            "confusion_matrix.png"

        )

    )

    plt.close()

    logging.info(

        "confusion_matrix.png saved."

    )

# ==========================================================
# SAVE ROC CURVE
# ==========================================================

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve
)


def save_roc_curve(metrics):

    fpr, tpr, _ = roc_curve(

        metrics["y_true"],

        metrics["probability"]

    )

    plt.figure(figsize=(8,6))

    plt.plot(

        fpr,

        tpr,

        label=f"AUC = {metrics['roc']:.4f}"

    )

    plt.plot(

        [0,1],

        [0,1],

        linestyle="--"

    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            EDA_DIR,

            "roc_curve.png"

        )

    )

    plt.close()

    logging.info(
        "roc_curve.png saved."
    )


# ==========================================================
# SAVE PRECISION RECALL CURVE
# ==========================================================

def save_precision_recall_curve(metrics):

    precision, recall, _ = precision_recall_curve(

        metrics["y_true"],

        metrics["probability"]

    )

    plt.figure(figsize=(8,6))

    plt.plot(

        recall,

        precision

    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            EDA_DIR,

            "precision_recall_curve.png"

        )

    )

    plt.close()

    logging.info(
        "precision_recall_curve.png saved."
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 70)
    print("EMPLOYEE ATTRITION ANN TRAINING")
    print("=" * 70)
    print()

    # Load dataset
    df = load_processed_dataset()

    # Split dataset
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(df)

    # Build model
    model = build_model(

        X_train.shape[1]

    )

    # Train
    history = train_model(

        model,

        X_train,

        y_train

    )

    # Save model & history
    save_final_model(model)

    save_accuracy_plot(history)

    save_loss_plot(history)

    save_training_history(history)

    # Evaluate
    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )

    # Tambahkan ground truth agar bisa dipakai kurva
    metrics["y_true"] = y_test

    # Save evaluation
    save_evaluation(metrics)

    save_confusion_matrix(metrics)

    save_roc_curve(metrics)

    save_precision_recall_curve(metrics)

    print()
    print("=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()

    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"ROC AUC   : {metrics['roc']:.4f}")

    print()
    print("Model")
    print(f"  {ANN_MODEL_PATH}")

    print()
    print("Report")
    print(f"  {EVALUATION_REPORT}")

    print()
    print("Training Graph")
    print(f"  {os.path.join(EDA_DIR,'training_accuracy.png')}")
    print(f"  {os.path.join(EDA_DIR,'training_loss.png')}")

    print()
    print("Evaluation Graph")
    print(f"  {os.path.join(EDA_DIR,'confusion_matrix.png')}")
    print(f"  {os.path.join(EDA_DIR,'roc_curve.png')}")
    print(f"  {os.path.join(EDA_DIR,'precision_recall_curve.png')}")

    print()
    print("=" * 70)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        logging.exception(error)

        print()
        print("=" * 70)
        print("TRAINING FAILED")
        print("=" * 70)
        print(error)