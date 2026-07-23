"""
=========================================================
config.py
Employee Attrition Prediction Using ANN
=========================================================
"""

import os

# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# =========================================================
# DATASET
# =========================================================

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

DATASET_PATH = os.path.join(
    DATASET_DIR,
    "employee_attrition_dataset_10000.csv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed"
)

PROCESSED_DATASET_PATH = os.path.join(
    PROCESSED_DIR,
    "processed_dataset.csv"
)

# =========================================================
# MODEL
# =========================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

ANN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ann_model.keras"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

# =========================================================
# DATABASE
# =========================================================

DATABASE_NAME = "predictions.db"

DATABASE_URI = f"sqlite:///{DATABASE_NAME}"

# =========================================================
# REPORT
# =========================================================

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

DATASET_SUMMARY = os.path.join(
    REPORT_DIR,
    "dataset_summary.txt"
)

MISSING_VALUE_REPORT = os.path.join(
    REPORT_DIR,
    "missing_value_report.csv"
)

EVALUATION_REPORT = os.path.join(
    REPORT_DIR,
    "evaluation.txt"
)

# =========================================================
# STATIC
# =========================================================

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

CSS_DIR = os.path.join(
    STATIC_DIR,
    "css"
)

IMAGE_DIR = os.path.join(
    STATIC_DIR,
    "img"
)

EDA_DIR = os.path.join(
    STATIC_DIR,
    "eda"
)

# =========================================================
# TEMPLATE
# =========================================================

TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "templates"
)

# =========================================================
# RANDOM STATE
# =========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

VALIDATION_SPLIT = 0.20

# =========================================================
# MODEL PARAMETER
# =========================================================

EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.30

EARLY_STOPPING_PATIENCE = 10

REDUCE_LR_PATIENCE = 5

# =========================================================
# ANN ARCHITECTURE
# =========================================================

INPUT_LAYER = 128

HIDDEN_LAYER_1 = 64

HIDDEN_LAYER_2 = 32

OUTPUT_LAYER = 1

# =========================================================
# FLASK
# =========================================================

SECRET_KEY = "employee_attrition_ann_2026"

HOST = "0.0.0.0"

PORT = 5000

DEBUG = True

# =========================================================
# LABEL
# =========================================================

TARGET_COLUMN = "Attrition"

POSITIVE_LABEL = "Yes"

NEGATIVE_LABEL = "No"

# =========================================================
# CREATE PROJECT DIRECTORY
# =========================================================

DIRECTORIES = [

    DATASET_DIR,

    PROCESSED_DIR,

    MODEL_DIR,

    REPORT_DIR,

    STATIC_DIR,

    CSS_DIR,

    IMAGE_DIR,

    EDA_DIR,

    TEMPLATE_DIR

]

for directory in DIRECTORIES:

    os.makedirs(

        directory,

        exist_ok=True

    )

# =========================================================
# PROJECT INFORMATION
# =========================================================

PROJECT_NAME = "Employee Attrition Prediction"

PROJECT_VERSION = "1.0.0"

AUTHOR = "Muhammad Faris"

UNIVERSITY = "Universitas Bale Bandung"

STUDY_PROGRAM = "Teknik Informatika"

FRAMEWORK = "Flask"

AI_MODEL = "Artificial Neural Network"

DATABASE = "SQLite"

# =========================================================
# END CONFIG
# =========================================================