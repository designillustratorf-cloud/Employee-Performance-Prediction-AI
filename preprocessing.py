"""
=========================================================
Employee Attrition Prediction Using ANN
Professional Preprocessing Pipeline
=========================================================

Author      : Muhammad Faris
University  : Universitas Bale Bandung
Program     : Teknik Informatika

=========================================================
"""

import os
import warnings
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from config import *

warnings.filterwarnings("ignore")

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

# ==========================================================
# CREATE DIRECTORY
# ==========================================================

for folder in [

    MODEL_DIR,

    REPORT_DIR,

    PROCESSED_DIR,

    EDA_DIR

]:

    os.makedirs(

        folder,

        exist_ok=True

    )

# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    logging.info(
        "Loading dataset..."
    )

    if not os.path.exists(
        DATASET_PATH
    ):

        raise FileNotFoundError(

            f"Dataset tidak ditemukan :\n{DATASET_PATH}"

        )

    df = pd.read_csv(
        DATASET_PATH
    )

    logging.info(
        f"Dataset Shape : {df.shape}"
    )

    return df

# ==========================================================
# VALIDATE DATASET
# ==========================================================

def validate_dataset(df):

    logging.info(
        "Validating dataset..."
    )

    if df.empty:

        raise ValueError(
            "Dataset kosong."
        )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target '{TARGET_COLUMN}' tidak ditemukan."
        )

    if df.shape[1] < 2:

        raise ValueError(
            "Jumlah kolom tidak valid."
        )

    logging.info(
        "Dataset validation passed."
    )

# ==========================================================
# BASIC INFORMATION
# ==========================================================

def dataset_information(df):

    logging.info(
        "Creating dataset summary..."
    )

    with open(

        DATASET_SUMMARY,

        "w",

        encoding="utf-8"

    ) as f:

        f.write("="*70 + "\n")

        f.write("EMPLOYEE ATTRITION DATASET SUMMARY\n")

        f.write("="*70 + "\n\n")

        f.write(
            f"Rows    : {df.shape[0]}\n"
        )

        f.write(
            f"Columns : {df.shape[1]}\n\n"
        )

        f.write("COLUMN\n")

        f.write("-"*70 + "\n")

        for col in df.columns:

            f.write(
                f"{col}\n"
            )

        f.write("\n")

        f.write("="*70 + "\n")

        f.write("DESCRIBE\n")

        f.write("="*70 + "\n\n")

        f.write(

            str(

                df.describe(

                    include="all"

                )

            )

        )

    logging.info(
        "Dataset summary saved."
    )


# ==========================================================
# REMOVE DUPLICATE
# ==========================================================

def remove_duplicate(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    logging.info(

        f"Duplicate Removed : {before-after}"

    )

    return df


# ==========================================================
# REMOVE ID COLUMN
# ==========================================================

def remove_identifier(df):

    identifier = [

        "Employee_ID",

        "EmployeeID",

        "ID",

        "Id"

    ]

    for col in identifier:

        if col in df.columns:

            df.drop(

                columns=col,

                inplace=True

            )

            logging.info(

                f"Column Removed : {col}"

            )

    return df


# ==========================================================
# MISSING VALUE REPORT
# ==========================================================

def missing_value_report(df):

    missing = df.isnull().sum()

    percentage = (

        missing / len(df)

    ) * 100

    report = pd.DataFrame({

        "Column":

            missing.index,

        "Missing":

            missing.values,

        "Percentage":

            percentage.values

    })

    report = report.sort_values(

        by="Missing",

        ascending=False

    )

    report.to_csv(

        MISSING_VALUE_REPORT,

        index=False

    )

    logging.info(

        "Missing value report saved."

    )


# ==========================================================
# HANDLE MISSING VALUE
# ==========================================================

def fill_missing_value(df):

    logging.info(

        "Handling missing value..."

    )

    categorical = df.select_dtypes(

        include="object"

    ).columns

    numerical = df.select_dtypes(

        include=np.number

    ).columns

    for col in categorical:

        df[col] = df[col].fillna(

            df[col].mode()[0]

        )

    for col in numerical:

        df[col] = df[col].fillna(

            df[col].median()

        )

    return df


# ==========================================================
# TARGET ENCODING
# ==========================================================

def encode_target(df):

    if TARGET_COLUMN not in df.columns:

        raise Exception(

            "Target column tidak ditemukan."

        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({

        POSITIVE_LABEL:1,

        NEGATIVE_LABEL:0

    })

    logging.info(

        "Target encoded."

    )

    return df

# ==========================================================
# REMOVE OUTLIER (IQR)
# ==========================================================

def remove_outlier(df):

    logging.info(
        "Removing outlier..."
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if TARGET_COLUMN in numeric_columns:

        numeric_columns.remove(
            TARGET_COLUMN
        )

    mask = pd.Series(
        True,
        index=df.index
    )

    for column in numeric_columns:

        Q1 = df[column].quantile(0.25)

        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR

        upper = Q3 + 1.5 * IQR

        mask &= (

            (df[column] >= lower)

            &

            (df[column] <= upper)

        )

    before = len(df)

    df = df.loc[mask].reset_index(drop=True)

    after = len(df)

    logging.info(

        f"Removed {before-after} rows."

    )

    return df

# ==========================================================
# HISTOGRAM
# ==========================================================

def create_histogram(df):

    logging.info(
        "Creating histogram..."
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        plt.figure(figsize=(8,5))

        plt.hist(
            df[column],
            bins=30
        )

        plt.title(column)

        plt.xlabel(column)

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                EDA_DIR,

                f"{column}_histogram.png"

            )

        )

        plt.close()


# ==========================================================
# BOXPLOT
# ==========================================================

def create_boxplot(df):

    logging.info(
        "Creating boxplot..."
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        plt.figure(figsize=(8,5))

        plt.boxplot(df[column])

        plt.title(column)

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                EDA_DIR,

                f"{column}_boxplot.png"

            )

        )

        plt.close()


# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

def create_target_distribution(df):

    logging.info(
        "Creating target distribution..."
    )

    target = df[TARGET_COLUMN].value_counts()

    plt.figure(figsize=(6,6))

    plt.pie(

        target,

        labels=["Stay","Leave"],

        autopct="%1.1f%%",

        startangle=90

    )

    plt.title("Employee Attrition")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            EDA_DIR,

            "target_distribution.png"

        )

    )

    plt.close()


# ==========================================================
# BAR CHART CATEGORICAL
# ==========================================================

def create_categorical_plot(df):

    logging.info(
        "Creating categorical plot..."
    )

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    for column in categorical_columns:

        plt.figure(figsize=(10,5))

        df[column].value_counts().plot(
            kind="bar"
        )

        plt.title(column)

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                EDA_DIR,

                f"{column}_bar.png"

            )

        )

        plt.close()


# ==========================================================
# CORRELATION MATRIX
# ==========================================================

def create_correlation_matrix(df):

    logging.info(
        "Creating correlation matrix..."
    )

    corr = df.corr(
        numeric_only=True
    )

    fig, ax = plt.subplots(
        figsize=(12,10)
    )

    image = ax.imshow(corr)

    plt.colorbar(image)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(
        corr.columns,
        rotation=90,
        fontsize=8
    )

    ax.set_yticklabels(
        corr.columns,
        fontsize=8
    )

    for i in range(len(corr.columns)):

        for j in range(len(corr.columns)):

            ax.text(

                j,

                i,

                f"{corr.iloc[i,j]:.2f}",

                ha="center",

                va="center",

                fontsize=5

            )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            EDA_DIR,

            "correlation_matrix.png"

        )

    )

    plt.close()

# ==========================================================
# GENERATE ALL EDA
# ==========================================================

def generate_eda(df):

    logging.info(
        "Generating EDA..."
    )

    create_histogram(df)

    create_boxplot(df)

    create_target_distribution(df)

    create_categorical_plot(df)

    create_correlation_matrix(df)

    logging.info(
        "EDA completed."
    )

# ==========================================================
# BUILD PREPROCESSOR
# ==========================================================

def build_preprocessor(df):

    logging.info(
        "Building preprocessing pipeline..."
    )

    feature_df = df.drop(
        columns=[TARGET_COLUMN]
    )

    categorical_columns = feature_df.select_dtypes(
        include="object"
    ).columns.tolist()

    numerical_columns = feature_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )

        ]

    )

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )

        ]

    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_pipeline,
                numerical_columns
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )

        ]

    )

    logging.info(
        "Preprocessor created."
    )

    return (

        preprocessor,

        numerical_columns,

        categorical_columns

    )


# ==========================================================
# TRANSFORM DATASET
# ==========================================================

def transform_dataset(

    df,

    preprocessor

):

    logging.info(
        "Transforming dataset..."
    )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    X_processed = preprocessor.fit_transform(X)

    if hasattr(
        X_processed,
        "toarray"
    ):

        X_processed = X_processed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(

        X_processed,

        columns=feature_names

    )

    processed_df[TARGET_COLUMN] = y.values

    logging.info(
        "Dataset transformed."
    )

    return (

        processed_df,

        feature_names

    )


# ==========================================================
# SAVE PREPROCESSOR
# ==========================================================

def save_preprocessor(

    preprocessor

):

    joblib.dump(

        preprocessor,

        PREPROCESSOR_PATH

    )

    logging.info(
        "preprocessor.pkl saved."
    )


# ==========================================================
# SAVE FEATURE COLUMNS
# ==========================================================

def save_feature_columns(

    feature_columns

):

    joblib.dump(

        list(feature_columns),

        FEATURE_COLUMNS_PATH

    )

    logging.info(
        "feature_columns.pkl saved."
    )


# ==========================================================
# SAVE PROCESSED DATASET
# ==========================================================

def save_processed_dataset(

    processed_df

):

    processed_df.to_csv(

        PROCESSED_DATASET_PATH,

        index=False

    )

    logging.info(
        "processed_dataset.csv saved."
    )


# ==========================================================
# SHOW INFORMATION
# ==========================================================

def preprocessing_information(

    processed_df,

    feature_columns

):

    print()

    print("=" * 60)

    print("PREPROCESSING FINISHED")

    print("=" * 60)

    print()

    print(
        f"Processed Shape : {processed_df.shape}"
    )

    print(
        f"Number of Features : {len(feature_columns)}"
    )

    print(
        f"Processed Dataset : {PROCESSED_DATASET_PATH}"
    )

    print(
        f"Preprocessor : {PREPROCESSOR_PATH}"
    )

    print(
        f"Feature Columns : {FEATURE_COLUMNS_PATH}"
    )

    print()

    print("=" * 60)

    print()

# ==========================================================
# MAIN PREPROCESSING
# ==========================================================

def main():

    print()
    print("=" * 70)
    print("EMPLOYEE ATTRITION PREPROCESSING")
    print("=" * 70)
    print()

    # ------------------------------------------------------
    # LOAD DATASET
    # ------------------------------------------------------

    df = load_dataset()

    dataset_information(df)

    # ------------------------------------------------------
    # DATA CLEANING
    # ------------------------------------------------------

    df = remove_duplicate(df)

    df = remove_identifier(df)

    missing_value_report(df)

    df = fill_missing_value(df)

    df = encode_target(df)

    df = remove_outlier(df)

    # ------------------------------------------------------
    # EDA
    # ------------------------------------------------------

    generate_eda(df)

    # ------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------

    (
        preprocessor,
        numerical_columns,
        categorical_columns
    ) = build_preprocessor(df)

    (
        processed_df,
        feature_columns
    ) = transform_dataset(
        df,
        preprocessor
    )

    # ------------------------------------------------------
    # SAVE FILE
    # ------------------------------------------------------

    save_preprocessor(
        preprocessor
    )

    save_feature_columns(
        feature_columns
    )

    save_processed_dataset(
        processed_df
    )

    # ------------------------------------------------------
    # INFORMATION
    # ------------------------------------------------------

    preprocessing_information(
        processed_df,
        feature_columns
    )

    print("Numerical Features")

    for column in numerical_columns:

        print(f"  - {column}")

    print()

    print("Categorical Features")

    for column in categorical_columns:

        print(f"  - {column}")

    print()

    print("=" * 70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()


# ==========================================================
# RUN PROGRAM
# ==========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        logging.exception(
            error
        )

        print()

        print("=" * 70)

        print("PREPROCESSING FAILED")

        print("=" * 70)

        print(error)