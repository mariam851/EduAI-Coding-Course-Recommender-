# src/preprocessing.py
import pandas as pd
import os

def load_raw_data(raw_csv_path):
    """Load raw CSV into DataFrame"""
    df = pd.read_csv(raw_csv_path)
    return df

def preprocess_sequences(df, min_interactions=5):
    """Filter students and sort by timestamp"""
    df = df.sort_values(["subject_id", "timestamp"])
    student_counts = df.groupby("subject_id")["item_id"].count()
    valid_students = student_counts[student_counts >= min_interactions].index
    df_filtered = df[df["subject_id"].isin(valid_students)]
    return df_filtered

def save_processed(df, processed_path):
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")
