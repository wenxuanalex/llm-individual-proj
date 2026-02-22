import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def prepare_splits(input_file="cleaned_medical_meadow.jsonl"):
    """Load cleaned data, perform 90/10 split, and analysis."""
    print(f"Loading cleaned dataset: {input_file}...")
    df = pd.read_json(input_file, orient='records', lines=True)
    
    print(f"Total sanitized rows: {len(df)}")
    
    # 90/10 Split with fixed seed
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
    
    print(f"\nSplit Results:")
    print(f"Training set:   {len(train_df)} rows")
    print(f"Evaluation set: {len(eval_df)} rows")
    
    # Save splits
    train_df.to_json("train_split.jsonl", orient='records', lines=True)
    eval_df.to_json("eval_split.jsonl", orient='records', lines=True)
    print("Files saved: train_split.jsonl, eval_split.jsonl")
    
    # Analysis: Word count on output field
    print("\nSequence Analysis (Output Field - Training Set):")
    output_word_counts = train_df['output'].apply(lambda x: len(str(x).split()))
    
    stats = {
        "Average Word Count": output_word_counts.mean(),
        "Median Word Count":  output_word_counts.median(),
        "Max Word Count":     output_word_counts.max(),
        "95th Percentile":    output_word_counts.quantile(0.95),
        "99th Percentile":    output_word_counts.quantile(0.99)
    }
    
    for label, val in stats.items():
        print(f"{label:20}: {val:.2f}")
        
    return stats

if __name__ == "__main__":
    # Check if sklearn is installed, if not install it
    try:
        import sklearn
    except ImportError:
        print("Installing scikit-learn for splitting...")
        os.system("pip install scikit-learn")
        
    prepare_splits()
