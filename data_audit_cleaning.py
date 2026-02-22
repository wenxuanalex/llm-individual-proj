import pandas as pd
from datasets import load_dataset
import random
import re
import os

# Set seed for reproducibility
random.seed(42)

def load_and_audit(dataset_name="medalpaca/medical_meadow_wikidoc"):
    """Phase 1: Load and sample dataset for auditing."""
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    df = pd.DataFrame(dataset['train'])
    
    print(f"Initial row count: {len(df)}")
    
    # Snapshot: 50 random rows
    sample_indices = random.sample(range(len(df)), 50)
    snapshot = df.iloc[sample_indices]
    
    return df, snapshot

def identify_noise(row):
    """Categorize noise patterns for reporting."""
    instruction = str(row['instruction']).strip().lower()
    output = str(row['output']).strip().lower()
    
    # Category 1: Empty/Near-empty
    if len(instruction) < 3 or len(output) < 3:
        return "Empty/Short Field"
    
    # Category 2: Meta-talk
    meta_patterns = [
        "as an ai language model",
        "i'm sorry, but i cannot",
        "the provided text does not contain",
        "the text provided is",
        "as an ai",
        "i do not have access to"
    ]
    if any(pattern in output for pattern in meta_patterns):
        return "Meta-talk"
    
    # Category 3: Clinical Utility (Short output)
    if len(output) < 20:
        return "Low Clinical Utility"
    
    # Category 4: Truncated/Nonsensical
    # Check if ends with punctuation
    if len(output) > 0 and output[-1] not in ['.', '?', '!', '"', ')']:
        # This is subtle, but often indicates truncation in synthetic sets
        # We only flag if it's also relatively long (to avoid short valid points)
        if len(output) > 100:
            return "Possible Truncation"

    return None

def clean_dataset(df):
    """Phase 2: Apply cleaning logic and track removals."""
    print("\nStarting cleaning process...")
    
    # Initial stats
    initial_count = len(df)
    
    # Identify noise for reporting
    df['noise_category'] = df.apply(identify_noise, axis=1)
    
    # Save a "Before" copy for the examples
    noisy_rows = df[df['noise_category'].notnull()].head(10)
    
    # Filter
    cleaned_df = df[df['noise_category'].isnull()].copy()
    
    # Breakdown statistics
    breakdown = df['noise_category'].value_counts()
    
    print("Cleaning complete.")
    return cleaned_df, breakdown, noisy_rows

def run_pipeline():
    # 1. Audit
    df, snapshot = load_and_audit()
    
    # 2. Clean
    cleaned_df, breakdown, noisy_samples = clean_dataset(df)
    
    # 3. Save
    cleaned_df.drop(columns=['noise_category']).to_json("cleaned_medical_meadow.jsonl", orient='records', lines=True)
    print(f"Cleaned dataset saved to: cleaned_medical_meadow.jsonl")
    
    # 4. Report
    print("\n" + "="*40)
    print("      DATA CLEANING REPORT")
    print("="*40)
    print(f"Total rows before cleaning: {len(df)}")
    print(f"Total rows after cleaning:  {len(cleaned_df)}")
    print(f"Rows removed:               {len(df) - len(cleaned_df)}")
    print("\nRemoval Breakdown by Category:")
    print(breakdown)
    
    print("\n--- Examples of Removed Rows (Before/After) ---")
    for idx, row in noisy_samples.head(5).iterrows():
        print(f"\n[Category: {row['noise_category']}]")
        print(f"Instruction: {row['instruction'][:80]}...")
        print(f"Output (Truncated): {row['output'][:100]}...")
        print("-" * 20)

if __name__ == "__main__":
    run_pipeline()
