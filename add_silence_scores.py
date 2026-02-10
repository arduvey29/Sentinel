import pandas as pd

def calculate_silence_score(row):
    """
    Calculate silence score (0-100) based on response status and days in system.
    Logic:
    - NO_RESPONSE: High silence (scales with time, max 100)
    - REJECTED: Very high silence (complaint was actively dismissed)
    - RESPONDED: Medium silence (responded but not resolved)
    - RESOLVED: Low silence (inverse of resolution speed)
    """
    days = row['days_in_system']
    status = row['response_status']
    
    if status == "NO_RESPONSE":
        # Never responded = maximum silence
        return 100 * min(days, 365) / 365
    
    elif status == "REJECTED":
        # Actively dismissed = even worse than silence
        # Starts at 70, scales to 100
        return 70 + 30 * min(days, 365) / 365
    
    elif status == "RESPONDED":
        # Responded but unresolved = moderate silence
        return 60 * min(days, 365) / 365
    
    else:  # RESOLVED
        # Resolved = minimal silence
        # Lower score if resolved quickly
        return max(0, 30 - (days / 30))

if __name__ == "__main__":
    # Load raw data
    print("Loading complaints_raw.csv...")
    df = pd.read_csv('data/complaints_raw.csv')
    
    print(f"✓ Loaded {len(df)} complaints")
    
    # Calculate silence scores
    print("Calculating silence scores...")
    df['silence_score'] = df.apply(calculate_silence_score, axis=1)
    
    # Round to 2 decimal places
    df['silence_score'] = df['silence_score'].round(2)
    
    # Validation
    print("\n--- VALIDATION ---")
    print(f"Min score: {df['silence_score'].min():.2f}")
    print(f"Max score: {df['silence_score'].max():.2f}")
    print(f"Mean score: {df['silence_score'].mean():.2f}")
    
    # Check for any invalid values
    assert df['silence_score'].min() >= 0, "ERROR: Found negative scores"
    assert df['silence_score'].max() <= 100, "ERROR: Found scores > 100"
    assert df['silence_score'].isna().sum() == 0, "ERROR: Found NaN scores"
    
    print("✓ All scores valid (0-100 range)")
    
    # Count highly silenced
    highly_silenced = (df['silence_score'] > 70).sum()
    print(f"✓ Highly silenced (>70): {highly_silenced} ({highly_silenced/len(df)*100:.1f}%)")
    
    # Save cleaned data
    output_path = 'data/complaints_clean.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n Saved to: {output_path}")
    print(f" Total rows: {len(df)}")
    print(f" Columns: {list(df.columns)}")
    
    # Show sample
    print("\n--- SAMPLE DATA ---")
    print(df[['id', 'text', 'response_status', 'days_in_system', 'silence_score']].head(10))
    
    print("\n SILENCE SCORES ADDED SUCCESSFULLY!")
