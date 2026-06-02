import re
import pandas as pd

def split_into_clauses(text, min_words=5):
    """
    Split pre-tokenized, lowercased reviews into clauses.
    Reviews lack sentence-ending punctuation, so we split on
    commas and any remaining punctuation (. ! ?).
    Filter out very short fragments (< min_words).
    """
    text = str(text).strip()
    # Split on comma, period, exclamation, question mark
    clauses = re.split(r'[,\.!\?]+', text)
    # Clean and filter
    clauses = [c.strip() for c in clauses if len(c.strip().split()) >= min_words]
    return clauses

def load_and_preprocess(file_path):
    """
    Load raw TripAdvisor reviews and split them into clauses.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} reviews.")
    
    print("Splitting reviews into clauses...")
    df['clauses'] = df['Review'].apply(split_into_clauses)
    
    # Flatten into a clause-level dataframe
    all_rows = []
    for idx, row in df.iterrows():
        for clause in row['clauses']:
            all_rows.append({
                'review_idx': idx, 
                'rating': row['Rating'], 
                'clause': clause
            })
            
    clause_df = pd.DataFrame(all_rows)
    print(f"Total clauses generated: {len(clause_df):,}")
    
    return df, clause_df

if __name__ == "__main__":
    df, clause_df = load_and_preprocess("../tripadvisor_hotel_reviews.csv")
    print("Preprocessing complete!")
