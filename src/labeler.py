import re
import pandas as pd
from transformers import pipeline
import torch

# --- Keywords Configuration ---
RULES = {
    'cleanliness': {
        'triggers': ['clean', 'dirty', 'spotless', 'filthy', 'hygiene', 'hygienic',
                     'dust', 'dusty', 'stain', 'stained', 'smell', 'smelly', 'odor',
                     'odour', 'gross', 'immaculate', 'sanitiz', 'tidy', 'messy',
                     'mold', 'mould', 'pest', 'cockroach'],
    },
    'staff_service': {
        'triggers': ['staff', 'service', 'front desk', 'concierge', 'receptionist',
                     'housekeep', 'bellm', 'valet', 'employee', 'manager',
                     'check in', 'check-in', 'checkout', 'check out', 'reception'],
    },
    'wifi_quality': {
        'triggers': ['wifi', 'wi-fi', 'internet', 'wireless', 'bandwidth', 'broadband'],
    },
    'noise_level': {
        'triggers': ['noise', 'noisy', 'loud', 'quiet', 'soundproof', 'hear', 'heard',
                     'banging', 'traffic noise', 'street noise', 'peaceful', 'disturb',
                     'thin wall', 'construction'],
    },
    'location': {
        'triggers': ['location', 'located', 'walking distance', 'distance from', 'nearby',
                     'transit', 'transport', 'central location', 'close to', 'far from',
                     'neighborhood', 'downtown', 'convenient location',
                     'metro', 'train station', 'bus stop', 'airport', 'attraction'],
    }
}

def contains_keyword(text, keyword):
    """Word-boundary aware keyword match."""
    pattern = r'\b' + re.escape(keyword) + r'\b'
    return bool(re.search(pattern, text, re.IGNORECASE))

class HybridLabeler:
    def __init__(self):
        print("Loading DistilBERT sentiment model for labeler...")
        # Automatically use GPU if available
        device = 0 if torch.cuda.is_available() else -1
        self.sentiment_model = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=1,
            device=device
        )
        self.rules = RULES
        self.attrs = list(RULES.keys())
        print("Labeler ready!")

    def label_clauses(self, clause_df, batch_size=64):
        """
        Applies keyword triggers and DistilBERT sentiment to label clauses.
        """
        print(f"Applying keyword triggers to {len(clause_df):,} clauses...")
        for attr in self.attrs:
            triggers = self.rules[attr]['triggers']
            clause_df[f'trigger_{attr}'] = clause_df['clause'].apply(
                lambda c: any(contains_keyword(c.lower(), kw) for kw in triggers)
            )
            
        # Get unique triggered clauses
        triggered_mask = clause_df[[f'trigger_{a}' for a in self.attrs]].any(axis=1)
        triggered_clauses = clause_df[triggered_mask]['clause'].unique().tolist()
        
        print(f"Found {len(triggered_clauses):,} unique triggered clauses. Running sentiment model...")
        
        # Batch process sentiment
        sentiment_cache = {}
        for i in range(0, len(triggered_clauses), batch_size):
            batch = triggered_clauses[i:i+batch_size]
            batch_truncated = [c[:512] for c in batch]
            try:
                results = self.sentiment_model(batch_truncated, batch_size=batch_size)
                for clause, result in zip(batch, results):
                    label = result[0]['label'].lower()
                    sentiment_cache[clause] = label
            except Exception as e:
                print(f"Error on batch: {e}")
                for clause in batch:
                    sentiment_cache[clause] = 'not_mentioned'
                    
        print("Applying cached sentiments to full dataset...")
        
        # Apply labels
        for attr in self.attrs:
            def get_label(row):
                if not row[f'trigger_{attr}']:
                    return 'not_mentioned'
                return sentiment_cache.get(row['clause'], 'not_mentioned')
                
            clause_df[attr] = clause_df.apply(get_label, axis=1)
            
        # Clean up trigger columns
        cols_to_drop = [f'trigger_{a}' for a in self.attrs]
        return clause_df.drop(columns=cols_to_drop)

if __name__ == "__main__":
    from preprocess import load_and_preprocess
    _, clause_df = load_and_preprocess("../tripadvisor_hotel_reviews.csv")
    
    labeler = HybridLabeler()
    labeled_df = labeler.label_clauses(clause_df)
    
    import os
    os.makedirs("../data", exist_ok=True)
    labeled_df.to_csv("../data/labeled_clauses.csv", index=False)
    print("Exported labels to data/labeled_clauses.csv")
