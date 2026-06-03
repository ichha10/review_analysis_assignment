import pandas as pd

import re
import pickle
import numpy as np
import os

def build_hotel_profiles(df, clause_df):
    """
    Extract real hotel names using a dictionary heuristic.
    Fallback to 'Unknown Hotel' if no brand is mentioned.
    """
    chains = ['hilton', 'kimpton', 'monaco', 'marriott', 'hyatt', 'sheraton', 'westin', 'holiday inn', 'best western', 'ramada', 'radisson', 'wyndham', 'ritz', 'four seasons', 'w hotel']
    pattern = re.compile(r'\b(' + '|'.join(chains) + r')\b', re.IGNORECASE)
    
    def extract_hotel(text):
        match = pattern.search(str(text))
        if match:
            return match.group(1).title() + " Hotel"
        return "Unknown Hotel"
        
    df['hotel_id'] = df['Review'].apply(extract_hotel)
    
    # Map back to clauses using review_idx
    hotel_map = df['hotel_id'].to_dict() # index -> hotel_id
    clause_df['hotel_id'] = clause_df['review_idx'].map(hotel_map)
    
    return clause_df

def calculate_tiers(hotel_df, vectorizer, ml_models):
    """
    Calculates the signal score for each attribute and maps it to a tier.
    Sorts evidence by model confidence.
    """
    tiers = {}
    attrs = ['cleanliness', 'staff_service', 'wifi_quality', 'noise_level', 'location']
    
    for attr in attrs:
        attr_clauses = hotel_df[hotel_df[attr] != 'not_mentioned']
        
        pos_count = (attr_clauses[attr] == 'positive').sum()
        neg_count = (attr_clauses[attr] == 'negative').sum()
        total_mentions = pos_count + neg_count
        
        if total_mentions < 3:
            tiers[attr] = {'tier': 'Uncertain', 'score': None, 'evidence': []}
            continue
            
        score = pos_count / total_mentions
        
        if score >= 0.80:
            tier = 'Elite'
        elif score >= 0.60:
            tier = 'Superior'
        elif score >= 0.40:
            tier = 'Premium'
        else:
            tier = 'Fail'
            
        # Sort evidence by confidence
        if vectorizer and attr in ml_models:
            clf = ml_models[attr]
            clauses = attr_clauses['clause'].tolist()
            if clauses:
                X_subset = vectorizer.transform(clauses)
                probs = clf.predict_proba(X_subset)
                classes = list(clf.classes_)
                if 'positive' in classes:
                    pos_idx = classes.index('positive')
                    neg_idx = classes.index('negative')
                    scores_for_sort = probs[:, neg_idx] if tier == 'Fail' else probs[:, pos_idx]
                    sorted_indices = np.argsort(scores_for_sort)[::-1]
                    evidence = [clauses[i] for i in sorted_indices[:3]]
                else:
                    evidence = attr_clauses['clause'].head(3).tolist()
            else:
                evidence = []
        else:
            evidence = attr_clauses['clause'].head(3).tolist()
        
        tiers[attr] = {
            'tier': tier,
            'score': round(score, 2),
            'evidence': evidence
        }
        
    return tiers

def run_aggregation(labeled_csv_path):
    print(f"Loading labels from {labeled_csv_path}...")
    clause_df = pd.read_csv(labeled_csv_path)
    
    # Load raw reviews to extract hotel names
    raw_df = pd.read_csv('tripadvisor_hotel_reviews.csv')
    
    clause_df = build_hotel_profiles(raw_df, clause_df)
    
    print("Loading ML models for evidence sorting...")
    vectorizer, ml_models = None, {}
    if os.path.exists('models/attribute_classifier/tfidf_vectorizer.pkl'):
        with open('models/attribute_classifier/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        for attr in ['cleanliness', 'staff_service', 'wifi_quality', 'noise_level', 'location']:
            try:
                with open(f'models/attribute_classifier/{attr}_model.pkl', 'rb') as f:
                    ml_models[attr] = pickle.load(f)
            except Exception:
                pass

    print("Calculating tiers per hotel...")
    hotel_profiles = {}
    
    for hotel_id, group in clause_df.groupby('hotel_id'):
        hotel_profiles[hotel_id] = calculate_tiers(group, vectorizer, ml_models)
        
    return hotel_profiles

if __name__ == "__main__":
    profiles = run_aggregation("data/labeled_clauses.csv")
    print(f"Aggregated tiers for {len(profiles)} pseudo-hotels.")
    # Show example for Hilton Hotel
    print("\nExample Profile (Hilton Hotel):")
    if "Hilton Hotel" in profiles:
        for attr, data in profiles["Hilton Hotel"].items():
            print(f"  {attr.upper()}: {data['tier']} (Score: {data['score']})")
            if data['evidence']:
                print(f"    Evidence: '{data['evidence'][0]}'")
    else:
        print("Hilton Hotel not found in extraction.")
