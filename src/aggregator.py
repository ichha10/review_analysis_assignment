import pandas as pd

def build_hotel_profiles(df, clause_df):
    """
    Since the dataset lacks hotel IDs, we group by rating and semantic clusters,
    or simply treat it as a massive corpus-level evaluation for the assignment.
    For simplicity in this pipeline, we will generate a pseudo-hotel ID 
    based on the review index (treating every 10 reviews as a 'hotel' cluster 
    just to demonstrate the aggregation logic).
    """
    # Assign a pseudo hotel_id (1 hotel = 10 reviews)
    clause_df['hotel_id'] = clause_df['review_idx'] // 10
    return clause_df

def calculate_tiers(hotel_df):
    """
    Calculates the signal score for each attribute and maps it to a tier.
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
            
        # Get top 3 sentences as evidence (for simplicity, we grab the first 3 relevant)
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
    
    clause_df = build_hotel_profiles(None, clause_df)
    
    print("Calculating tiers per hotel...")
    hotel_profiles = {}
    
    for hotel_id, group in clause_df.groupby('hotel_id'):
        hotel_profiles[hotel_id] = calculate_tiers(group)
        
    return hotel_profiles

if __name__ == "__main__":
    profiles = run_aggregation("../data/labeled_clauses.csv")
    print(f"Aggregated tiers for {len(profiles)} pseudo-hotels.")
    # Show example for hotel 0
    print("\nExample Profile (Hotel 0):")
    for attr, data in profiles[0].items():
        print(f"  {attr.upper()}: {data['tier']} (Score: {data['score']})")
        if data['evidence']:
            print(f"    Evidence: '{data['evidence'][0]}'")
