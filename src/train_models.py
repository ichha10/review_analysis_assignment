import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os

def train_and_save_models():
    print("Loading Labeled Data...")
    df = pd.read_csv('data/labeled_clauses.csv')
    print(f"Loaded {len(df):,} labeled clauses!\n")
    
    attrs = ['cleanliness', 'staff_service', 'wifi_quality', 'noise_level', 'location']
    
    print("Training fast ML models...\n")
    print("Using TF-IDF with Trigrams (max 25,000 features)...")
    vectorizer = TfidfVectorizer(max_features=25000, stop_words='english', ngram_range=(1, 3))
    X = vectorizer.fit_transform(df['clause'].fillna(''))
    
    models = {}
    
    for attr in attrs:
        print(f"Training model for: {attr.upper()}...")
        mask = df[attr] != 'not_mentioned'
        y = df.loc[mask, attr]
        X_subset = X[mask.values]
        
        if len(y) < 10:
            print(f"  Skipping {attr} - not enough data.")
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)
        
        svm = LinearSVC(max_iter=2000, class_weight='balanced')
        clf = CalibratedClassifierCV(svm)
        clf.fit(X_train, y_train)
        
        models[attr] = clf
        score = clf.score(X_test, y_test)
        print(f"  Accuracy: {score*100:.1f}%\n")
        
    os.makedirs('models/attribute_classifier', exist_ok=True)
    
    with open('models/attribute_classifier/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    for attr, clf in models.items():
        with open(f'models/attribute_classifier/{attr}_model.pkl', 'wb') as f:
            pickle.dump(clf, f)
            
    print("Models saved to models/attribute_classifier/ ✅")

if __name__ == "__main__":
    train_and_save_models()
