# Hotel Curator: Machine Learning Pipeline Architecture

## 1. Executive Summary
The goal of this project was to design and prototype an end-to-end Machine Learning pipeline capable of extracting structured quality signals from unstructured hotel reviews. These signals are mapped to discrete, defensible Tier Ratings (Elite, Superior, Premium, Fail) for five specific attributes (Cleanliness, Staff Service, Wi-Fi Quality, Noise Level, and Location) to power a curated corporate travel booking product.

To achieve maximum throughput and scalability for production inference, the architecture leverages a **Weak Supervision** approach. A heavy Pre-Trained Language Model (DistilBERT) was used purely offline to generate historical labels, which were then used to train lightning-fast, highly calibrated Linear Support Vector Machines (LinearSVC) for the live API.

## 2. Exploratory Data Analysis (EDA) & System Design
During the initial EDA phase, three significant challenges were identified that drove the core architectural decisions:

1. **Conflicting Sentiments in Raw Text:** Users frequently mixed sentiments within a single review (e.g., *"The staff was amazing but the wifi was terrible"*). Feeding full reviews into a model caused sentiment cancellation. 
   * *Solution:* A preprocessor was built to split raw text into independent clauses using punctuation boundaries before any inference occurs.
2. **Class Imbalance:** The review dataset is heavily skewed toward positive 5-star ratings. Left untreated, a model would optimize for the majority class and ignore complaints.
3. **Missing Entity Identifiers:** The dataset lacked Hotel IDs. Furthermore, because the text was entirely lowercased and lacked proper capitalization, standard Named Entity Recognition (NER) models failed to accurately extract hotel names without massive hallucination.
   * *Solution:* I implemented a **Dictionary-Based Heuristic Extraction** strategy. Assuming a corporate travel product already possesses an internal database of valid hotel chains, the pipeline scans text against a fast regex dictionary of global brands (Hilton, Marriott, etc.). Unnamed reviews fall back safely to an "Unknown Hotel" bucket, ensuring zero hallucinations and O(N) execution speed.

## 3. Feature Engineering & Weak Supervision
Because the dataset lacked human-annotated sentiment labels for the 5 attributes, I built a Weak Supervision pipeline:
* **The Regex Router:** Clauses were passed through a dictionary of attribute-specific keywords. Irrelevant clauses were immediately discarded to save compute power.
* **The Teacher Model (DistilBERT):** Relevant clauses were passed to `distilbert-base-uncased-finetuned-sst-2-english` offline to generate the baseline Positive/Negative labels.

### TF-IDF & Trigrams
For the live production models, deep learning embeddings were bypassed in favor of a highly optimized `TfidfVectorizer`. 
* **Trigrams:** `ngram_range=(1, 3)` was utilized to ensure the model did not lose context on negations (e.g., distinguishing between *"clean"* and *"not clean"*).
* **Dimensionality Cap:** The feature space was capped at 25,000 features. TF-IDF naturally penalized useless stop-words (like *"the"*) while boosting the mathematical signal of rare sentiment words (like *"filthy"*), providing a rich, sparse matrix for the algorithms.

## 4. Modeling Choices & Benchmarking
An automated benchmark was constructed to evaluate multiple algorithms (Logistic Regression, Random Forest, XGBoost, and a Neural Network). 

**The Winner: Linear Support Vector Machine (LinearSVC)**
LinearSVC proved to be the most accurate model. Because text data transformed by TF-IDF is highly dimensional and sparse, SVMs are mathematically proven to draw sharper decision boundaries than tree-based models. 

**Calibration and Imbalance Handling**
* To solve the class imbalance identified during EDA, `class_weight='balanced'` was applied to heavily penalize the model for missing minority-class negative complaints.
* Because standard SVMs do not output probability scores, the LinearSVC was wrapped in `CalibratedClassifierCV`. This mathematically forced the model to output true, trustworthy probability percentages (essential for downstream sorting).

## 5. Aggregation, Tiering & Evidence Surfacing
The final phase aggregates the ML predictions into actionable business logic.

**1. The Low-Evidence Safety Check:**
To prevent False Positives driven by low sample sizes, the aggregator enforces a strict rule: If a hotel has fewer than 3 total mentions for an attribute, it is immediately assigned an **`Uncertain`** tier. 

**2. The Signal Score Formula:**
For valid attributes, a Signal Score is calculated: `Positive Mentions / (Positive Mentions + Negative Mentions)`.

**3. Defensible Tier Mapping:**
* **Elite (>= 0.80):** Universally excellent (equivalent to 4/5 stars).
* **Superior (0.60 - 0.79):** Mostly positive, minor flaws.
* **Premium (0.40 - 0.59):** Highly mixed signals.
* **Fail (< 0.40):** More complaints than praises.

**4. ML-Sorted Evidence:**
When surfacing evidence for the frontend UI, the API dynamically passes the sentences back through the Calibrated ML models using `.predict_proba()`. If a hotel receives a **Fail** tier, it sorts and surfaces the top 3 sentences with the highest *Negative* confidence. For **Elite** tiers, it surfaces the highest *Positive* confidence sentences.

## 6. Future Architecture Considerations
With additional time, the following upgrades would be implemented:
* **Multi-Label Classification:** Upgrading the Regex trigger router into a multi-label model so that a single clause could trigger multiple attributes simultaneously in the event of keyword collisions.
* **Active Learning Loop:** Implementing an entropy-based sampler to surface the clauses the model is mathematically most confused about. A human-in-the-loop (via a Streamlit app) would manually label these edge cases to iteratively retrain and fine-tune the model, replacing the Weak Labels with Ground Truth over time.
