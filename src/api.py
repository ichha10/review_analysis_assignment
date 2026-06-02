from fastapi import FastAPI, HTTPException
import pandas as pd
import json
import os
import sys

# Ensure we can import from src when running from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.aggregator import run_aggregation

app = FastAPI(title="Hotel Curator API", description="Surfaces hotel tiers and evidence.")

# Load data at startup
print("Starting API and precomputing hotel tiers...")
try:
    # This assumes we run uvicorn from the project root
    hotel_profiles = run_aggregation("data/labeled_clauses.csv")
    print("Hotel tiers loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load hotel profiles. {e}")
    hotel_profiles = {}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Hotel Curator API", "total_hotels": len(hotel_profiles)}

@app.get("/hotels")
def get_all_hotel_ids():
    """Return a list of all available hotel IDs."""
    return {"hotel_ids": list(hotel_profiles.keys())}

@app.get("/hotels/{hotel_id}/attributes")
def get_hotel_attributes(hotel_id: int):
    """
    Returns the Tier (Elite/Superior/Premium/Fail/Uncertain), Score, 
    and Top 3 pieces of Evidence for the requested Hotel ID.
    """
    if hotel_id not in hotel_profiles:
        raise HTTPException(status_code=404, detail="Hotel not found")
        
    return {
        "hotel_id": hotel_id,
        "attributes": hotel_profiles[hotel_id]
    }
