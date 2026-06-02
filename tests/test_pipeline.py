import pytest
import sys
import os

# Ensure src is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import split_into_clauses
from src.labeler import contains_keyword
from src.aggregator import calculate_tier

def test_split_into_clauses():
    text = "The room was great, but the wifi was terrible! Would not recommend."
    clauses = split_into_clauses(text, min_words=3)
    assert len(clauses) >= 2
    
    # Ensure it's lowercased by the user before passing or handled properly
    # In our pipeline, we assume pre-tokenized or we just split.
    # We test the basic splitting functionality.
    assert any("room was great" in c.lower() for c in clauses)

def test_contains_keyword():
    # Word boundary checks
    assert contains_keyword("the wifi is bad", "wifi") == True
    assert contains_keyword("this is a bus", "busy") == False
    assert contains_keyword("very noisy outside", "noise") == False # strict match
    assert contains_keyword("noise outside", "noise") == True

def test_calculate_tier():
    # Elite threshold >= 0.8
    tier, score = calculate_tier(pos_count=9, neg_count=1)
    assert tier == 'Elite'
    assert score == 0.9
    
    # Fail threshold < 0.4
    tier, score = calculate_tier(pos_count=2, neg_count=8)
    assert tier == 'Fail'
    assert score == 0.2
    
    # Uncertain < 3 total mentions
    tier, score = calculate_tier(pos_count=1, neg_count=1)
    assert tier == 'Uncertain'
    assert score is None
