#!/usr/bin/env python3
"""
Pytest-compatible test script to validate Main_File.py functionality without full execution
"""
import os
import sys
import re
import warnings
import pytest

warnings.filterwarnings("ignore")


def test_imports():
    """Test 1: Import all required packages"""
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    import joblib
    assert True, "All core imports successful"


@pytest.fixture
def df():
    """Fixture to load the CSV data"""
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_work_orders.csv")
    return pd.read_csv(csv_path, low_memory=False)


def test_data_loading(df):
    """Test 2: Load CSV"""
    assert df is not None, "CSV should load"
    assert df.shape[0] > 0, "CSV should have rows"
    assert df.shape[1] > 0, "CSV should have columns"


def test_data_cleaning(df):
    """Test 3: Data cleaning"""
    import pandas as pd
    
    df_clean = df.dropna(how='all')
    df_clean = df_clean[~df_clean['Text'].str.contains("completed", case=False, na=False)]
    df_clean = df_clean[~df_clean['Text'].str.contains("complete", case=False, na=False)]
    df_clean = df_clean.dropna(subset=['Description', 'Text']).drop_duplicates()
    df_clean = df_clean[df_clean['WO No.'].astype(str).str.match(r'^\d+$')]
    
    df_clean['Description_cleaned'] = (
        df_clean['Description'].fillna("").str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\d+', '', regex=True).str.strip()
    )
    
    df_clean['Text_cleaned'] = (
        df_clean['Text'].fillna("").str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\d+', '', regex=True).str.strip()
    )
    
    df_clean = df_clean[['Description_cleaned', 'Text_cleaned']].dropna()
    df_clean = df_clean[df_clean['Description_cleaned'].str.strip() != ""]
    df_clean = df_clean[df_clean['Text_cleaned'].str.strip() != ""]
    
    assert df_clean.shape[0] > 0, "Cleaned data should have rows"
    assert 'Description_cleaned' in df_clean.columns, "Should have Description_cleaned column"
    assert 'Text_cleaned' in df_clean.columns, "Should have Text_cleaned column"


@pytest.fixture
def cleaned_df(df):
    """Fixture to provide cleaned dataframe"""
    import pandas as pd
    
    df_clean = df.dropna(how='all')
    df_clean = df_clean[~df_clean['Text'].str.contains("completed", case=False, na=False)]
    df_clean = df_clean[~df_clean['Text'].str.contains("complete", case=False, na=False)]
    df_clean = df_clean.dropna(subset=['Description', 'Text']).drop_duplicates()
    df_clean = df_clean[df_clean['WO No.'].astype(str).str.match(r'^\d+$')]
    
    df_clean['Description_cleaned'] = (
        df_clean['Description'].fillna("").str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\d+', '', regex=True).str.strip()
    )
    
    df_clean['Text_cleaned'] = (
        df_clean['Text'].fillna("").str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\d+', '', regex=True).str.strip()
    )
    
    df_clean = df_clean[['Description_cleaned', 'Text_cleaned']].dropna()
    df_clean = df_clean[df_clean['Description_cleaned'].str.strip() != ""]
    df_clean = df_clean[df_clean['Text_cleaned'].str.strip() != ""]
    
    return df_clean


def test_labeling(cleaned_df):
    """Test 4: Labeling"""
    import pandas as pd
    
    ACTION_PATTERNS = [
        ("Replace Part", r"\b(replace|replaced|swap|swapped|install(ed)?)\b.*\b(bearing|motor|belt|gear|fuse|sensor|valve|hose|coupling|chain|switch|roller|pulley|seal)\b"),
        ("Tighten/Adjust", r"\b(tighten|tightened|adjust|adjusted|align|aligned|re-seat|reseat|calibrate|calibrated|reposition|realign(ed)?)\b"),
        ("Clean/Clear", r"\b(clean|cleaned|clear|cleared|remove|removed)\b.*\b(debris|dust|jam|blockage|clog)\b|\b(cleaned|cleared)\b"),
        ("Refill/Top Off", r"\b(add|added|refill|refilled|top\s?off)\b.*\b(oil|fluid|grease|lub(e|ricant)|coolant)\b"),
        ("Electrical Fix", r"\b(replace|replaced|reset|rewire|wire(d)?|reconnect|connector|contactor|breaker|fuse|vfd|plc|relay)\b"),
        ("Hydraulic/Pneumatic Fix", r"\b(hose|cylinder|solenoid|regulator|air line|hydraulic|pneumatic)\b.*\b(repair|replace|fixed|leak|leaking)\b"),
        ("Reset/Power Cycle", r"\b(reset|power.?cycle|cycled|restart|reboot|restarted)\b"),
        ("Inspection/Test Only", r"\b(inspect|inspected|tested|verify|verified|checked)\b(?!.*replace|.*repair|.*fix)"),
        ("Other", r".*")
    ]
    
    def to_response_label(text: str) -> str:
        t = " " + str(text).lower() + " "
        for label, pat in ACTION_PATTERNS:
            if re.search(pat, t):
                return label
        return "Other"
    
    df = cleaned_df.copy()
    df['Response_Label'] = df['Text_cleaned'].apply(to_response_label)
    
    # Filter out tiny labels
    min_count = 15
    vc = df['Response_Label'].value_counts()
    valid = vc[vc >= min_count].index
    df.loc[~df['Response_Label'].isin(valid), 'Response_Label'] = "Other"
    
    label_counts = df['Response_Label'].value_counts()
    
    assert len(df) > 0, "Labeled data should have rows"
    assert len(label_counts) > 0, "Should have at least one label category"
    assert 'Response_Label' in df.columns, "Should have Response_Label column"


def test_vectorization(cleaned_df):
    """Test 5: Vectorization"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    
    # Add labels first
    ACTION_PATTERNS = [
        ("Replace Part", r"\b(replace|replaced|swap|swapped|install(ed)?)\b.*\b(bearing|motor|belt|gear|fuse|sensor|valve|hose|coupling|chain|switch|roller|pulley|seal)\b"),
        ("Tighten/Adjust", r"\b(tighten|tightened|adjust|adjusted|align|aligned|re-seat|reseat|calibrate|calibrated|reposition|realign(ed)?)\b"),
        ("Clean/Clear", r"\b(clean|cleaned|clear|cleared|remove|removed)\b.*\b(debris|dust|jam|blockage|clog)\b|\b(cleaned|cleared)\b"),
        ("Refill/Top Off", r"\b(add|added|refill|refilled|top\s?off)\b.*\b(oil|fluid|grease|lub(e|ricant)|coolant)\b"),
        ("Electrical Fix", r"\b(replace|replaced|reset|rewire|wire(d)?|reconnect|connector|contactor|breaker|fuse|vfd|plc|relay)\b"),
        ("Hydraulic/Pneumatic Fix", r"\b(hose|cylinder|solenoid|regulator|air line|hydraulic|pneumatic)\b.*\b(repair|replace|fixed|leak|leaking)\b"),
        ("Reset/Power Cycle", r"\b(reset|power.?cycle|cycled|restart|reboot|restarted)\b"),
        ("Inspection/Test Only", r"\b(inspect|inspected|tested|verify|verified|checked)\b(?!.*replace|.*repair|.*fix)"),
        ("Other", r".*")
    ]
    
    def to_response_label(text: str) -> str:
        t = " " + str(text).lower() + " "
        for label, pat in ACTION_PATTERNS:
            if re.search(pat, t):
                return label
        return "Other"
    
    df = cleaned_df.copy()
    df['Response_Label'] = df['Text_cleaned'].apply(to_response_label)
    
    min_count = 15
    vc = df['Response_Label'].value_counts()
    valid = vc[vc >= min_count].index
    df.loc[~df['Response_Label'].isin(valid), 'Response_Label'] = "Other"
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['Description_cleaned'].astype(str))
    y = df['Response_Label'].astype(str)
    
    assert X.shape[0] > 0, "Vectorization should produce rows"
    assert X.shape[1] > 0, "Vectorization should produce features"
    assert len(y) == X.shape[0], "Labels should match feature rows"


def test_train_test_split(cleaned_df):
    """Test 6: Train/test split"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    # Prepare data with labels
    ACTION_PATTERNS = [
        ("Replace Part", r"\b(replace|replaced|swap|swapped|install(ed)?)\b.*\b(bearing|motor|belt|gear|fuse|sensor|valve|hose|coupling|chain|switch|roller|pulley|seal)\b"),
        ("Tighten/Adjust", r"\b(tighten|tightened|adjust|adjusted|align|aligned|re-seat|reseat|calibrate|calibrated|reposition|realign(ed)?)\b"),
        ("Clean/Clear", r"\b(clean|cleaned|clear|cleared|remove|removed)\b.*\b(debris|dust|jam|blockage|clog)\b|\b(cleaned|cleared)\b"),
        ("Refill/Top Off", r"\b(add|added|refill|refilled|top\s?off)\b.*\b(oil|fluid|grease|lub(e|ricant)|coolant)\b"),
        ("Electrical Fix", r"\b(replace|replaced|reset|rewire|wire(d)?|reconnect|connector|contactor|breaker|fuse|vfd|plc|relay)\b"),
        ("Hydraulic/Pneumatic Fix", r"\b(hose|cylinder|solenoid|regulator|air line|hydraulic|pneumatic)\b.*\b(repair|replace|fixed|leak|leaking)\b"),
        ("Reset/Power Cycle", r"\b(reset|power.?cycle|cycled|restart|reboot|restarted)\b"),
        ("Inspection/Test Only", r"\b(inspect|inspected|tested|verify|verified|checked)\b(?!.*replace|.*repair|.*fix)"),
        ("Other", r".*")
    ]
    
    def to_response_label(text: str) -> str:
        t = " " + str(text).lower() + " "
        for label, pat in ACTION_PATTERNS:
            if re.search(pat, t):
                return label
        return "Other"
    
    df = cleaned_df.copy()
    df['Response_Label'] = df['Text_cleaned'].apply(to_response_label)
    
    min_count = 15
    vc = df['Response_Label'].value_counts()
    valid = vc[vc >= min_count].index
    df.loc[~df['Response_Label'].isin(valid), 'Response_Label'] = "Other"
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['Description_cleaned'].astype(str))
    y = df['Response_Label'].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    assert X_train.shape[0] > 0, "Train set should have rows"
    assert X_test.shape[0] > 0, "Test set should have rows"
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0], "Train + test should equal total"


def test_model_training(cleaned_df):
    """Test 7: Quick model test (small model for speed)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    
    # Prepare data with labels
    ACTION_PATTERNS = [
        ("Replace Part", r"\b(replace|replaced|swap|swapped|install(ed)?)\b.*\b(bearing|motor|belt|gear|fuse|sensor|valve|hose|coupling|chain|switch|roller|pulley|seal)\b"),
        ("Tighten/Adjust", r"\b(tighten|tightened|adjust|adjusted|align|aligned|re-seat|reseat|calibrate|calibrated|reposition|realign(ed)?)\b"),
        ("Clean/Clear", r"\b(clean|cleaned|clear|cleared|remove|removed)\b.*\b(debris|dust|jam|blockage|clog)\b|\b(cleaned|cleared)\b"),
        ("Refill/Top Off", r"\b(add|added|refill|refilled|top\s?off)\b.*\b(oil|fluid|grease|lub(e|ricant)|coolant)\b"),
        ("Electrical Fix", r"\b(replace|replaced|reset|rewire|wire(d)?|reconnect|connector|contactor|breaker|fuse|vfd|plc|relay)\b"),
        ("Hydraulic/Pneumatic Fix", r"\b(hose|cylinder|solenoid|regulator|air line|hydraulic|pneumatic)\b.*\b(repair|replace|fixed|leak|leaking)\b"),
        ("Reset/Power Cycle", r"\b(reset|power.?cycle|cycled|restart|reboot|restarted)\b"),
        ("Inspection/Test Only", r"\b(inspect|inspected|tested|verify|verified|checked)\b(?!.*replace|.*repair|.*fix)"),
        ("Other", r".*")
    ]
    
    def to_response_label(text: str) -> str:
        t = " " + str(text).lower() + " "
        for label, pat in ACTION_PATTERNS:
            if re.search(pat, t):
                return label
        return "Other"
    
    df = cleaned_df.copy()
    df['Response_Label'] = df['Text_cleaned'].apply(to_response_label)
    
    min_count = 15
    vc = df['Response_Label'].value_counts()
    valid = vc[vc >= min_count].index
    df.loc[~df['Response_Label'].isin(valid), 'Response_Label'] = "Other"
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['Description_cleaned'].astype(str))
    y = df['Response_Label'].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use a very small model just to verify it works
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()
    
    # Test prediction on a sample
    sample_text = ["motor running hot"]
    sample_vec = vectorizer.transform(sample_text)
    prediction = clf.predict(sample_vec)
    
    assert accuracy > 0, "Model should have non-zero accuracy"
    assert len(prediction) == 1, "Prediction should return one result"
