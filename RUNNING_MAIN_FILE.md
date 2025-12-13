# Running Main_File.py

This document explains how to run the debugged Main_File.py script.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn sentence-transformers umap-learn joblib tqdm imbalanced-learn xgboost fuzzywuzzy python-Levenshtein

# Run the script
cd notebooks
python Main_File.py
```

Or use the wrapper script:
```bash
python run_main_file.py
```

## What Was Fixed

The original Main_File.py was a Jupyter notebook converted to Python format. Several issues were fixed:

### 1. Syntax Errors
- **Fixed unterminated string literal** on line 1154 - changed backslash continuation to triple-quoted string
- Script now compiles without syntax errors

### 2. Jupyter-Specific Code
- **Replaced `get_ipython()` calls** with `subprocess.check_call()` for package installation
- **Replaced `display()` calls** with `print()` for output
- **Added matplotlib backend** configuration for headless environments

### 3. File Paths
- **Fixed CSV path** to use relative path from repository root instead of `~/Downloads`
- Path now works correctly: `os.path.dirname(os.path.dirname(__file__))/cleaned_work_orders.csv`

### 4. Variable Dependencies
- **Added missing `SAVE_DIR` definition** for artifact storage
- **Added early definition of `X_train_text`, `y_train_text`** variables to fix forward references
- **Added variable existence checks** where needed

### 5. Data Flow
- **Ensured train/test splits** are created before model training sections
- Data cleaning → Labeling → Vectorization → Train/Test Split → Model Training

## Script Structure

The Main_File.py script follows this workflow:

1. **Data Management** (lines 1-186)
   - Install dependencies
   - Import libraries
   - Load and clean CSV data
   - Result: ~49,000 cleaned maintenance records

2. **System Labeling** (lines 188-286)
   - Apply pattern-based action labeling
   - Categories: Replace Part, Tighten/Adjust, Clean/Clear, etc.
   - Result: 9 distinct maintenance action labels

3. **Dimensionality Reduction** (lines 287-479)
   - PCA, t-SNE, UMAP visualizations
   - Save artifacts for analysis

4. **Train/Test Split** (lines 508-557)
   - TF-IDF vectorization
   - SMOTE for class balancing
   - 80/20 train/test split

5. **Model Training** (lines 576-1101)
   - Naive Bayes
   - Random Forest
   - Decision Tree
   - XGBoost
   - MLP Neural Network
   - Cross-validation and evaluation

6. **Narrative Generation** (lines 1102-1251)
   - RAG-based recommendation system
   - Prediction logic
   - Generate actionable guidance

## Testing

A comprehensive test script is provided: `test_main_file.py`

```bash
python test_main_file.py
```

This validates:
- All imports work
- CSV loads correctly
- Data cleaning pipeline functions
- Labeling system works
- Vectorization successful
- Train/test split correct
- Quick model training test

## Expected Output

When running Main_File.py, you should see:

1. Data loading and cleaning progress
2. Label distribution statistics  
3. Visualization plots (saved to artifacts/)
4. Model training progress
5. Classification reports for each model
6. Cross-validation scores
7. Confusion matrices
8. Narrative generation examples

## Execution Time

- **Test script**: ~30 seconds
- **Full Main_File.py**: 10-30 minutes depending on hardware
  - Data loading/cleaning: ~1 minute
  - Labeling: ~1 minute  
  - Dimensionality reduction: ~2-5 minutes
  - Model training: ~5-20 minutes (depends on CPU cores)
  - Visualization: ~2-5 minutes

## Requirements

- Python 3.9+
- 4GB RAM minimum (8GB recommended)
- No GPU required (CPU-only)
- ~1GB disk space for artifacts

## Troubleshooting

### Module Not Found Errors
```bash
pip install <missing-module>
```

### Memory Issues
- Reduce `max_features` in TfidfVectorizer
- Reduce `n_estimators` in RandomForest
- Comment out memory-intensive visualizations

### CSV Not Found
- Ensure `cleaned_work_orders.csv` is in repository root
- Check file path matches your setup

### Visualization Errors
- Matplotlib backend is set to 'Agg' for headless environments
- Plots are saved to artifacts/ directory instead of displaying

## Next Steps

After successful execution:
1. Review generated artifacts in `artifacts/` directory
2. Examine model performance metrics
3. Test prediction on new maintenance descriptions
4. Integrate with production systems

## Support

For issues or questions:
- Check the main README.md
- Review test_main_file.py for examples
- Contact: fizerco@gmail.com
