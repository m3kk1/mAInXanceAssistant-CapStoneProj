# mAInXanceAssistant-CapStoneProj
Capstone Project Report – mAInXance Assistant

1. Project Overview

This project builds a predictive maintenance assistant that takes technician notes or problem descriptions and suggests the most likely maintenance actions. The workflow covers data management, model training, performance evaluation, and a narrative conversion step that makes predictions interpretable and useful for end users.

The goal is to demonstrate both technical knowledge of machine learning pipelines and practical application for a diagnostic assistant.

2. Data Management

Cleaning: Text descriptions were preprocessed (tokenization, lowercasing, removing stopwords) and mapped to labeled maintenance actions.

Labeling: Actions were categorized into classes such as Tighten/Adjust, Electrical Fix, Replace Part, etc.

3. Dimensionality Reduction and Visualizations

PCA: Showed overlapping class clusters, confirming the dataset is not perfectly separable.

t-SNE: Produced tighter local clusters, highlighting similarities within action categories.

UMAP: Preserved global structure, showing stronger grouping than PCA.

Interpretation: These plots confirm that while patterns exist, there is inherent overlap between classes. This explains why accuracy does not reach 100%.

4. Model Training and Testing
Baseline

Accuracy: ~29%

Macro F1: ~0.05

Serves as the benchmark.

Naive Bayes

Slight improvement over baseline, but still weak.

Underfitting: struggles to capture feature interactions in text data.

Decision Tree

Train accuracy ~99%, test accuracy ~94%.

Cross-validation macro F1 dropped significantly (~0.12).

Conclusion: Classic overfitting. Memorizes training data but fails to generalize.

Random Forest

Accuracy: ~94–95%

Balanced F1 scores across most labels.

Less prone to overfitting due to ensemble averaging.

Conclusion: Reliable, strong model for this dataset.

MLP (Neural Net)

Did not outperform Random Forest.

Requires more tuning and larger compute resources.

Conclusion: Adequate but not the best choice given dataset and resources.

XGBoost

Accuracy comparable to Random Forest.

Handles structured, sparse data (like TF-IDF vectors) well.

Conclusion: One of the top-performing models.

5. Overfitting and Generalization

Most overfitting: Decision Tree.

Most underfitting: Naive Bayes.

Best balance: Random Forest and XGBoost.

Neural Net: Did not overfit heavily but underperformed relative to ensembles.

6. Narrative Conversion

Raw predictions are not directly useful to technicians. To address this, a narrative generation step was introduced.

Combines predicted action probabilities with retrieval of similar past cases.

Produces short, professional summaries of what the issue likely is, what to check first, and recommended next actions.

This transforms the project from a technical classifier into a usable assistant app.

7. Key Findings and Lessons Learned

Not all models are equal: traditional ensembles outperformed deep learning on this dataset.

Visualization confirmed that classes overlap, which explains misclassifications.

Cross-validation was critical to reveal overfitting in decision trees.

Narrative conversion makes the system practical and user-focused.

8. Next Steps

Expand dataset with more balanced class distribution.

Explore model tuning for MLP with more compute resources.

Integrate a retrieval-augmented generation (RAG) approach for narrative function to blend structured predictions with free-text reasoning.

Deploy Random Forest or XGBoost as the backbone model, with narrative conversion layered on top.
