from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from teradataml import DataFrame
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf = train_df.to_pandas(all_rows=True)

    # split data into X and y
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]

    print("Starting training...")

    # fit RandomForest classifier to training data
    model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    print("Finished training")

    # export model artefact
    joblib.dump(model, f"{context.artifact_output_path}/model.joblib")
    print("Saved trained model")

    # Feature importance
    rf_model = model.named_steps['rf']
    importances = rf_model.feature_importances_
    feature_importance = dict(zip(feature_names, importances))

    # Save feature importance plot
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [feature_names[i] for i in sorted_idx[:10]]
    top_importances = importances[sorted_idx[:10]]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], top_importances[::-1])
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Feature Importances (Random Forest Classifier)")
    save_plot("feature_importance.png", context=context)

    # Record training stats
    record_training_stats(train_df,
                          features=feature_names,
                          targets=[target_name],
                          categorical=[target_name],
                          feature_importance=feature_importance,
                          context=context)
