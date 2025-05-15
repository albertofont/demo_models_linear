from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from teradataml import DataFrame, copy_to_sql
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    model = joblib.load(f"{context.artifact_input_path}/model.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    test_df = DataFrame.from_query(context.dataset_info.sql)
    test_pdf = test_df.to_pandas(all_rows=True)

    X_test = test_pdf[feature_names]
    y_test = test_pdf[target_name]

    print("Scoring")
    y_pred = model.predict(X_test)

    y_pred_tdf = pd.DataFrame(y_pred, columns=[target_name])
    y_pred_tdf["PatientId"] = test_pdf["PatientId"].values

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred, average='binary')),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred, average='binary')),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred, average='binary'))
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    save_plot('Confusion Matrix', context=context)

    # ROC Curve — binary only
    if len(np.unique(y_test)) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_proba)
        save_plot('ROC Curve', context=context)

    # Feature Importance from Random Forest
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
    save_plot('Feature Importance', context=context)

    # Save predictions to temp table in Teradata
    predictions_table = "evaluation_preds_tmp"
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    record_evaluation_stats(
        features_df=test_df,
        predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
        importance=feature_importance,
        context=context
    )
