from teradataml import copy_to_sql, DataFrame
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

import joblib
import pandas as pd


def score(context: ModelContext, **kwargs):

    aoa_create_context()

    model = joblib.load(f"{context.artifact_input_path}/model.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    features_tdf = DataFrame.from_query(context.dataset_info.sql)
    features_pdf = features_tdf.to_pandas(all_rows=True)

    print("Scoring")
    predictions_pdf = model.predict(features_pdf[feature_names])
    print("Finished Scoring")

    # Format predictions
    predictions_pdf = pd.DataFrame(predictions_pdf, columns=[target_name])
    predictions_pdf[entity_key] = features_pdf.index.values
    predictions_pdf["job_id"] = context.job_id
    predictions_pdf["json_report"] = ""

    # Reorder columns
    predictions_pdf = predictions_pdf[["job_id", entity_key, target_name, "json_report"]]

    # Save to Teradata
    copy_to_sql(df=predictions_pdf,
                schema_name=context.dataset_info.predictions_database,
                table_name=context.dataset_info.predictions_table,
                index=False,
                if_exists="append",
                primary_index=["job_id", "PatientId"],
                set_table=True)

    print("Saved predictions in Teradata")

    # Retrieve predictions to record stats
    predictions_df = DataFrame.from_query(f"""
        SELECT 
            * 
        FROM {context.dataset_info.get_predictions_metadata_fqtn()} 
        WHERE job_id = '{context.job_id}'
    """)

    record_scoring_stats(features_df=features_tdf, predicted_df=predictions_df, context=context)


# REST scoring interface (unchanged)
class ModelScorer(object):

    def __init__(self):
        self.model = joblib.load("artifacts/input/model.joblib")

    def predict(self, data):
        return self.model.predict(data)
