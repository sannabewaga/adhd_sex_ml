from src.pipeline.predict_pipeline import PredictPipeline


if __name__ == "__main__":
    pipeline = PredictPipeline()
    prediction_df = pipeline.predict_from_test_data()
    print(prediction_df.head())
