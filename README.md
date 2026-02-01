# bakery-sales

## Overview
This project demonstrates an end-to-end bakery sales prediction pipeline,
covering both model development and a production-style batch inference workflow.

## Inference Pipeline (MLOps)
- Input data stored in Amazon S3
- Trained regression model stored as an artifact in S3
- Batch inference script downloads input and model, runs prediction, and uploads results to S3
- Output files are generated once per day with date-suffixed filenames
- Designed to be scheduled via cron (example configuration provided)

## S3 Structure
- s3://<bucket>/input/        : inference input CSV
- s3://<bucket>/artifacts/    : trained model artifacts
- s3://<bucket>/output/       : prediction results

## Model Development (Notebook)
Model training and feature engineering were performed in a Jupyter Notebook.
Details on preprocessing, feature engineering, and model comparison
are documented in `bakery_sales.ipynb`.

## Repository Structure

bakery-sales/
├── data_sample/            # Sample input data (for reference)
├── notebooks/
│   └── bakery_sales.ipynb  # Model development and analysis notebook
├── src/
│   ├── features.py         # Feature selection and preprocessing logic
│   ├── io_s3.py            # S3 input/output utilities
│   └── predict.py          # Batch inference entry point
├── README.md
└── requirements.txt


## Daily Batch Execution

The inference script is designed to be executed once per day.
An example cron configuration (not enabled by default) is shown below:

```cron
# Daily batch inference at 06:00 (JST)
# 0 6 * * * /usr/bin/python /path/to/bakery-sales/src/predict.py \
#   --input_s3  s3://<bucket>/input/X_infer.csv \
#   --output_s3 s3://<bucket>/output \
#   --model_s3  s3://<bucket>/artifacts/BakarySales.pkl \
#   --id_cols "" \
#   --feature_cols <comma-separated feature columns>
```cron

## About Dataset

The dataset is not included in this repository due to license restrictions. Please download it directly from Kaggle.

https://www.kaggle.com/competitions/bakery-sales-prediction-summer-2025/data

## Notes

This project prioritizes clarity of pipeline design and operational flow
rather than providing a fully automated production system.

The inference pipeline can be extended to scheduled execution
using cron or AWS-managed schedulers.
