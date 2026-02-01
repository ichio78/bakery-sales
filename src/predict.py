import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from io_s3 import download_s3_to_local, upload_local_to_s3
from features import build_features


def setup_logger():
    logger = logging.getLogger("bakery_predict")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def schema_check(df: pd.DataFrame, required_cols: list[str], allow_null: bool, logger):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = df[required_cols].isna().sum().to_dict()
    total_null = int(sum(null_counts.values()))
    logger.info(f"Null counts (required cols): {json.dumps(null_counts, ensure_ascii=False)}")

    if (not allow_null) and total_null > 0:
        raise ValueError(f"Found nulls in required columns (total={total_null}).")


def summarize_predictions(pred: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float)
    return {
        "n": int(pred.size),
        "min": float(np.min(pred)) if pred.size else None,
        "max": float(np.max(pred)) if pred.size else None,
        "mean": float(np.mean(pred)) if pred.size else None,
        "std": float(np.std(pred)) if pred.size else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_s3", required=True, help="s3://.../input.csv")
    parser.add_argument("--output_s3", required=True, help="s3://.../output(directory, filename will be auto-generated)")

    # ★ The trained model is downloaded from the S3 artifacts directory 
    parser.add_argument("--model_s3", required=True, help="s3://.../artifacts/.../model.pkl")

    parser.add_argument("--id_cols", default="date,store_id", help="Columns to keep in output")
    parser.add_argument("--feature_cols", required=True, help="Comma-separated feature columns used by the model")
    parser.add_argument("--allow_null", action="store_true", help="Allow nulls in required columns")
    parser.add_argument("--local_workdir", default="/tmp/bakery_run")
    args = parser.parse_args()

    logger = setup_logger()
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_date = datetime.now().strftime("%Y%m%d")
    output_filename = f"predictions_{run_date}.csv"
    output_s3_path = args.output_s3.rstrip("/") + "/" + output_filename

    t0 = time.time()

    os.makedirs(args.local_workdir, exist_ok=True)
    local_in = os.path.join(args.local_workdir, f"input_{run_id}.csv")
    local_model = os.path.join(args.local_workdir, f"model_{run_id}.pkl")
    local_out = os.path.join(args.local_workdir, f"pred_{run_id}.csv")

    logger.info(f"Run start: run_id={run_id}")
    logger.info(f"input_s3={args.input_s3}")
    logger.info(f"output_s3={args.output_s3}")
    logger.info(f"model_s3={args.model_s3}")

    # 1) Download input data from S3 
    download_s3_to_local(args.input_s3, local_in)
    df = pd.read_csv(local_in)
    logger.info(f"Loaded input: rows={len(df)} cols={len(df.columns)}")

    # 2) Download the trained model artifact from S3
    download_s3_to_local(args.model_s3, local_model)

    # 3) Validate input schema
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    required_cols = sorted(set(id_cols + feature_cols))
    schema_check(df, required_cols=required_cols, allow_null=args.allow_null, logger=logger)

    # 4) Build features and run inference
    X = build_features(df, feature_cols=feature_cols)
    model = joblib.load(local_model)
    pred = model.predict(X)

    # 5) Create output file
    out_df = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
    out_df["prediction"] = pred
    out_df.to_csv(local_out, index=False)

    # 6) Log prediction summary statistics
    stats = summarize_predictions(pred)
    logger.info(f"Prediction stats: {json.dumps(stats, ensure_ascii=False)}")

    # 7) Upload prediction results to S3
    upload_local_to_s3(local_out, output_s3_path)
    logger.info(f"Uploaded output: {output_s3_path}")

    elapsed = time.time() - t0
    logger.info(f"Run end: run_id={run_id} elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()
