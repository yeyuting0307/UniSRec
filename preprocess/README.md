# Recommendation System Preprocessing Pipeline

This repository contains a Python-based preprocessing pipeline for building a recommendation system. It leverages Google Cloud Platform (GCP) services such as BigQuery and Cloud Storage for data extraction, transformation, and storage.

## Features

- Extracts raw data from BigQuery using SQL queries.
- Transforms and preprocesses data for both pretraining and finetuning recommendation models.
- Supports excluding cold-start users and items.
- Handles user-item interaction data, including sequence generation.
- Uploads preprocessed data to Cloud Storage for model training and inference.

## Scripts

- **get_raw_ga4.py**: Extracts raw Google Analytics 4 data from BigQuery and saves it to Cloud Storage.
- **get_raw_post.py**: Extracts raw post data from BigQuery and saves it to Cloud Storage.
- **make_recsys_raw_data.py**: Creates raw data for the recommendation system, including user tokens, item tokens, and item text.
- **make_pretrain_data.py**: Prepares data for pretraining recommendation models.
- **make_finetune_data.py**: Prepares data for finetuning recommendation models.
- **make_inference_data.py**: Prepares data for generating recommendations during inference.
- **bq_update_user_token.py**: Updates user token information in BigQuery.
- **bq_update_item_token.py**: Updates item token information in BigQuery.
- **bq_update_item_text.py**: Updates item text information in BigQuery.
- **bq_update_user_item_interact.py**: Updates user-item interaction data in BigQuery.
- **auto_readme.py**: A utility script to automatically generate parts of this README file.

## Pipeline Architecture
![preprocess](preprocess_pipelines.png)

## Configuration

- Project-specific configurations, such as GCP project ID, bucket name, and location, can be set as environment variables or modified within the scripts.

## Usage

1. Set up GCP credentials and configure environment variables.
2. Install the required Python libraries.
3. Execute the scripts in the appropriate order based on your specific needs.

## Notes

- The pipeline assumes the existence of specific BigQuery tables and Cloud Storage buckets.
- The code is intended as a starting point and may require modifications based on your specific use case and data schema.
