# MLOps Airflow Lab 1 – KMeans Clustering Pipeline

This lab  implements a custom Apache Airflow DAG to automate a KMeans clustering workflow.

The goal of this lab was to build a working Airflow pipeline, customize the clustering logic, and ensure the workflow runs successfully end-to-end.

---

## What This Pipeline Does

The DAG consists of three tasks:

1. **load_data_task**
   - Reads the dataset from the data directory
   - Cleans missing values
   - Validates successful loading

2. **preprocess_task**
   - Removes non-numeric columns
   - Applies StandardScaler for feature normalization

3. **train_model_task**
   - Uses the Elbow Method with KneeLocator to determine optimal clusters
   - Trains a final KMeans model
   - Evaluates performance using Silhouette Score
   - Saves the trained model as a `.pkl` file

Task flow:

load_data_task → preprocess_task → train_model_task

---

## Custom Improvements Added

Compared to the base lab template, the following improvements were implemented:

- Filtered non-numeric columns before scaling to prevent conversion errors
- Implemented Elbow Method using KneeLocator
- Added Silhouette Score evaluation for cluster quality
- Saved trained model with timestamp-based filename
- Removed unnecessary XCom usage to simplify DAG design
- Refactored PythonOperator usage for cleaner architecture

---

## How to Run

1. Start Airflow:

docker compose up --build

2. Open Airflow UI:

http://localhost:8080


3. Trigger the DAG:
   `kmeans_clustering_pipeline_kishlaya`

---

## lab Structure

- `dags/` – Airflow DAG definition
- `src/` – Clustering logic implementation
- `data/` – Input dataset
- `docker-compose.yml` – Airflow setup configuration

---

This lab  demonstrates building and debugging a production-style ML workflow using Apache Airflow
