import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.clustering import load_data, preprocess_data, train_kmeans


default_args = {
    "owner": "kishlaya",
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
}

dag = DAG(
    "kmeans_clustering_pipeline_kishlaya",
    default_args=default_args,
    description="Custom Airflow clustering pipeline",
    schedule_interval=None,
    catchup=False,
)

load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id="preprocess_task",
    python_callable=preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id="train_model_task",
    python_callable=train_kmeans,
    dag=dag,
)

load_data_task >> preprocess_task >> train_model_task
