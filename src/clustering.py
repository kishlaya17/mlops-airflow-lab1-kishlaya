import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from datetime import datetime


def load_data(**kwargs):
    df = pd.read_csv("/opt/airflow/data/file.csv")
    df = df.dropna()
    print(f"Dataset loaded with shape: {df.shape}")
    

def preprocess_data(**kwargs):
    df = pd.read_csv("/opt/airflow/data/file.csv")
    df = df.dropna()

    df = df.select_dtypes(include=['number'])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    print("Preprocessing completed successfully.")


def train_kmeans(**kwargs):
    df = pd.read_csv("/opt/airflow/data/file.csv")
    df = df.dropna()

    df = df.select_dtypes(include=['number'])
    
    scaler = StandardScaler()
    data = scaler.fit_transform(df)

    sse = []
    K = range(1, 10)

    for k in K:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data)
        sse.append(model.inertia_)

    knee = KneeLocator(K, sse, curve="convex", direction="decreasing")
    optimal_k = knee.elbow if knee.elbow else 3

    final_model = KMeans(n_clusters=optimal_k, random_state=42)
    final_model.fit(data)

    score = silhouette_score(data, final_model.labels_)
    print(f"Optimal clusters: {optimal_k}")
    print(f"Silhouette score: {score}")

    filename = f"/opt/airflow/data/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(final_model, f)

    print(f"Model saved successfully at {filename}")
