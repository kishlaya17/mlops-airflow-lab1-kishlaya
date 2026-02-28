import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from datetime import datetime


def load_data():
    df = pd.read_csv("data/file.csv")
    df = df.dropna()
    print(f"Dataset loaded with shape: {df.shape}")
    return df


def preprocess_data(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    print("Preprocessing complete.")
    return scaled


def train_kmeans(data):
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

    filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(final_model, f)

    return filename
