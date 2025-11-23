import os
import numpy as np
import pandas as pd

import kagglehub

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


# Carregar a base (mesma da Questao 1)

def carregar_base():
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(path, "creditcard.csv")
    print("Diretorio do dataset:", path)
    print("Caminho do CSV:", csv_path)

    df = pd.read_csv(csv_path)
    print("Formato original:", df.shape)
    print(df.head())
    print("\nDistribuicao original da classe (0=normal, 1=fraude):")
    print(df["Class"].value_counts())
    return df


# Pre-processamento basico para AGRUPAMENTO


def preprocessar_para_agrupamento(df):
    df = df.copy()

    # Remover duplicatas
    num_dup = df.duplicated().sum()
    print("\n[Pre-processamento] Linhas duplicadas:", num_dup)
    if num_dup > 0:
        df = df.drop_duplicates()
        print("[Pre-processamento] Novo formato apos remover duplicatas:", df.shape)

    # Tratar outliers em Amount (IQR + "capping")
    Q1 = df["Amount"].quantile(0.25)
    Q3 = df["Amount"].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    mask_out = (df["Amount"] < lim_inf) | (df["Amount"] > lim_sup)
    print("[Pre-processamento] Outliers em Amount:", mask_out.sum())

    df.loc[df["Amount"] < lim_inf, "Amount"] = lim_inf
    df.loc[df["Amount"] > lim_sup, "Amount"] = lim_sup

    # Separar alvo antes de escalar
    y_true = df["Class"].values
    features = [c for c in df.columns if c != "Class"]

    # Escalar TODOS os atributos preditores (Time, Amount e V1..V28)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    print("[Pre-processamento] X_scaled shape:", X_scaled.shape)
    return X_scaled, y_true, features


# Funcoes de avaliacao de agrupamento

def avaliar_clusters(X, labels, nome_modelo, sample_size=50000):
    labels = np.asarray(labels)

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n=== {nome_modelo}: resumo dos clusters ===")
    for lab, cnt in zip(unique_labels, counts):
        print(f"Cluster {lab}: {cnt} pontos")

    # Ignorar pontos marcados como ruido (-1), se houver
    mask_valid = labels != -1
    X_valid = X[mask_valid]
    labels_valid = labels[mask_valid]

    n_clusters = len(np.unique(labels_valid))
    print(f"Numero de clusters (sem ruido): {n_clusters}")

    if n_clusters < 2:
        print("Menos de 2 clusters validos -> nao eh possivel calcular silhueta.")
        return

    # Subamostragem para acelerar as metricas, se necessario
    if X_valid.shape[0] > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_valid.shape[0], size=sample_size, replace=False)
        X_eval = X_valid[idx]
        labels_eval = labels_valid[idx]
        print(f"Calculando metricas em amostra de {sample_size} pontos.")
    else:
        X_eval = X_valid
        labels_eval = labels_valid

    sil = silhouette_score(X_eval, labels_eval)
    ch = calinski_harabasz_score(X_eval, labels_eval)
    db = davies_bouldin_score(X_eval, labels_eval)

    print(f"Silhueta media: {sil:.4f}")
    print(f"Calinski-Harabasz: {ch:.2f}")
    print(f"Davies-Bouldin: {db:.4f}")


def comparar_com_classe(y_true, labels, nome_modelo):
    print(f"\n=== {nome_modelo}: comparacao cluster x classe real ===")
    df_cross = pd.crosstab(labels, y_true, rownames=["cluster"], colnames=["Class"])
    print(df_cross)


# KMeans com k=2

def rodar_kmeans(X, y_true):
    print("\n==============================")
    print("KMEANS (k=2)")
    print("==============================")

    kmeans = KMeans(
        n_clusters=2,
        n_init=10,
        random_state=42
    )
    labels = kmeans.fit_predict(X)

    avaliar_clusters(X, labels, "KMEANS")
    comparar_com_classe(y_true, labels, "KMEANS")


# DBSCAN

def rodar_dbscan(X, y_true):
    print("\n==============================")
    print("DBSCAN")
    print("==============================")

    # Valores iniciais de hiperparametros (podem ser ajustados)
    # eps controla o "raio" de vizinhanca; min_samples, o minimo de pontos por grupo
    dbscan = DBSCAN(
        eps=2.0,
        min_samples=50,
        n_jobs=-1
    )
    labels = dbscan.fit_predict(X)

    avaliar_clusters(X, labels, "DBSCAN")
    comparar_com_classe(y_true, labels, "DBSCAN")


# SOM simples (2 neuronios)

class SimpleSOM:
    """
    SOM 2D bem simples, apenas para fins de trabalho.
    Aqui vamos usar um mapa 2x1 (ou seja, 2 neuronios) para forcar 2 grupos.
    """

    def __init__(self, m, n, input_dim, sigma=1.0, learning_rate=0.5, random_state=42):
        self.m = m
        self.n = n
        self.input_dim = input_dim
        self.sigma0 = sigma
        self.learning_rate0 = learning_rate

        rng = np.random.RandomState(random_state)
        # pesos: (m*n, input_dim)
        self.weights = rng.normal(size=(m * n, input_dim))
        # posicoes na grade para cada neuronio
        self.locations = np.array([(i, j) for i in range(m) for j in range(n)])

    def _bmu(self, x):
        # x: (input_dim,)
        diff = self.weights - x  # (m*n, input_dim)
        dist2 = np.einsum("ij,ij->i", diff, diff)
        return np.argmin(dist2)

    def _neighborhood(self, bmu_idx, sigma):
        bmu_loc = self.locations[bmu_idx]
        dist2 = np.sum((self.locations - bmu_loc) ** 2, axis=1)
        h = np.exp(-dist2 / (2 * (sigma ** 2)))
        return h  # shape: (m*n,)

    def fit(self, X, num_epochs=10):
        n_samples = X.shape[0]
        for epoch in range(num_epochs):
            # decaimento simples de sigma e learning_rate
            t = epoch / max(1, num_epochs - 1)
            sigma = self.sigma0 * (1.0 - t)
            lr = self.learning_rate0 * (1.0 - t)

            print(f"Epoca {epoch+1}/{num_epochs} - sigma={sigma:.3f}, lr={lr:.3f}")

            # embaralhar dados a cada epoca
            idx = np.random.permutation(n_samples)
            for i in idx:
                x = X[i]
                bmu_idx = self._bmu(x)
                h = self._neighborhood(bmu_idx, sigma)  # (m*n,)
                # atualiza todos os neuronios
                self.weights += lr * h[:, np.newaxis] * (x - self.weights)

    def predict(self, X):
        labels = []
        for x in X:
            labels.append(self._bmu(x))
        return np.array(labels)


def rodar_som(X, y_true):
    print("\n==============================")
    print("SOM (2x1)")
    print("==============================")

    input_dim = X.shape[1]
    som = SimpleSOM(m=2, n=1, input_dim=input_dim, sigma=1.0, learning_rate=0.5, random_state=42)

    # Para acelerar o treino, usar amostra se a base for muito grande
    if X.shape[0] > 50000:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], size=50000, replace=False)
        X_train_som = X[idx]
        print("[SOM] Treinando em amostra de 50000 pontos.")
    else:
        X_train_som = X

    som.fit(X_train_som, num_epochs=10)

    labels = som.predict(X)

    avaliar_clusters(X, labels, "SOM")
    comparar_com_classe(y_true, labels, "SOM")


# MAIN

def main():
    df = carregar_base()
    X, y_true, features = preprocessar_para_agrupamento(df)

    # Importante: algoritmo de agrupamento NAO usa y_true na entrada.
    # Estamos guardando y_true apenas para ANALISAR depois.

    rodar_kmeans(X, y_true)
    rodar_dbscan(X, y_true)
    rodar_som(X, y_true)


if __name__ == "__main__":
    main()
