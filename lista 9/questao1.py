import kagglehub
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# Carregar a base

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = os.path.join(path, "creditcard.csv")

print("Diretorio do dataset:", path)
print("Caminho do CSV:", csv_path)

df = pd.read_csv(csv_path)

# Guardar uma cópia "bruta" para o modelo ANTES do pré-processamento
df_raw = df.copy()

print("\n=== VISUALIZACAO INICIAL (Etapa 1) ===")
print("Formato da base:", df.shape)
print(df.head())
print("\nDistribuicao da classe (0 = normal, 1 = fraude):")
print(df["Class"].value_counts())

#  Valores ausentes

print("\n=== VALORES AUSENTES (Etapa 2) ===")
print(df.isna().sum())

# (Se tivesse valores faltantes, aqui você escolheria se vai
#  remover linhas/colunas ou preencher com média/mediana etc.)

# Redundância / inconsistência

print("\n=== DUPLICATAS (Etapa 3) ===")
num_dup = df.duplicated().sum()
print("Numero de linhas duplicadas:", num_dup)

if num_dup > 0:
    df = df.drop_duplicates()
    print("Novo formato da base apos remover duplicatas:", df.shape)

# Outliers em Amount (exemplo com IQR)

print("\n=== OUTLIERS EM Amount (Etapa 4) ===")
Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR

mask_out = (df["Amount"] < lim_inf) | (df["Amount"] > lim_sup)
print("Quantidade de outliers em Amount:", mask_out.sum())

# Estratégia simples: "capar" (truncar) os valores nos limites
df.loc[df["Amount"] > lim_sup, "Amount"] = lim_sup
df.loc[df["Amount"] < lim_inf, "Amount"] = lim_inf

# Normalização / padronização

print("\n=== NORMALIZACAO (Etapa 5) ===")
scaler = StandardScaler()
df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
print(df[["Time", "Amount"]].head())

# Correlação / multicolinearidade

print("\n=== CORRELACAO COM A CLASSE (Etapa 6) ===")
corr = df.corr()
corr_com_classe = corr["Class"].sort_values(ascending=False)
print(corr_com_classe.head(15))


# Codificação de variáveis

print("\n=== CODIFICACAO (Etapa 7) ===")
print("Nenhuma variável categorica: nao foi necessario One-Hot/Label Encoding.")

# Funções auxiliares para treinar modelo

def treinar_modelo_sem_preprocess(df_base):
    """Modelo ANTES do pre-processamento:
       - usa dados brutos
       - sem balanceamento
       - sem normalizacao extra (ja vem como esta no CSV)
    """
    print("\n======== MODELO ANTES DO PRe-PROCESSAMENTO ========")

    X = df_base.drop("Class", axis=1)
    y = df_base["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Distribuicao treino (y_train):")
    print(np.bincount(y_train))

    modelo = LogisticRegression(max_iter=5000, n_jobs=-1)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\nRelatorio de classificacao (antes):")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nMatriz de confusao (antes):")
    print(confusion_matrix(y_test, y_pred))


def treinar_modelo_com_preprocess(df_base):
    """Modelo DEPOIS do pre-processamento:
       - usa df ja tratado (sem duplicatas, com outliers tratados, time/amount escalados)
       - aplica SMOTE no conjunto de treino para balancear
    """
    print("\n======== MODELO DEPOIS DO PRe-PROCESSAMENTO ========")

    X = df_base.drop("Class", axis=1)
    y = df_base["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Distribuicao treino ANTES do SMOTE:")
    print(np.bincount(y_train))

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("Distribuicao treino DEPOIS do SMOTE:")
    print(np.bincount(y_train_bal))

    modelo = LogisticRegression(max_iter=5000, n_jobs=-1)
    modelo.fit(X_train_bal, y_train_bal)

    y_pred = modelo.predict(X_test)

    print("\nRelatorio de classificacao (depois):")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nMatriz de confusao (depois):")
    print(confusion_matrix(y_test, y_pred))


# Balanceamento Treino/validação

treinar_modelo_sem_preprocess(df_raw)

treinar_modelo_com_preprocess(df)
