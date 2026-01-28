import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# =========================
# CONFIGURAÇÕES
# =========================
DATASET_PATH = "Dataset.csv"
MODEL_PATH = "modelo_estoque_inteligente.pkl"
RANDOM_STATE = 42

# =========================
# CARREGAMENTO
# =========================
df = pd.read_csv(DATASET_PATH)

# =========================
# PRÉ-PROCESSAMENTO
# =========================
df["DATA_EVENTO"] = pd.to_datetime(df["DATA_EVENTO"])

df = df.sort_values(by=["ID_PRODUTO", "DATA_EVENTO"])

# Consumo diário (aprendido pelo modelo)
df["CONSUMO"] = (
    df.groupby("ID_PRODUTO")["QUANTIDADE_ESTOQUE"]
    .shift(1) - df["QUANTIDADE_ESTOQUE"]
)

df["CONSUMO"] = df["CONSUMO"].fillna(0)

# Features temporais
df["DIA"] = df["DATA_EVENTO"].dt.day
df["MES"] = df["DATA_EVENTO"].dt.month
df["DIA_SEMANA"] = df["DATA_EVENTO"].dt.weekday

# =========================
# FEATURES E TARGET
# =========================
features = [
    "ID_PRODUTO",
    "PRECO",
    "FLAG_PROMOCAO",
    "DIA",
    "MES",
    "DIA_SEMANA",
    "QUANTIDADE_ESTOQUE"
]

X = df[features]
y = df["CONSUMO"]

# =========================
# TREINO / TESTE
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

modelo = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

modelo.fit(X_train, y_train)

# =========================
# AVALIAÇÃO
# =========================
y_pred = modelo.predict(X_test)

print("Avaliação do Modelo")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# =========================
# SALVAR MODELO
# =========================
joblib.dump(modelo, MODEL_PATH)
print(f"Modelo salvo em: {MODEL_PATH}")
