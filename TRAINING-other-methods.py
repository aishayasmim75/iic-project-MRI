import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ==================================================================
# Carregamento dos dados e preparação para as etapas de treinamento
# ==================================================================
df = pd.read_csv(r"PROJETO\dataset-combined.csv", sep=",")
df["Age"] = df["Age"] * 100


# ROIs selecionadas na etapa de feature selection: -------------
with open("rois_selecionadas_definitivo.txt", "r") as f:
    # .strip() remove o caractere de quebra de linha (\n)
    rois_finais = [linha.strip() for linha in f.readlines()]

X = df[rois_finais]  # features = ROIs selecionadas
y = df["AD"].to_numpy()  # AD = 0 → controle, AD = 1 → Alzheimer

# Separação dos dados: 30% teste, 70% treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Dados carregados com sucesso. Treinando modelos baseados em instâncias (KNN) ou em árvores (Árvores de Decisão, Random Forests e Boosting) com {X_train.shape[1]} ROIs.")


# ==================================================================
# MODELO 1 - KNN (K-NEAREST NEIGHBORS)
# ==================================================================
print("\n---Treinando o modelo com KNN ---")
pipe_knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]) 
## padronização obrigatória já que o método depende das distâncias

param_grid_knn = {
    "knn__n_neighbors": [3, 5, 7, 11, 15],  # Número de vizinhos a testar
     # (n° ímpares para evitar empate na votação)
    "knn__weights": ["uniform", "distance"],  # Peso dos vizinhos 
     # uniform → mesmo peso para todos os vizihnos
     # distance → vizinhos mais proximos à observação analisada tem mais peso
    "knn__metric": ["euclidean", "manhattan"],  # Métrica de distância geométrica
    }

grid_knn = GridSearchCV(pipe_knn, param_grid_knn, cv=5, scoring="roc_auc", n_jobs=-1)
grid_knn.fit(X_train, y_train)

print(f"Melhor configuração KNN: {grid_knn.best_params_}")
print(f"Melhor AUC-ROC (CV): {grid_knn.best_score_:.4f}")


# ==================================================================
# MODELO 2 - ÁRVORE DE DECISÃO SIMPLES
# ==================================================================
print("\n---Treinando o modelo com Árvore de Decisão Simples ---")
# esse método já n precisa de padronização/escalonamento, mas vou manter o uso de pipeline para manter o padrão dos códigos
pipe_dt = Pipeline([("dt", DecisionTreeClassifier(random_state=42, class_weight='balanced'))])

param_grid_dt = {
    "dt__max_depth": [3, 5, 10, None],  # Controla o crescimento/profundidade
    "dt__min_samples_split": [2, 5, 10],  # Mínimo de dados para gerar um nó
    "dt__criterion": ["gini", "entropy"],  # Métrica de pureza da divisão
}

grid_dt = GridSearchCV(pipe_dt, param_grid_dt, cv=5, scoring="roc_auc", n_jobs=-1)
grid_dt.fit(X_train, y_train)

print(f"Melhor configuração Árvore: {grid_dt.best_params_}")
print(f"Melhor AUC-ROC (CV): {grid_dt.best_score_:.4f}")

# ==================================================================
# MODELO 3 - RANDOM FOREST (ensemble por Bagging)
# ==================================================================
print("\n---Treinando o modelo com Random Forest ---")
pipe_rf = Pipeline([("rf", RandomForestClassifier(random_state=42, class_weight='balanced'))])

param_grid_rf = {
    "rf__n_estimators": [50, 100, 200],  # Quantidade de árvores na floresta
    "rf__max_depth": [5, 10, None],
    "rf__max_features": ["sqrt", "log2"],  # Número de ROIs sorteadas por árvore
    # função que vai controlar a 'diversidade forçada' do subconjunto de atributos
    ## que uma arvore do ensemble pode usar como nó inicial, ja que queremos árvores diferentes entre si
}

grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="roc_auc", n_jobs=-1)
grid_rf.fit(X_train, y_train)

print(f"Melhor configuração Random Forest: {grid_rf.best_params_}")
print(f"Melhor AUC-ROC (CV): {grid_rf.best_score_:.4f}")

# ==================================================================
# MODELO 4 - AdaBoost (adaptive boosting → ensemble por Boosting)
# ==================================================================
print("\n---Treinando o modelo com AdaBoost ---")
pipe_ada = Pipeline([("ada", AdaBoostClassifier(random_state=42))])

param_grid_ada = {"ada__n_estimators": [50, 100, 200],
                  "ada__learning_rate": [0.01, 0.1, 1.0],}

grid_ada = GridSearchCV(pipe_ada, param_grid_ada, cv=5, scoring="roc_auc", n_jobs=-1)
grid_ada.fit(X_train, y_train)
print(f"Melhor configuração AdaBoost: {grid_ada.best_params_}")
print(f"Melhor AUC-ROC (AdaBoost): {grid_ada.best_score_:.4f}")

# ==================================================================
# MODELO 5 - GBM (Gradient Boosting Machine Clássico → ensemble por Boosting)
# ==================================================================
print("\n---Treinando o modelo com GBM ---")
pipe_gbm = Pipeline([("gbm", GradientBoostingClassifier(random_state=42))])

param_grid_gbm = {
    "gbm__n_estimators": [50, 100, 150],
    "gbm__learning_rate": [0.01, 0.1, 0.2],
    "gbm__max_depth": [3, 5],  # GBM tradicional usa tocos/árvores rasas
}

grid_gbm = GridSearchCV(pipe_gbm, param_grid_gbm, cv=5, scoring="roc_auc", n_jobs=-1)
grid_gbm.fit(X_train, y_train)
print(f"Melhor configuração GBM: {grid_gbm.best_params_}")
print(f"Melhor AUC-ROC (GBM): {grid_gbm.best_score_:.4f}")

# ==================================================================
# MODELO 6 - XGBOOST (Extreme Gradient Boosting Machine → ensemble por Boosting)
# ==================================================================
print("\n---Treinando o modelo com XGBOOST ---")

pipe_xgb = Pipeline([("xgb", XGBClassifier(random_state=42, eval_metric="logloss"))])

param_grid_xgb = {
    "xgb__n_estimators": [50, 100, 150],  # Número de árvores sequenciais
    "xgb__learning_rate": [0.01, 0.1, 0.2],  # Passo de aprendizado do algoritmo
    "xgb__max_depth": [3, 5, 7],  # Árvores de boosting costumam ser mais rasas
}

grid_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, cv=5, scoring="roc_auc", n_jobs=-1)
grid_xgb.fit(X_train, y_train)

print(f"Melhor configuration XGBoost: {grid_xgb.best_params_}")
print(f"Melhor AUC-ROC (CV): {grid_xgb.best_score_:.4f}")

# =====================================================================
# SALVAMENTO DOS RESULTADOS E MODELOS 
# =====================================================================
modelos_ensemble_instancia = {
    "KNN": grid_knn.best_estimator_,
    "Arvore_Decisao": grid_dt.best_estimator_,
    "Random_Forest": grid_rf.best_estimator_,
    "XGBoost": grid_xgb.best_estimator_,
    "AdaBoost": grid_ada.best_estimator_,
    "GBM": grid_gbm.best_estimator_,
}

# Salvando os novos 6 modelos
joblib.dump(modelos_ensemble_instancia, "modelos_ensemble_treinados.pkl")

# Gerando a tabela resumo expandida
resumo_treino_ens = pd.DataFrame(
    {
        "Modelo": [
            "KNN",
            "Árvore de Decisão",
            "Random Forest",
            "XGBoost",
            "AdaBoost",
            "GBM",
        ],
        "Melhor_Parametro": [
            grid_knn.best_params_,
            grid_dt.best_params_,
            grid_rf.best_params_,
            grid_xgb.best_params_,
            grid_ada.best_params_,
            grid_gbm.best_params_,
        ],
        "AUC_ROC_CV_Treino": [
            grid_knn.best_score_,
            grid_dt.best_score_,
            grid_rf.best_score_,
            grid_xgb.best_score_,
            grid_ada.best_score_,
            grid_gbm.best_score_,
        ],
    }
)
resumo_treino_ens.to_csv("resumo_treino_ensembles.csv", index=False)

print("\n" + "=" * 55)
print(" Todos os 6 modelos salvos em arquivos pkl/csv.")
print("=" * 55)