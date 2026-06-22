import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import joblib

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
    # apesar dos dados já serem balanceados, o parâmetro "stratify=y" 
    # # garante que a proporção de classes seja mantida tanto no conjunto de treino quanto no conjunto de teste.

# salvando o conjunto de teste para para o script final de avaliação dos modelos
joblib.dump((X_test, y_test), "dados_teste.pkl")

#OBS: A PADRONIZAÇÃO DOS DADOS SERÁ FEITA DENTRO DOS PIPELINES DE CADA MODELO, 
# # PARA EVITAR VAZAMENTO DE DADOS (DATA LEAKAGE)


print(f"Dados carregados. Treinando com {X_train.shape[1]} ROIs selecionadas.")
# -----------------------------------------------------------------

# ==================================================================
# MODELO LINEAR 1: REGRESSÃO LOGÍSTICA 
# ================================================================== 
print("\n--- Treinando Regressão Logística ---")

## pineline
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),  # Padronização dos dados
    ("lr", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42, class_weight='balanced'))
    # O parâmetro "solver='lbfgs'" é um algoritmo de otimização eficiente para problemas de regressão logística,
    # e "max_iter=1000" garante que o modelo tenha tempo suficiente para convergir (n° grande de features)
    # a função ja realiza a regularização L2 por padrão, e o parâmetro "C" controla a força dessa regularização 
    ##(menor C = mais regularização)
    # apesar de eu ja ter realizado uma etapa de feature extraction das ROIs, o Ridge (regularização com penalidade=L2)
    ## ajuda na estabilidade da função LogisticRegression, ainda mais com esse tipo de dados que estão propensos
    ### a uma multicolinearidade alta 
])

# Testando diferentes forças de regularização Ridge (L2)
param_grid_lr = {"lr__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="roc_auc", n_jobs=-1)
 # busca do melhor hiperparâmetro "C" usando validação cruzada (cv=5) e 
 ## métrica AUC-ROC para avaliação do desempenho do modelo
grid_lr.fit(X_train, y_train)

print(f"Melhor C (Reg. Logística): {grid_lr.best_params_['lr__C']}")
print(f"Melhor AUC-ROC (CV): {grid_lr.best_score_:.4f}")

# ==================================================================
# MODELO LINEAR 2: SVM LINEAR (Support Vector Machine Linear)
# ==================================================================
print("\n--- Treinando SVM Linear ---")

pipe_svm = Pipeline(
    [("scaler", StandardScaler()),
    # dual=False é recomendado quando n_samples > n_features
    ("svm", LinearSVC(dual=False, max_iter=5000, random_state=42, class_weight='balanced')),])

param_grid_svm = {"svm__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=5, scoring="roc_auc", n_jobs=-1)
grid_svm.fit(X_train, y_train)

print(f"Melhor C (SVM Linear): {grid_svm.best_params_['svm__C']}")
print(f"Melhor AUC-ROC (CV): {grid_svm.best_score_:.4f}")

# ==================================================================
# MODELO LINEAR 3: ANÁLISE DISCRIMINANTE LINEAR (LDA)
# ==================================================================
print("\n--- Treinando LDA ---")

pipe_lda = Pipeline([("scaler", StandardScaler()), ("lda", LinearDiscriminantAnalysis())])

# O LDA tradicional não tem hiperparâmetros críticos para buscar em grade,
# mas rodamos no GridSearchCV para computar a métrica no mesmo esquema de validação cruzada
grid_lda = GridSearchCV(pipe_lda, param_grid={}, cv=5, scoring="roc_auc", n_jobs=-1)
grid_lda.fit(X_train, y_train)

print(f"AUC-ROC do LDA (CV): {grid_lda.best_score_:.4f}")

# ==================================================================
# MODELO LINEAR 4: NAÏVE BAYES GAUSSINAO (GaussianNB)
# ==================================================================
print("\n--- Treinando Naïve Bayes ---")

pipe_nb = Pipeline([("scaler", StandardScaler()),("nb", GaussianNB())])

# Grade vazia porque o Naïve Bayes não precisa de ajuste de hiperparâmetros
grid_nb = GridSearchCV(pipe_nb, param_grid={}, cv=5, scoring="roc_auc", n_jobs=-1)
grid_nb.fit(X_train, y_train)

print(f"AUC-ROC do Naïve Bayes (CV): {grid_nb.best_score_:.4f}")

# ==================================================================
# SALVANDO OS RESULTADOS E OS MODELOS TREINADOS
# ==================================================================
# Dicionário com os melhores modelos extraídos do GridSearch (já treinados no X_train completo)
modelos_lineares = {
    "Regressao_Logistica": grid_lr.best_estimator_,
    "SVM_Linear": grid_svm.best_estimator_,
    "LDA": grid_lda.best_estimator_,
    "Naive_Bayes": grid_nb.best_estimator_,  
}

# Salvar o dicionário atualizado (Sobrescreve o arquivo antigo)
joblib.dump(modelos_lineares, "modelos_lineares_treinados.pkl")

# tabela de resumo dos resultados de treino (melhores parâmetros e AUC-ROC de validação cruzada)
resumo_treino = pd.DataFrame(
    {
        "Modelo": ["Regressão Logística", "SVM Linear", "LDA", "Naïve Bayes"],
        "Melhor_Parametro": [
            grid_lr.best_params_,
            grid_svm.best_params_,
            "N/A",
            "N/A",
        ],
        "AUC_ROC_CV_Treino": [
            grid_lr.best_score_,
            grid_svm.best_score_,
            grid_lda.best_score_,
            grid_nb.best_score_,  
        ],
    }
)
resumo_treino.to_csv("resumo_treino_lineares.csv", index=False)

print("\n" + "=" * 50)
print("Todos os 4 modelos salvos em arquivos pkl/csv.")
print("=" * 50)