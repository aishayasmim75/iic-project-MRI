import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

print(f"Dados carregados com sucesso. Treinando modelos não-lineares com {X_train.shape[1]} ROIs.")


# ==================================================================
# MODELO 1 - SVM POLINOMIAL E RADIAL (Support Vector Machine)
# ==================================================================
print("\n---Treinando o modelo com SVM Não-Linear (Polinomial e RBF) ---")

pipe_svm_kernel = Pipeline([("scaler", StandardScaler()), ("svm", SVC(random_state=42, class_weight='balanced'))])

# grade separando kernel=poly e kernel=rbf
## usando só um dicionário para ambos o sklearn tenta passar o parâmetro degree no kernel RBF e da erro
param_grid_svm = [
    # Cenário A: Kernel Radial (RBF)
    {
        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1.0, 10.0, 100.0],
        "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
    # Cenário B: Kernel Polinomial
    {
        "svm__kernel": ["poly"],
        "svm__C": [0.1, 1.0, 10.0],
        "svm__degree": [2, 3, 4, 5],  # Testando polinômios de graus 2,3,4,5
        "svm__gamma": ["scale"],
    },
]

#validação cruzada  com k=5 avaliando por AUC-ROC
grid_svm_kernel = GridSearchCV(pipe_svm_kernel, param_grid_svm, cv=5, scoring="roc_auc", n_jobs=-1)
grid_svm_kernel.fit(X_train, y_train)

print(f"Melhor configuração de SVM encontrada:")
print(grid_svm_kernel.best_params_)
print(f"Maior AUC-ROC obtida (CV): {grid_svm_kernel.best_score_:.4f}")

# =====================================================================
#  MODELO NÃO-LINEAR 2: REDE NEURAL (Multi-Layer Perceptron)
# =====================================================================
print("\n--- Otimizando Rede Neural Artificial (MLP) ---")


pipe_mlp = Pipeline(
    [("scaler", StandardScaler()),
     ("mlp",MLPClassifier(max_iter=1500, early_stopping=True, random_state=42))])


# Grade de arquiteturas de neurônios e forças de regularização (alpha)
param_grid_mlp = {
    # Testando uma camada oculta com 30 neurônios OU duas camadas ocultas com (20, 10) neurônios
    "mlp__hidden_layer_sizes": [(30,), (20, 10)],
    "mlp__activation": ["tanh", "relu"],
    "mlp__alpha": [0.0001, 0.001, 0.01], #parâmetro de regularização de rede
}

grid_mlp = GridSearchCV(pipe_mlp, param_grid_mlp, cv=5, scoring="roc_auc", n_jobs=-1)
grid_mlp.fit(X_train, y_train)

print(f"Melhor configuração de MLP encontrada:")
print(grid_mlp.best_params_)
print(f"Maior AUC-ROC obtida (CV): {grid_mlp.best_score_:.4f}")


# =====================================================================
# SALVAMENTO DOS RESULTADOS E MODELOS 
# =====================================================================
# Guardar os modelos vencedores desta rodada
modelos_nao_lineares = {
    "SVM_Kernel_Vencedor": grid_svm_kernel.best_estimator_,
    "Rede_Neural_MLP": grid_mlp.best_estimator_,
}

# Salvando 
joblib.dump(modelos_nao_lineares, "modelos_nao_lineares_treinados.pkl")

# Gerando a tabela resumo 
resumo_treino_nl = pd.DataFrame(
    {
        "Modelo": [
            f"SVM ({grid_svm_kernel.best_params_['svm__kernel']})",
            "Rede Neural MLP",],
        "Melhor_Parametro": [grid_svm_kernel.best_params_, grid_mlp.best_params_],
        "AUC_ROC_CV_Treino": [grid_svm_kernel.best_score_, grid_mlp.best_score_],
    }
)
resumo_treino_nl.to_csv("resumo_treino_nao_lineares.csv", index=False)

print("\n" + "=" * 55)
print("Modelos não-lineares salvos em arquivos pkl/csv.")
print("=" * 55)