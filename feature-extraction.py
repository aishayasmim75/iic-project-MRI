import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# pegando os dados -------------------------------------------( )
df = pd.read_csv(r"PROJETO\dataset-combined.csv", sep=',')
df['Age'] = df['Age'] * 100
## Isolando as ROIs (Regiões de Interesse) dos demais preditores
rois = df.columns[2:-3]  

# ------------------------------
X = df[rois] # features = ROIs
y = df['AD'] # AD = 0 → controle, AD = 1 → Alzheimer
#print('X: ', X)
#print('y: ', y)
print("Formato de X (Deve ser 509, 91):", X.shape)
print("Formato de y (Deve ser 509,):", y.shape)


## TRAIN-TEST SPLITING → 70% teste e 30% treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
## PADRONIZAÇÃO 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# transformando de volta para DataFrame para facilitar a análise e visualização
X_train_scaled = pd.DataFrame(X_train_scaled, columns=rois)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=rois)

# =====================================================================
# METODOLOGIA 1: Seleção Automática via LASSO (Logistic Regression L1)
# =====================================================================
## OBS: não posso usar as funções LASSO ou LASSOCV do sklearn que usei para os exercícios
### pq elas são para regressão linear, e agora é um problema de classificação (AD = 0 ou 1)
# preciso usar a regularização L1 dentro de um modelo de regressão logística
#  (LogisticRegression com penalty='l1')
# onde o parâmetro C controla a força da regularização (C = 1/lambda)
# C muito pequeno → regularização forte → mais coeficientes zerados
# C muito grande → regularização fraca → menos coeficientes zerados
print("Iniciando a seleção de features com LASSO Logistic Regression...")

# definindo um lista de valores de C para testar
valores_C = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 10.0]
# criando um modelo com validação cruzada para escolher o melhor C (k=5 folds)
lasso_cv = LogisticRegressionCV(Cs=valores_C, cv=5, penalty='l1', solver='liblinear', n_jobs=-1, scoring='roc_auc')
# treino do modelo
lasso_cv.fit(X_train_scaled, y_train)
## RESULTADOS: -------------------------------------------------------------
print(f"Melhor valor de C encontrado: {lasso_cv.C_[0]}")
print(f"Equivalente a um Alpha de: {1 / lasso_cv.C_[0]:.4f}")

# plotando a curva de AUC para cada valor de C testado -----------
# extraindo os scores da validação cruzada para cada C
# lasso_cv.scores_[1] contém uma matriz de formato (n_folds, n_cs)
scores_por_fold = lasso_cv.scores_[1]
scores_medios = np.mean(scores_por_fold, axis=0)

#estilo do gráfico
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# plotando a linha com os pontos testados
# eixo x com escala logarítmica para melhor visualização
plt.semilogx(valores_C, scores_medios, marker="o", color="darkblue", linewidth=2)

# Destacar o melhor C encontrado
melhor_c = lasso_cv.C_[0]
melhor_score = np.max(scores_medios)
plt.axvline(x=melhor_c,color="red",linestyle="--",label=f"Melhor C ({melhor_c}) - AUC: {melhor_score:.3f}")
plt.title("LASSO \n Seleção de Hiperparâmetro: AUC-ROC Média vs. Parâmetro de Regularização (C)",fontsize=14,pad=15)
plt.xlabel("Força da Regularização (C) - Escala Logarítmica\n(Valores menores = Mais penalidade/Menos ROIs)",fontsize=12)
plt.ylabel("Área Sob a Curva ROC (AUC-ROC) Média", fontsize=12)
plt.xticks(valores_C, labels=[str(c) for c in valores_C])
plt.legend(fontsize=11, loc="lower right")
plt.tight_layout()
plt.show()

# Verificar quais ROIs foram selecionadas pelo melhor modelo
coeficientes_otimos = lasso_cv.coef_[0]
rois_selecionadas = X_train_scaled.columns[coeficientes_otimos != 0].tolist()

print(f"\nCom o C ótimo, o modelo reduziu de 91 para {len(rois_selecionadas)} ROIs.")
print("Regiões selecionadas:", rois_selecionadas)


# =====================================================================
# METODOLOGIA 2: Forward Selection
# =====================================================================
print("Iniciando a seleção de features com Forward Selection...")

# pipeline → encapsla a o processo de padronização + seleção de features + modelo final em um único objeto
    # selector → vao entrar com as 91 ROIs e vão sair com as x melhores ROIs escolhidas pelo Forward
    # classifier → modelo final que vai usar apenas as ROIs selecionadas para fazer a classificação (AD = 0 ou 1)
# GridSearchCV → para testar diferentes números de features selecionadas (ex: 5, 10, 15) e escolher o melhor com base na AUC-ROC

estimator = LogisticRegression(max_iter=1000, random_state=42) # avaliará cada ROI 
sfs = SequentialFeatureSelector(estimator,direction="forward",cv=5,n_jobs=-1)

pipeline_forward = Pipeline([("selector", sfs), ("classifier", estimator)])
grade_parametros = {"selector__n_features_to_select": [10, 20, 30, 40, 50, 60]} # testando diferentes números de features selecionadas

grid_forward = GridSearchCV(pipeline_forward, param_grid=grade_parametros, cv=5, scoring="roc_auc", n_jobs=-1)
grid_forward.fit(X_train_scaled, y_train)

# RESULTADOS: -------------------------------------------------------------
melhor_n_rois = grid_forward.best_params_["selector__n_features_to_select"]
print(f"Melhor número de ROIs para o Forward Selection: {melhor_n_rois}")
print(f"Maior AUC-ROC média alcançada: {grid_forward.best_score_:.3f}")

# Extrair quais foram as ROIs selecionadas pelo melhor modelo do GridSearch
# Acessamos o seletor de dentro do melhor pipeline encontrado
melhor_seletor = grid_forward.best_estimator_.named_steps["selector"]
forward_selected_rois = X_train_scaled.columns[melhor_seletor.get_support()].tolist()
    # get_support() retorna um array booleano indicando quais features foram selecionadas (True) e quais não foram (False)
    # usamos esse array para filtrar as colunas do DataFrame e obter os nomes das ROIs selecionadas

print(f"\nAs {melhor_n_rois} regiões selecionadas pelo Forward foram:")
print(forward_selected_rois)

# Extrair as pontuações médias para cada número de ROIs testado
scores_forward = grid_forward.cv_results_["mean_test_score"]
n_rois_testadas = grade_parametros["selector__n_features_to_select"]

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(
    n_rois_testadas, scores_forward, marker="s", color="darkgreen", linewidth=2
)
plt.axvline(
    x=melhor_n_rois,
    color="red",
    linestyle="--",
    label=f"Número Ótimo ({melhor_n_rois} ROIs) - AUC: {grid_forward.best_score_:.3f}",
)

# Customização
plt.title(
    "Forward Selection: AUC-ROC Média vs. Número de Variáveis Selecionadas",
    fontsize=14,
    pad=15,
)

plt.xlabel("Quantidade de Regiões de Interesse (ROIs)", fontsize=12)
plt.ylabel("Área Sob a Curva ROC (AUC-ROC) Média", fontsize=12)
plt.xticks(n_rois_testadas)
plt.legend(fontsize=11, loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()


# =====================================================================
# Salvando as ROIs selecionadas por cada método
# =====================================================================
# --- SALVANDO AS ROIs DO LASSO ---
with open("rois_selecionadas_lasso.txt", "w") as f:
    for roi in rois_selecionadas:
        f.write(f"{roi}\n")

# --- SALVANDO AS ROIs DO FORWARD ---
with open("rois_selecionadas_forward.txt", "w") as f:
    for roi in forward_selected_rois:
        f.write(f"{roi}\n")

print("Arquivos .txt salvos com sucesso.")



# =====================================================================
# COMPARAÇÃO DOS MÉTODOS E ESCOLHA DO MELHOR SUBCONJUNTO FINAL DE ROIs
# =====================================================================
# cálculo da interseção (ROIs selecionadas por ambos os métodos)
intersecao_rois = list(set(rois_selecionadas).intersection(set(forward_selected_rois)))

# resgate das melhores pontuações de cada método
melhor_auc_lasso = lasso_cv.scores_[1].mean(axis=0).max()
 #[1] acessa a matriz de scores da classe AD=1 (alzheimer), que tem formato (n_folds, n_Cs)
 # .mean(axis=0) calcula a média da AUC-ROC para cada valor de C testado, resultando em um array de tamanho n_Cs
 # .max() pega o valor máximo dessa média, que corresponde ao melhor C encontrado
melhor_auc_forward = grid_forward.best_score_
 #GridSearchCV já tem um atributo queretorna a melhor pontuação média (AUC-ROC) 
 # # alcançada para o melhor número de ROIs selecionadas

# construindo o relatório para a tomada de decisão
print("\n" + "="*50)
print("       RELATÓRIO COMPARATIVO DE SELEÇÃO DE ROIs")
print("="*50)
print(f"1. METODOLOGIA LASSO LOGÍSTICO:")
print(f"   - Quantidade de ROIs selecionadas: {len(rois_selecionadas)}")
print(f"   - Maior AUC-ROC obtida no Treino (CV): {melhor_auc_lasso:.4f}")

print(f"\n2. METODOLOGIA FORWARD SELECTION:")
print(f"   - Quantidade de ROIs selecionadas: {len(forward_selected_rois)}")
print(f"   - Maior AUC-ROC obtida no Treino (CV): {melhor_auc_forward:.4f}")

print(f"\n3. CONSENSO DOS ALGORITMOS (Interseção):")
print(f"   - Quantidade de ROIs em comum: {len(intersecao_rois)}")
print(f"   - Regiões de Consenso: {intersecao_rois}")
print("="*50)

# -----------------------------------------------------------------
# lógica da decisão final automática → escolha de um subconjunto final único
# CRITÉRIO: MAIOR AUC-ROC MÉDIA ENTRE OS DOIS MÉTODOS
# CASO DE EMPATE → PRIORIDADE PARA O MÉTODO COM MENOS ROIs (MAIOR SIMPLICIDADE→ PARCIMONIA)
diferenca_auc = abs(melhor_auc_lasso - melhor_auc_forward)

if diferenca_auc < 0.005:
    print("\n Decisão por Parcimônia: As performances são estatisticamente equivalentes.")
    if len(rois_selecionadas) < len(forward_selected_rois):
        subconjunto_final = rois_selecionadas
        metodo_vencedor = "Lasso (Modelo mais enxuto)"
    else:
        subconjunto_final = forward_selected_rois
        metodo_vencedor = "Forward Selection (Modelo mais enxuto)"
else:
    print("\n Decisão por Performance: Um método superou o outro significativamente.")
    if melhor_auc_lasso > melhor_auc_forward:
        subconjunto_final = rois_selecionadas
        metodo_vencedor = "Lasso (Maior AUC)"
    else:
        subconjunto_final = forward_selected_rois
        metodo_vencedor = "Forward Selection (Maior AUC)"

print(f" O subconjunto definitivo escolhido para o restante do projeto foi o do: {metodo_vencedor}")
print(f" Quantidade de ROIs no subconjunto final: {len(subconjunto_final)}")
print(f" ROIs do subconjunto final: {subconjunto_final}")
# -----------------------------------------------------------------
# salvando o subconjunto final em um arquivo .txt
with open("rois_selecionadas_definitivo.txt", "w") as f:
    for roi in subconjunto_final:
        f.write(f"{roi}\n")
print("Arquivo 'rois_selecionadas_definitivo.txt' salvo!")