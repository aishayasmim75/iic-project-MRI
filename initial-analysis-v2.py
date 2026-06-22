import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
import os

# Configuração estética e criação do diretório de saída
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6] 

pasta_saida = 'resultados-analise-inicial_projeto'
os.makedirs(pasta_saida, exist_ok=True)

# Cores e ordens padronizadas para o projeto inteiro
cores_padrao = {0: '#1f77b4', 1: '#d62728'}  # 0 = Azul (Controle), 1 = Vermelho (Alzheimer)
hue_ordem = [0, 1]

# lendo os arquivos-----------------------------------------------------------------------
df = pd.read_csv('PROJETO\dataset-combined.csv', sep=',')
#print('Formato do dataset: ', df.shape)
#print('Colunas do dataset: ', df.columns.tolist())

# olhando se os dados estão balanceados (AD=0 vs AD=1) 
print(df['AD'].value_counts())
print(df['GENDER'].value_counts())

# Multiplicando por 100 para voltar à escala em anos
df['Age'] = df['Age'] * 100
print("Idade média: ", df['Age'].mean())
print("Desvio padrão da idade: ", df['Age'].std())

# relação gênero-diagnóstico
print("--- Tabela de Contingência: Gênero vs Alzheimer ---")
print(pd.crosstab(df['GENDER'], df['AD'], normalize='index') * 100)

# Isolar ROIs
rois = df.columns[2:-3]
#--------------------------------------------------------------------------------------------
# =====================================================================
# VISUALIZAÇÕES INICIAIS 
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x='AD', hue='AD', ax=axes[0], palette=cores_padrao, hue_order=hue_ordem, legend=False)
axes[0].set_title('Distribuição da Variável Alvo (AD)')
axes[0].set_xlabel('Diagnóstico (0 = Controle, 1 = Alzheimer)')
axes[0].set_ylabel('Número de Observações')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Controle (0)', 'Alzheimer (1)'])

sns.countplot(data=df, x='GENDER', hue='GENDER', ax=axes[1], palette='Pastel1', legend=False)
axes[1].set_title('Distribuição por Gênero')
axes[1].set_xlabel('Gênero (0 ou 1)')
axes[1].set_ylabel('Número de Observações')
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, '01_distribuicao_ad_genero.png'), dpi=300)
plt.show()

# Histograma das idades
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='AD', bins=15, multiple='stack', palette=cores_padrao, hue_order=hue_ordem, edgecolor='white')
plt.title('Distribuição de Idade Estratificada por Alzheimer (AD)')
plt.xlabel('Idade (Anos)')
plt.ylabel('Número de Casos')
plt.legend(title='Grupo', labels=['Alzheimer (1)', 'Controle (0)'])
plt.savefig(os.path.join(pasta_saida, '02_histograma_idade.png'), dpi=300)
plt.show()

# =====================================================================
#  MATRIZ DE CORRELAÇÃO E FILTRAGEM
# (é esperado que haja muita correlação entre elas,
# # o que pode levar a problemas relacionados à multicolinearidade)
# =====================================================================
plt.figure(figsize=(12, 10))
matriz_corr = df[rois].corr(method='search' if 'search' in dir(df[rois].corr) else 'pearson')
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
sns.heatmap(
    matriz_corr, mask=mask, cmap='coolwarm', 
    vmin=-1, vmax=1, center=0, #força os limites da escala de cores
    xticklabels=False, yticklabels=False, cbar_kws={"shrink": 0.8, "label": "Pearson r"})
plt.title('Matriz de Correlação de Pearson entre as 91 ROIs Cerebrais', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, '03_heatmap_correlacao.png'), dpi=300)
plt.show()

# listando os nomes das ROIs mais correlacionadas 
# (correlação > 0.8 ou < -0.8, por exemplo)
limite_correlacao = 0.5
matriz_corr_abs = matriz_corr.abs()
mask_abs = np.triu(np.ones_like(matriz_corr_abs, dtype=bool))
correlacoes_unicas = matriz_corr_abs.where(~mask_abs)
pares_correlacionados = correlacoes_unicas.stack().reset_index()
pares_correlacionados.columns = ['ROI_1', 'ROI_2', 'Correlacao']
pares_correlacionados_filtrados = pares_correlacionados[pares_correlacionados['Correlacao'] > limite_correlacao]
pares_ordenados = pares_correlacionados_filtrados.sort_values(by='Correlacao', ascending=False)

# extraindo e Salvando os pares correlacionados em TXT
with open(os.path.join(pasta_saida, 'relatorio_correlacoes_fortes.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Pares de ROIs com correlação absoluta maior que {limite_correlacao}:\n")
    f.write(pares_ordenados.to_string(index=False))

print(f"Relatório de correlações salvo com sucesso em '{pasta_saida}'!")

# =======================================================================
# TESTE ESTATÍSTICO (teste t de student)
## testanddo se a diferença das médias entre o grupo controle e o grupo AD
## é estatisticamente significativa (p_value>0.05)
# =====================================================================
resultados = []
grupo_controle = df[df['AD'] == 0]
grupo_alzheimer = df[df['AD'] == 1]

for roi in rois:
    t_stat, p_val = stats.ttest_ind(grupo_controle[roi], grupo_alzheimer[roi], equal_var=False)
    resultados.append({'ROI': roi, 'p_value': p_val})

df_testes = pd.DataFrame(resultados).sort_values(by='p_value')

# Salvando ranking do Teste-T em CSV e TXT
df_testes.to_csv(os.path.join(pasta_saida, 'ranking_teste_t.csv'), index=False)
with open(os.path.join(pasta_saida, 'relatorio_teste_t_top10.txt'), 'w', encoding='utf-8') as f:
    f.write("As 10 ROIs com maior diferença estatística entre os grupos:\n")
    f.write(df_testes.head(10).to_string(index=False))

# ==========================================================================================
# Analisando as áreas neuroanatômicas de maior importância no estudo da AD
#-------------------------------------------------------------------------
# 1) Hipocampo → ROIs: 'Left-Hippocampus' e 'Right-Hippocampus'
# 2) Córtez Entorrinal → ROIs 'ctx-lh-entorhinal' e 'ctx-rh-entorhinal'
# 3) Amígdala → ROIs 'Left-Amygdala' e 'Right-Amygdala'
# 4) Giro Para-Hipocampal → ROIs 'ctx-lh-parahippocampal' e 'ctx-rh-parahippocampal'
# 5) Neocórtex Temporal (Giro Fusiforme e Giro Temporal Médio/Inferior)
# # 5.1) Giro Fusiforme → ROIs 'ctx-lh-fusiform' e 'ctx-rh-fusiform'
# # 5.2) Giro Temporal Médio → ROIs 'ctx-lh-middletemporal' e 'ctx-rh-middletemporal'
# # 5.3) Giro Temporal Inferior → ROIs 'ctx-lh-inferiortemporal' e 'ctx-rh-inferiortemporal'
# ==========================================================================================

# Queremos comparar a atrofia das diferentes regiões em pacientes saudáveis e em pacientes com AD
# KDE Plot:
    # x → volume estrutural da região da região do cérebro
    # y → densidade (concentração de pacientes)
# Se as curvas AD=0 e AD=1 tiverem muito overlap, isso quer dizer que pacientes saudáveis e
## pacientes com AD apresentam volume similar para essa região,
## logo, essa região não é um bom classificador da doença

# # Queremos ROIs com muita separabilidade entre os dois grupos

def calcular_overlap(df, coluna, target='AD'): #----------------------------------------------------
    # Separa os dados dos dois grupos
    dados_0 = df[df[target] == 0][coluna].dropna()
    dados_1 = df[df[target] == 1][coluna].dropna()
    # Se algum grupo estiver vazio, retorna 0
    if len(dados_0) == 0 or len(dados_1) == 0:
        return 0.0
    # Define o intervalo comum para avaliar as curvas (eixo X)
    xmin = min(dados_0.min(), dados_1.min())
    xmax = max(dados_0.max(), dados_1.max())
    x_eval = np.linspace(xmin, xmax, 1000)
    # Calcula a estimativa do Kernel de Densidade para cada grupo
    kde_0 = gaussian_kde(dados_0)(x_eval)
    kde_1 = gaussian_kde(dados_1)(x_eval)
    
    # O overlap é a área sob o mínimo das duas curvas de densidade
    # Usamos o método dos trapézios  para integrar a área
     # area_intersecao = np.trapz(np.minimum(kde_0, kde_1), x_eval) # função np.trapz ta desatualizada
    area_intersecao = trapezoid(np.minimum(kde_0, kde_1), x_eval)
    # Retorna o valor em porcentagem
    return area_intersecao * 100

# Dicionário mapeando as regiões da literatura para as ROIs exatas do dataset
regioes_analise = {
    "1. Hipocampo": ("Left-Hippocampus", "Right-Hippocampus"),
    "2. Córtex Entorrinal": ("ctx-lh-entorhinal", "ctx-rh-entorhinal"),
    # "3. Amígdala": ("Left-Amygdala", "Right-Amygdala"),
    #OBS: Por algum motivo a ROI 'Right-Amygdala' não consta no dataset???
    "3. Giro Para-hipocampal": ("ctx-lh-parahippocampal", "ctx-rh-parahippocampal"),
    "4. Giro Fusiforme (Neocórtex Temporal)": ("ctx-lh-fusiform", "ctx-rh-fusiform"),
    "5. Giro Temporal Médio (Neocórtex Temporal)": ("ctx-lh-middletemporal","ctx-rh-middletemporal"),
    "6. Giro Temporal Inferior (Neocórtex Temporal)": ("ctx-lh-inferiortemporal","ctx-rh-inferiortemporal")}
#------------------------------------------------------------------------------------------------

# PLOTS DE DENSIDADE (KDE)

for nome, (roi_esq, roi_dir) in regioes_analise.items():
    overlap_esq, overlap_dir = calcular_overlap(df, roi_esq), calcular_overlap(df, roi_dir)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Análise de Densidade Volumétrica: {nome}', fontsize=14, fontweight='bold')
    
    sns.kdeplot(data=df, x=roi_esq, hue='AD', fill=True, common_norm=False, palette=cores_padrao, hue_order=hue_ordem, ax=axes[0])
    axes[0].set_title(f'Hemisfério Esquerdo\nInterseção: {overlap_esq:.2f}%')
    
    sns.kdeplot(data=df, x=roi_dir, hue='AD', fill=True, common_norm=False, palette=cores_padrao, hue_order=hue_ordem, ax=axes[1])
    axes[1].set_title(f'Hemisfério Direito\nInterseção: {overlap_dir:.2f}%')
    
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, f'04_kde_{nome.lower().replace(" ", "_").replace(".", "")}.png'), dpi=200)
    plt.show()

# KDE Isolado Amígdala Esquerda
# (já que não tem dados sobre a amígdala direita no dataset)
overlap_amigdala = calcular_overlap(df, 'Left-Amygdala')
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='Left-Amygdala', hue='AD', fill=True, common_norm=False, palette=cores_padrao, hue_order=hue_ordem)
plt.title(f'Distribuição Volumétrica da Amígdala Esquerda\nInterseção das Curvas: {overlap_amigdala:.2f}%', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, '05_kde_amigdala_esquerda.png'), dpi=200)
plt.show()

# Fazendo o mapeamento também dos VENTRÍCULOS ------------------------------------------------
# Agora espero obter o comportamento inversos dos gráficos anteriores,
## já que há atropia do tecido cerebral, os ventrículos cerebrais aumentam
## HIDROCEFALIA EX-VÁCUO → O líquor (Líquido Cefalorraquidiano) passa a ocupar o 
### espaço extra deixado pelo tecido perdido, dilatando os ventrículos
#-----------------------------------------------------------------------------------------------

# KDE Ventrículos Pareados 
ventriculos_pareados = {
    "Ventrículo Lateral": ("Left-Lateral-Ventricle", "Right-Lateral-Ventricle"),
    "Ventrículo Lateral Inferior": ("Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent")}

for nome, (roi_esq, roi_dir) in ventriculos_pareados.items():
    overlap_esq, overlap_dir = calcular_overlap(df, roi_esq), calcular_overlap(df, roi_dir)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Análise de Densidade Volumétrica: {nome} (Hidrocefalia ex-vacuo)', fontsize=14, fontweight='bold')
    sns.kdeplot(data=df, x=roi_esq, hue='AD', fill=True, common_norm=False, palette=cores_padrao, hue_order=hue_ordem, ax=axes[0])
    axes[0].set_title(f'Hemisfério Esquerdo\nInterseção: {overlap_esq:.2f}%')
    sns.kdeplot(data=df, x=roi_dir, hue='AD', fill=True, common_norm=False, palette=cores_padrao, hue_order=hue_ordem, ax=axes[1])
    axes[1].set_title(f'Hemisfério Direito\nInterseção: {overlap_dir:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, f'06_kde_{nome.lower().replace(" ", "_")}.png'), dpi=200)
    plt.show()

# KDE Ventrículos Centrais:
# Análise de densidade volumétrica para as estruturas únicas (3° e 4° ventrículos)
# diferente dos ventrículos lateral e lat inferior trabalhados acima, o 3° e o 4° ventrículos
## não estão separados por hemisférios no dataset
overlap_3rd, overlap_4th = calcular_overlap(df, '3rd-Ventricle'), calcular_overlap(df, '4th-Ventricle')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Análise de Densidade Volumétrica: Ventrículos Centrais (Linha Média)', fontsize=14, fontweight='bold')
sns.kdeplot(data=df, x='3rd-Ventricle', hue='AD', fill=True, common_norm=False, palette=cores_padrao, hue_order=hue_ordem, ax=axes[0])
axes[0].set_title(f'Terceiro Ventrículo\nInterseção: {overlap_3rd:.2f}%')
sns.kdeplot(data=df, x='4th-Ventricle', hue='AD', fill=True, common_norm=False, palette=cores_padrao, hue_order=hue_ordem, ax=axes[1])
axes[1].set_title(f'Quarto Ventrículo\nInterseção: {overlap_4th:.2f}%')
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, '07_kde_ventriculos_centrais.png'), dpi=200)
plt.show()

# ===========================================================================================================
# GRÁFICOS DE DISPERSÃO:relação entre envelhecimento e atrofia das ROIs mais relevantes
# ===================================================================================================
# Selecionando as 6 ROIs chaves (4 que reduzem com a DA e 2 ventrículo que aumenta)
rois_dispersao = ['Left-Hippocampus', 'ctx-lh-entorhinal', 'Left-Inf-Lat-Vent', 'Left-Amygdala', 'Right-Hippocampus', 'Left-Lateral-Ventricle']

# Cálculo dinâmico de linhas e colunas para a grade de gráficos
n_rois = len(rois_dispersao)
n_cols = 2
n_rows = (n_rois + n_cols - 1) // n_cols  # Arredondamento para cima da divisão
# Criando a figura adaptável
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
axes = axes.flatten() # Transforma a matriz de eixos em uma lista linear

for i, roi in enumerate(rois_dispersao):
    # Plotando os dados e a reta de regressão para cada grupo separadamente
    for grupo, dados_grupo in df.groupby('AD'):
        sns.scatterplot(data=dados_grupo, x='Age', y=roi, color=cores_padrao[grupo], alpha=0.5, ax=axes[i],
                         label='Alzheimer (1)' if grupo == 1 and i == 0 else ('Controle (0)' if grupo == 0 and i == 0 else ""))
                        ### Adiciona legenda apenas no primeiro plot para não duplicar
        sns.regplot(data=dados_grupo, x='Age', y=roi, scatter=False, color=cores_padrao[grupo], ax=axes[i], ci=None)
             # Reta de tendência linear (ci=None remove a sombra do intervalo de confiança para limpar o gráfico)
    axes[i].set_title(f'Dinâmica do Envelhecimento: {roi}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Idade (Anos)')
    axes[i].set_ylabel('Volume Normalizado')

fig.axes[0].legend(title="Grupo (AD)", loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, '08_dispersao_idade_vs_atrofia.png'), dpi=200)
plt.show()


# ===========================================================================================================
# Análise de assimetria hemisférica dos ROIs pareadas
# se o grupo AD=1 apresentar resultados muito diferentes do grupo AD=0
## para rois específicas, isso pode significar que a diferença volumétrica dessa ROI de um hemisfério
### para o outro é um atributo importante na classificação 
#-----------------------------------------------------------------
# valores positivos → lado esquerdo é maior
# valores negativos → lado direito é maior
# --------------------------------------------------------------
# Métrica de assimetria → ÍNDICE DE ASSIMETRIA (AI)
## AI = \frac{Left - Right}{Left + Right} * 100
# ===================================================================================================

regioes_pareadas = {
    "Hipocampo": ("Left-Hippocampus", "Right-Hippocampus"),
    "Córtex Entorrinal": ("ctx-lh-entorhinal", "ctx-rh-entorhinal"),
    "Giro Para-hipocampal": ("ctx-lh-parahippocampal", "ctx-rh-parahippocampal"),
    "Ventrículo Lateral Inferior": ("Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent")}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (nome_regiao, (roi_esq, roi_dir)) in enumerate(regioes_pareadas.items()):
    coluna_ai = f'AI_{nome_regiao}'
    df[coluna_ai] = ((df[roi_esq] - df[roi_dir]) / (df[roi_esq] + df[roi_dir])) * 100
    
    # CORREÇÃO DA VISUALIZAÇÃO: showfliers=False remove outliers da visualização achatada
    sns.boxplot(data=df, x='AD', y=coluna_ai, hue='AD', palette=cores_padrao, hue_order=hue_ordem, ax=axes[i], legend=False, showfliers=False)
    
    axes[i].set_title(f'Índice de Assimetria: {nome_regiao}', fontsize=12, fontweight='bold')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['Controle (0)', 'Alzheimer (1)'])
    axes[i].set_xlabel('Grupo')
    axes[i].set_ylabel('Índice de Assimetria (%)')
    axes[i].axhline(0, color='gray', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, '09_boxplot_assimetria_limpo.png'), dpi=300)
plt.show()
#------------------------------------------------------------------------------------------
print(f"Fim de execução do pipeline: resultados e gráficos salvos na pasta '{pasta_saida}'.")