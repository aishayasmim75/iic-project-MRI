import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

d = pd.read_csv('PROJETO\dataset-combined.csv', sep=',')
print(d.head())
# 1. Visualizar as primeiras linhas
print("--- Primeiras 5 linhas do Dataset ---")
print(d.head())

# 2. Resumo estrutural (tipos de dados e valores nulos)
print("\n--- Informações sobre as colunas ---")
print(d.info())


print("n° atributos: ", len(d.columns))
print("n° amostras: ", d.shape[0])
print(d.columns[0:5])

# 3. Estatística descritiva de todas as variáveis numéricas (Média, Desvio Padrão, etc.)
print("\n--- Resumo Estatístico Geral ---")
print(d.describe())

# 4. Criando a "Tabela 1" do artigo científico: Comparando os grupos
#  a coluna alvo se chama 'AD' (0 = Não, 1 = Sim)
print("\n--- Perfil Demográfico por Diagnóstico ---")
perfil_demografico = d.groupby('AD')[['Age', 'GENDER']].describe()
print(perfil_demografico)

# Verificando a média de volume de algumas ROIs específicas por grupo
# Substitua 'Volume_Hipocampo' pelo nome exato da sua coluna
rois_interesse = ['Left-Hippocampus','Right-Hippocampus','ctx-rh-entorhinal'] 
if all(roi in d.columns for roi in rois_interesse):
    print("\n--- Média de Volume das ROIs por Diagnóstico ---")
    print(d.groupby('AD')[rois_interesse].mean())
