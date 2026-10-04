import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregamento da Base de Dados (Opção A: Risco de Crédito)
# Certifique-se de que o arquivo 'credit_risk_dataset.csv' está na mesma pasta
df = pd.read_csv('credit_risk_dataset.csv')

# 2. Análise Descritiva e Estatística (Exigência da Fase 1)
print("--- Dimensões da Base (Linhas, Colunas) ---")
print(df.shape)

print("\n--- Tipos de Dados ---")
print(df.dtypes)

print("\n--- Sumário Estatístico Descritivo ---")
print(df.describe())

# 3. Gráficos Explicativos (Mínimo 3 gráficos exigidos)
# Gráfico 1: Distribuição da Variável Alvo (loan_status)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='loan_status')
plt.title('Desbalanceamento da Variável Alvo (Loan Status)')
plt.savefig('grafico_desbalanceamento.png')
plt.show()

# Gráfico 2: Histograma da Idade dos Clientes
plt.figure(figsize=(8, 4))
sns.histplot(df['person_age'], bins=30, kde=True, color='blue')
plt.title('Distribuição de Idade dos Clientes')
plt.savefig('grafico_idade.png')
plt.show()

# Gráfico 3: Mapa de Calor - Correlação de Pearson (apenas colunas numéricas)
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor - Correlação de Pearson')
plt.savefig('grafico_correlacao.png')
plt.show()