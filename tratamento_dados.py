import pandas as pd
import numpy as np

# 1. Carregar a base de dados
df = pd.read_csv('credit_risk_dataset.csv')

print(f"Linhas iniciais: {df.shape[0]}")
print("Colunas disponíveis na base:", df.columns.tolist())

# 2. Fase 2: Tratamento e Limpeza (Duplicadas e Nulos)
df = df.drop_duplicates()
print(f"Linhas após remover duplicadas: {df.shape[0]}")

# Preencher nulos de forma segura (sem usar inplace=True encadeado)
for coluna in df.select_dtypes(include=['number']).columns:
    if df[coluna].isnull().sum() > 0:
        mediana = df[coluna].median()
        df[coluna] = df[coluna].fillna(mediana)
        print(f"Valores nulos da coluna {coluna} preenchidos com a mediana: {mediana}")

# 3. Fase 3: Feature Engineering (Coluna Calculada)
# Identificando o nome correto da coluna de empréstimo (loan_amnt ou loan_amount)
coluna_emprestimo = 'loan_amnt' if 'loan_amnt' in df.columns else 'loan_amount'

# Tratamento prévio para evitar divisão por zero
df['person_income'] = df['person_income'].replace(0, np.nan)
df['person_income'] = df['person_income'].fillna(df['person_income'].median())

# Criando a coluna calculada exigida pelo projeto
df['comprometimento_renda'] = (df[coluna_emprestimo] / df['person_income']) * 100

print("\n--- Amostra com a Nova Coluna Calculada ---")
print(df[[coluna_emprestimo, 'person_income', 'comprometimento_renda']].head())

# Salvando a base tratada para a próxima fase
df.to_csv('credit_risk_tratado.csv', index=False)
print("\nBase tratada salva com sucesso como 'credit_risk_tratado.csv'!")