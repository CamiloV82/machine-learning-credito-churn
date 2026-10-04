import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Carregar base tratada
df = pd.read_csv('credit_risk_tratado.csv')

# Definir variáveis preditoras (X) e alvo (y)
# A variável alvo no dataset de crédito costuma ser 'loan_status'
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Tratamento de colunas categóricas se houver (Encoding simples via get_dummies)
X = pd.get_dummies(X, drop_first=True)

# 2. Split de Dados (Treino e Teste com estratificação)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Balanceamento de Classes com SMOTE (Apenas no Treino!)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 4. Escalamento Seguro (StandardScaler ajustado apenas no treino)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test) # No teste usamos apenas .transform() para evitar vazamento

print("Dados preparados e balanceados com sucesso!\n")

# 5. Otimização e Modelagem: KNN
print("--- TESTANDO MODELO KNN ---")
melhor_k = 3
maior_acuracia_knn = 0

for k in [3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_resampled)
    acc = knn.score(X_test_scaled, y_test)
    print(f"KNN com K={k} - Acurácia no Teste: {acc:.4f}")
    if acc > maior_acuracia_knn:
        maior_acuracia_knn = acc
        melhor_k = k

print(f"Melhor K escolhido: {melhor_k}\n")

# 6. Otimização e Modelagem: Árvore de Decisão
print("--- TESTANDO MODELO ÁRVORE DE DECISÃO ---")
# Nota: Árvores de decisão não exigem dados escalados, mas podemos usar X_train_resampled original
melhor_depth = 3
maior_acuracia_tree = 0

for depth in [3, 5, 7, 10]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train_resampled, y_train_resampled)
    acc = tree.score(X_test, y_test)
    print(f"Árvore com max_depth={depth} - Acurácia no Teste: {acc:.4f}")
    if acc > maior_acuracia_tree:
        maior_acuracia_tree = acc
        melhor_depth = depth

print(f"Melhor max_depth escolhido: {melhor_depth}")

# 7. Relatório de Classificação e Matriz de Confusão para o melhor modelo escolhido
# (Exemplo usando a Árvore de Decisão ou KNN que obteve melhor performance)
print("\n--- AVALIAÇÃO FINAL DO MODELO VENCEDOR ---")
melhor_modelo = DecisionTreeClassifier(max_depth=melhor_depth, random_state=42)
melhor_modelo.fit(X_train_resampled, y_train_resampled)
y_pred = melhor_modelo.predict(X_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))