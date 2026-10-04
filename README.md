# Pipeline Preditivo de Risco de Crédito

Repositório desenvolvido no âmbito da Situação de Aprendizagem (Projeto Avaliativo) do módulo de Machine Learning e Visão Computacional.

## 1. Contextualização e Problema de Negócio
O objetivo principal deste projeto é construir um modelo preditivo capaz de prever se um cliente se tornará inadimplente (`loan_status = 1`) ou se pagará o empréstimo em dia (`loan_status = 0`), mitigando perdas financeiras por calote para a instituição bancária.

## 2. Análise Exploratória de Dados (EDA) e Limpeza
* **Tratamento de Dados:** Remoção de duplicadas e preenchimento de valores omissos utilizando medianas para mitigar o impacto de *outliers*.
* **Feature Engineering:** Criação da coluna calculada obrigatória de comprometimento de renda: 
  $$\text{comprometimento\_renda} = (\text{loan\_amt} / \text{person\_income}) \times 100$$

## 3. Modelagem e Validação
* **Split e Balanceamento:** Divisão dos dados em treino e teste com estratificação (`stratify=y`) e aplicação de **SMOTE** exclusivamente no conjunto de treino para evitar vazamento de dados (*data leakage*).
* **Escalamento Seguro:** Utilização de `StandardScaler` ajustado estritamente nos dados de treino.
* **Algoritmos Testados:** K-Nearest Neighbors (KNN) e Árvore de Decisão (*Decision Tree*).

## 4. Resumo Executivo e Veredito de Negócio
* O modelo de Árvore de Decisão apresentou uma **acurácia de 91%** e um excelente equilíbrio na matriz de confusão.
* **Justificativa para a Diretoria:** O modelo minimiza de forma robusta os Falsos Negativos (aprovação de maus pagadores), protegendo o caixa do banco contra prejuízos financeiros diretos de forma sustentável e segura para produção.