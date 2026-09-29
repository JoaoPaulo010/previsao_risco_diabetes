# -*- coding: utf-8 -*-
"""
Modelo Preditivo para Identificação Precoce do Risco de Diabetes
Disciplina: Data Mining and Predictive Analytics - UniFECAF

Como executar (script local):
    1. Coloque o arquivo diabetes_prediction_dataset.csv na mesma pasta deste script.
    2. Instale as dependências: pip install pandas numpy matplotlib scikit-learn xgboost joblib
    3. Rode: python projeto_diabetes.py

Como executar (Google Colab / Jupyter):
    Cole o código em uma célula (ou divida por seções) e execute. No Colab
    será aberta uma janela para enviar o CSV; no Jupyter, deixe o CSV na
    mesma pasta do notebook.

Os gráficos gerados são salvos na pasta "graficos/" para uso no relatório.
"""


# ======================================================================
#
# 1. ENTENDIMENTO DO NEGÓCIO
#
# Uma rede nacional de clínicas deseja identificar automaticamente
# pacientes com maior probabilidade de desenvolver diabetes, para
# priorizar exames, orientações e programas de acompanhamento
# preventivo. Hoje essa identificação é feita de forma manual.
#
# Objetivo do projeto: construir e comparar modelos de classificação
# supervisionada que estimem, a partir de dados clínicos e
# demográficos, a probabilidade de um paciente ter diabetes.
#
# Critério de sucesso: em triagem de saúde, um falso negativo
# (paciente com diabetes que o modelo não identifica) é mais grave do
# que um falso positivo (paciente saudável que é chamado para um exame
# extra). Por isso, além da AUC, daremos atenção especial ao Recall
# (Sensibilidade).
# ======================================================================

# ======================================================================
#
# 0. IMPORTAÇÃO DAS BIBLIOTECAS
#
# No Google Colab todas as bibliotecas abaixo já vêm instaladas.
# ======================================================================

# Manipulação de dados
import pandas as pd
import numpy as np

# Visualização
import matplotlib.pyplot as plt

# Separação dos dados, validação cruzada e pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Algoritmos de classificação
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Métricas de avaliação
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")

# Semente fixa para garantir reprodutibilidade dos resultados
RANDOM_STATE = 42


# ======================================================================
#
# TRADUÇÃO PARA EXIBIÇÃO
#
# A base original está em inglês. Os dados são mantidos como estão
# (para não alterar nenhum resultado), e estes dicionários traduzem
# apenas o que é mostrado no terminal e nos gráficos.
# ======================================================================

NOMES_PT = {
    "gender": "gênero",
    "age": "idade",
    "hypertension": "hipertensão",
    "heart_disease": "doença_cardíaca",
    "smoking_history": "tabagismo",
    "bmi": "imc",
    "HbA1c_level": "hemoglobina_glicada",
    "blood_glucose_level": "glicose",
    "diabetes": "diabetes",
    "bmi_category": "faixa_imc",
    "hba1c_category": "faixa_hemoglobina",
    "glucose_category": "faixa_glicose",
    "age_group": "faixa_etária",
    "comorbidades": "comorbidades",
}

VALORES_PT = {
    "Female": "Feminino",
    "Male": "Masculino",
    "Other": "Outro",
    "never": "nunca fumou",
    "No Info": "Sem informação",
    "current": "fumante atual",
    "former": "ex-fumante",
    "ever": "já fumou",
    "not current": "não fuma atualmente",
    "Pre-diabetes": "Pré-diabetes",
}

SIM_NAO = {0: "Não", 1: "Sim"}

ESTATISTICAS_PT = {
    "count": "contagem", "unique": "únicos", "top": "mais frequente",
    "freq": "frequência", "mean": "média", "std": "desvio padrão",
    "min": "mínimo", "25%": "25%", "50%": "mediana", "75%": "75%", "max": "máximo",
}

TIPOS_PT = {"object": "texto", "str": "texto", "string": "texto",
            "float64": "decimal", "int64": "inteiro"}


def traduzir_tabela(tabela):
    """Devolve uma cópia da tabela com colunas e categorias em português."""
    t = tabela.rename(columns=NOMES_PT)
    for col in t.columns:
        if not pd.api.types.is_numeric_dtype(t[col]):
            t[col] = t[col].replace(VALORES_PT)
    return t


def rotulo_pt(col, valor):
    """Traduz o valor de uma categoria (0/1 vira Não/Sim nas colunas binárias)."""
    if col in ("hypertension", "heart_disease"):
        return SIM_NAO.get(valor, valor)
    return VALORES_PT.get(valor, valor)


def mostrar_taxa_por(col, ordenar=False):
    """Mostra a taxa de diabetes (%) em cada categoria de uma variável."""
    taxa = df.groupby(col)["diabetes"].mean().mul(100).round(2)
    if ordenar:
        taxa = taxa.sort_values(ascending=False)
    tabela = pd.DataFrame({
        NOMES_PT[col]: [rotulo_pt(col, v) for v in taxa.index],
        "taxa de diabetes (%)": taxa.values
    })
    print(f"\nTaxa de diabetes por {NOMES_PT[col]}:")
    print(tabela.to_string(index=False))


def mostrar_contagem(col):
    """Mostra quantos pacientes há em cada categoria de uma variável."""
    contagem_col = df[col].value_counts()
    tabela = pd.DataFrame({
        NOMES_PT[col]: [rotulo_pt(col, v) for v in contagem_col.index],
        "quantidade": contagem_col.values
    })
    print(tabela.to_string(index=False))


# ======================================================================
#
# 2. ENTENDIMENTO DOS DADOS
#
# 2.1 CARREGAMENTO DA BASE
# No Colab, será aberta uma janela para enviar o arquivo
# diabetes_prediction_dataset.csv. Em um Jupyter local, basta deixar o
# arquivo na mesma pasta do notebook.
# ======================================================================

from pathlib import Path
import sys

# Pasta do projeto: a pasta do script (.py) ou, no Jupyter/Colab, a pasta atual
try:
    pasta_script = Path(__file__).resolve().parent
except NameError:
    pasta_script = Path.cwd()

nome_arquivo = pasta_script / "diabetes_prediction_dataset.csv"

# No Colab, se o CSV ainda não foi enviado, abre a janela de upload
try:
    from google.colab import files
    if not nome_arquivo.exists():
        files.upload()
except ImportError:
    pass

if not nome_arquivo.exists():
    # Tenta encontrar qualquer CSV parecido na pasta (ex.: nome com ".csv.csv" ou "(1)")
    candidatos = list(pasta_script.glob("diabetes*.csv*"))
    if candidatos:
        nome_arquivo = candidatos[0]
    else:
        print(f"ERRO: arquivo 'diabetes_prediction_dataset.csv' não encontrado em:\n  {pasta_script}")
        print("Arquivos encontrados nessa pasta:")
        for f in pasta_script.iterdir():
            print("  -", f.name)
        print("Baixe a base no Kaggle, EXTRAIA o .zip e coloque o CSV nessa pasta.")
        sys.exit(1)

print("Lendo arquivo:", nome_arquivo)
df = pd.read_csv(nome_arquivo)

# Pasta onde os gráficos serão salvos (para inserir no relatório)
pasta_graficos = pasta_script / "graficos"
pasta_graficos.mkdir(exist_ok=True)


def salvar_e_mostrar(nome):
    """Salva a figura atual em graficos/<nome>.png e depois a exibe."""
    plt.savefig(pasta_graficos / f"{nome}.png", dpi=150, bbox_inches="tight")
    plt.show()


print("\nPrimeiros registros da base:")
print(traduzir_tabela(df.head()).to_string())


# ======================================================================
#
# 2.2 ANÁLISE INICIAL: DIMENSÕES, TIPOS E ESTATÍSTICAS
#
# ======================================================================

print(f"\nDimensões da base: {df.shape[0]} linhas e {df.shape[1]} colunas")

# Resumo das colunas (equivalente ao df.info(), em português)
resumo_colunas = pd.DataFrame({
    "coluna": [NOMES_PT[c] for c in df.columns],
    "valores não nulos": df.notnull().sum().values,
    "tipo": [TIPOS_PT.get(str(t), str(t)) for t in df.dtypes]
})
print("\nColunas da base:")
print(resumo_colunas.to_string(index=False))


# Estatísticas descritivas de todas as colunas (numéricas e categóricas)
estatisticas = (
    df.describe(include="all").T
    .rename(index=NOMES_PT, columns=ESTATISTICAS_PT)
)
estatisticas["mais frequente"] = estatisticas["mais frequente"].replace(VALORES_PT)
print("\nEstatísticas descritivas:")
print(estatisticas.to_string(na_rep="-"))


# ======================================================================
#
# 2.3 VALORES AUSENTES
# Verificamos os nulos "explícitos" (NaN) e também valores que
# representam ausência de informação de forma disfarçada, como a
# categoria "No Info" (Sem informação) em smoking_history.
# ======================================================================

nulos = df.isnull().sum()
print("\nValores ausentes por coluna:")
print(pd.DataFrame({
    "coluna": [NOMES_PT[c] for c in nulos.index],
    "valores ausentes": nulos.values
}).to_string(index=False))
print("\nTotal de valores ausentes:", nulos.sum())

# Ausência "disfarçada": pacientes sem informação sobre tabagismo
qtd_no_info = (df["smoking_history"] == "No Info").sum()
print(f"\nPacientes sem informação de tabagismo: {qtd_no_info} "
      f"({qtd_no_info / len(df):.1%} da base)")


# ======================================================================
#
# 2.4 REGISTROS DUPLICADOS
#
# ======================================================================

print("Quantidade de registros duplicados:", df.duplicated().sum())


# ======================================================================
#
# 2.5 DISTRIBUIÇÃO DA VARIÁVEL ALVO (DIABETES)
# É importante verificar se as classes estão balanceadas, pois isso
# influencia a escolha das métricas e das técnicas de modelagem.
# ======================================================================

contagem = df["diabetes"].value_counts()
percentual = df["diabetes"].value_counts(normalize=True).mul(100).round(2)

print("\nDistribuição da variável alvo:")
print(pd.DataFrame({
    "diabetes": [SIM_NAO[v] for v in contagem.index],
    "quantidade": contagem.values,
    "percentual (%)": percentual.values
}).to_string(index=False))

contagem.plot(kind="bar", figsize=(7, 5))
plt.title("Distribuição da variável Diabetes")
plt.xlabel("Diabetes (0 = Não, 1 = Sim)")
plt.ylabel("Quantidade de pacientes")
plt.xticks(rotation=0)
salvar_e_mostrar("01_distribuicao_alvo")


# ======================================================================
#
# 2.6 VARIÁVEIS CATEGÓRICAS
#
# ======================================================================

print("\nGênero:")
mostrar_contagem("gender")

print("\nHistórico de tabagismo:")
mostrar_contagem("smoking_history")


# Taxa de diabetes (%) em cada categoria: mostra quais grupos têm maior risco
for col in ["gender", "smoking_history", "hypertension", "heart_disease"]:
    mostrar_taxa_por(col, ordenar=True)


# ======================================================================
#
# 2.7 VARIÁVEIS NUMÉRICAS: DISTRIBUIÇÃO E OUTLIERS
#
# ======================================================================

numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

df[numeric_cols].rename(columns=NOMES_PT).hist(figsize=(12, 8), bins=30)
plt.suptitle("Distribuição das variáveis numéricas")
plt.tight_layout()
salvar_e_mostrar("02_histogramas")

print("\nEstatísticas das variáveis numéricas:")
print(df[numeric_cols].describe().T.rename(index=NOMES_PT, columns=ESTATISTICAS_PT).to_string())


# Boxplots para identificar possíveis outliers
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, numeric_cols):
    ax.boxplot(df[col].dropna())
    ax.set_title(f"Boxplot - {NOMES_PT[col]}")
plt.tight_layout()
salvar_e_mostrar("03_boxplots_outliers")


# ======================================================================
#
# 2.8 COMPARAÇÃO ENTRE PACIENTES COM E SEM DIABETES
#
# ======================================================================

# Média das variáveis numéricas em cada grupo
medias = df.groupby("diabetes")[numeric_cols].mean().round(2)
medias = medias.rename(columns=NOMES_PT, index=SIM_NAO)
medias.index.name = "diabetes"
print("\nMédia das variáveis numéricas (sem e com diabetes):")
print(medias.to_string())


# Boxplots por classe: mostram visualmente quais variáveis separam melhor os grupos
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, numeric_cols):
    dados = [df.loc[df["diabetes"] == 0, col], df.loc[df["diabetes"] == 1, col]]
    ax.boxplot(dados)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Não", "Sim"])
    ax.set_title(NOMES_PT[col])
plt.suptitle("Variáveis numéricas por presença de diabetes")
plt.tight_layout()
salvar_e_mostrar("04_boxplots_por_classe")


# Correlação entre as variáveis numéricas e o alvo
corr = df[numeric_cols + ["hypertension", "heart_disease", "diabetes"]].corr()
corr = corr.rename(index=NOMES_PT, columns=NOMES_PT)

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Matriz de correlação")
plt.tight_layout()
salvar_e_mostrar("05_matriz_correlacao")


# ======================================================================
# Problemas identificados na base:
# - Registros duplicados, que podem enviesar o treinamento e inflar as
#   métricas de teste.
# - Categoria "Other" em gender com pouquíssimos registros,
#   insuficiente para o modelo aprender.
# - "No Info" em smoking_history, que na prática é um valor ausente.
#   Optamos por mantê-la como uma categoria própria, pois remover esses
#   registros descartaria uma parte grande da base, e a própria ausência
#   de informação pode carregar algum padrão.
# - Valores extremos: IMC chega a ~96 e a idade começa em 0,08 (bebês).
#   Mantivemos esses registros, pois são clinicamente possíveis
#   (obesidade mórbida e pacientes pediátricos), e os modelos de árvore
#   são robustos a outliers.
# - Desbalanceamento da variável alvo: a classe positiva (diabetes) é
#   minoritária, o que torna a Acurácia uma métrica enganosa e exige
#   tratamento na modelagem.
# ======================================================================

# ======================================================================
#
# 3. PREPARAÇÃO DOS DADOS
#
# 3.1 LIMPEZA
#
# ======================================================================

# Remoção de registros duplicados
print(f"\nAntes da limpeza: {df.shape[0]} linhas")
df = df.drop_duplicates().reset_index(drop=True)

# Remoção da categoria 'Other' em gender (poucos registros)
df = df[df["gender"] != "Other"].reset_index(drop=True)
print(f"Após a limpeza:   {df.shape[0]} linhas")


# ======================================================================
#
# 3.2 CRIAÇÃO DE ATRIBUTOS DERIVADOS
# Transformamos variáveis contínuas em faixas com significado clínico.
# Isso ajuda na interpretação dos resultados e pode facilitar o
# aprendizado dos modelos:
#
# - bmi_category: classificação do IMC segundo a OMS.
# - hba1c_category: faixas de hemoglobina glicada segundo a American
#   Diabetes Association (normal < 5,7; pré-diabetes 5,7 a 6,4; diabetes
#   ≥ 6,5).
# - glucose_category: faixas de glicemia (normal < 140; alterada 140 a
#   199; elevada ≥ 200).
# - age_group: faixas etárias.
# - comorbidades: soma de hipertensão e doença cardíaca (0, 1 ou 2).
# ======================================================================

df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, np.inf],
    labels=["Abaixo do peso", "Normal", "Sobrepeso", "Obesidade"],
    right=False
).astype(str)

df["hba1c_category"] = pd.cut(
    df["HbA1c_level"],
    bins=[0, 5.7, 6.5, np.inf],
    labels=["Normal", "Pre-diabetes", "Diabetes"],
    right=False
).astype(str)

df["glucose_category"] = pd.cut(
    df["blood_glucose_level"],
    bins=[0, 140, 200, np.inf],
    labels=["Normal", "Alterada", "Elevada"],
    right=False
).astype(str)

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 18, 40, 60, np.inf],
    labels=["0-17", "18-39", "40-59", "60+"],
    right=False
).astype(str)

df["comorbidades"] = df["hypertension"] + df["heart_disease"]

print("\nBase com os atributos derivados:")
print(traduzir_tabela(df.head()).to_string())


# Verificando se os novos atributos têm relação com o alvo (taxa de diabetes em %)
for col in ["bmi_category", "hba1c_category", "glucose_category", "age_group", "comorbidades"]:
    mostrar_taxa_por(col)


# ======================================================================
#
# 3.3 SEPARAÇÃO ENTRE TREINO E TESTE
# Usamos 80% dos dados para treino e 20% para teste. O parâmetro
# stratify=y garante que a proporção de pacientes com diabetes seja a
# mesma nos dois conjuntos.
# ======================================================================

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTreino: {X_train.shape[0]} pacientes | Teste: {X_test.shape[0]} pacientes")
print("Proporção de diabetes no treino: {:.2%}".format(y_train.mean()))
print("Proporção de diabetes no teste:  {:.2%}".format(y_test.mean()))


# ======================================================================
#
# 3.4 PIPELINE DE PRÉ-PROCESSAMENTO
#
# - Variáveis numéricas: padronizadas com StandardScaler (média 0 e
#   desvio padrão 1). Modelos baseados em árvores não são sensíveis à
#   escala, mas a padronização mantém o pipeline pronto caso outros
#   algoritmos sejam testados e não prejudica os resultados.
# - Variáveis categóricas: transformadas com OneHotEncoder, que cria
#   uma coluna binária para cada categoria. handle_unknown="ignore"
#   evita erros caso surja uma categoria nova em produção.
#
# O pré-processamento fica dentro de um Pipeline, então ele é ajustado
# apenas com os dados de treino, evitando vazamento de informação
# (*data leakage*) para o conjunto de teste.
# ======================================================================

numeric_features = [
    "age", "hypertension", "heart_disease", "bmi",
    "HbA1c_level", "blood_glucose_level", "comorbidades"
]

categorical_features = [
    "gender", "smoking_history", "bmi_category",
    "hba1c_category", "glucose_category", "age_group"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# ======================================================================
#
# 4. MODELAGEM
#
# Os três modelos tratam o desbalanceamento dando mais peso à classe
# minoritária:
# - Decision Tree e Random Forest: class_weight="balanced";
# - XGBoost: scale_pos_weight = (nº de negativos / nº de positivos) no
#   treino.
#
# Essa abordagem foi escolhida no lugar da criação de dados sintéticos
# (como o SMOTE) por ser mais simples, não gerar pacientes artificiais
# e ser suportada nativamente pelos três algoritmos.
# ======================================================================

# ======================================================================
#
# 4.1 DECISION TREE
# Modelo de árvore única, muito interpretável. Limitamos a
# profundidade (max_depth=6) para evitar *overfitting* (a árvore
# "decorar" os dados de treino).
# ======================================================================

decision_tree = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

decision_tree.fit(X_train, y_train)

y_pred_dt = decision_tree.predict(X_test)
y_prob_dt = decision_tree.predict_proba(X_test)[:, 1]


# ======================================================================
#
# 4.2 RANDOM FOREST
# Conjunto (*ensemble*) de muitas árvores treinadas em amostras
# diferentes dos dados (*bagging*). A previsão final é a votação das
# árvores, o que reduz a variância e o overfitting.
# ======================================================================

random_forest = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,          # evita folhas com pouquíssimos pacientes
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)
y_prob_rf = random_forest.predict_proba(X_test)[:, 1]


# ======================================================================
#
# 4.3 XGBOOST
# Algoritmo de *boosting*: as árvores são construídas em sequência, e
# cada nova árvore corrige os erros das anteriores. Costuma ter
# excelente desempenho em dados tabulares.
# ======================================================================

# Peso da classe positiva para compensar o desbalanceamento
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nPeso da classe com diabetes no XGBoost (scale_pos_weight) = {ratio:.2f}")

xgboost_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=ratio,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

xgboost_model.fit(X_train, y_train)

y_pred_xgb = xgboost_model.predict(X_test)
y_prob_xgb = xgboost_model.predict_proba(X_test)[:, 1]


# ======================================================================
#
# 5. AVALIAÇÃO DOS MODELOS
#
# Métricas utilizadas:
# - Acurácia (Accuracy): proporção de acertos totais. Enganosa em bases
#   desbalanceadas.
# - Precisão (Precision): dos pacientes que o modelo apontou como
#   diabéticos, quantos realmente são.
# - Sensibilidade (Recall): dos pacientes realmente diabéticos,
#   quantos o modelo encontrou. É a métrica mais importante para
#   triagem, pois mede os casos que *não* deixamos escapar.
# - F1: média harmônica entre Precisão e Sensibilidade.
# - AUC (Área sob a curva ROC): capacidade do modelo de separar as
#   classes considerando todos os limiares de decisão possíveis (1,0 =
#   perfeito; 0,5 = aleatório).
# ======================================================================

METRICAS = ["Acurácia", "Precisão", "Sensibilidade", "F1", "AUC"]


def avaliar_modelo(nome, y_true, y_pred, y_prob):
    """Calcula as principais métricas de classificação para um modelo."""
    return {
        "Modelo": nome,
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precisão": precision_score(y_true, y_pred, zero_division=0),
        "Sensibilidade": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob)
    }


# Dicionário central com os modelos e suas previsões (evita repetição de código)
modelos = {
    "Decision Tree":  (decision_tree, y_pred_dt, y_prob_dt),
    "Random Forest": (random_forest, y_pred_rf, y_prob_rf),
    "XGBoost":            (xgboost_model, y_pred_xgb, y_prob_xgb),
}


# Métricas individuais de cada modelo (conjunto de teste)
for nome, (_, y_pred, y_prob) in modelos.items():
    metricas = avaliar_modelo(nome, y_test, y_pred, y_prob)
    print(f"\n===== {nome} =====")
    for chave in METRICAS:
        print(f"{chave:<14}: {metricas[chave]:.4f}")


# ======================================================================
#
# 5.1 MATRIZES DE CONFUSÃO
#
# ======================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

print()
for ax, (nome, (_, y_pred, _)) in zip(axes, modelos.items()):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Não Diabetes", "Diabetes"]
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Matriz de Confusão - {nome}")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")

    tn, fp, fn, tp = cm.ravel()
    print(f"{nome}: VN={tn} | FP={fp} | FN={fn} (diabéticos não identificados) | VP={tp}")

plt.tight_layout()
salvar_e_mostrar("06_matrizes_confusao")


# ======================================================================
#
# 5.2 CURVA ROC E AUC
#
# ======================================================================

plt.figure(figsize=(9, 6))

for nome, (_, _, y_prob) in modelos.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{nome} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório (AUC = 0.500)")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)")
plt.title("Curva ROC - Comparação dos Modelos")
plt.legend()
salvar_e_mostrar("07_curva_roc")


# ======================================================================
#
# 6. COMPARAÇÃO DOS MODELOS
#
# 6.1 TABELA COMPARATIVA (CONJUNTO DE TESTE)
#
# ======================================================================

resultados_df = pd.DataFrame([
    avaliar_modelo(nome, y_test, y_pred, y_prob)
    for nome, (_, y_pred, y_prob) in modelos.items()
])

resultados_df = resultados_df.sort_values(by="AUC", ascending=False).reset_index(drop=True)
print("\nTabela comparativa (conjunto de teste):")
print(resultados_df.round(4).to_string(index=False))


# ======================================================================
#
# 6.2 VALIDAÇÃO CRUZADA
# Para confirmar que os resultados não dependem de uma única divisão
# treino/teste, aplicamos validação cruzada estratificada com 5 partes
# (*folds*) no conjunto de treino. Esta célula pode levar alguns
# minutos.
# ======================================================================

print("\nRodando a validação cruzada (pode levar alguns minutos)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

resultados_cv = []
for nome, (modelo, _, _) in modelos.items():
    auc_cv = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1_cv = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    resultados_cv.append({
        "Modelo": nome,
        "AUC (média)": auc_cv.mean(), "AUC (desvio)": auc_cv.std(),
        "F1 (média)": f1_cv.mean(), "F1 (desvio)": f1_cv.std()
    })

print("\nValidação cruzada (5 partes):")
print((pd.DataFrame(resultados_cv).round(4)).to_string(index=False))


# ======================================================================
#
# 6.3 ESCOLHA DO MELHOR MODELO
# O melhor modelo é escolhido pela AUC, pois ela avalia a capacidade
# de separar pacientes com e sem diabetes independentemente do limiar
# de decisão. O limiar pode ser ajustado depois, conforme a prioridade
# da clínica (seção 6.4).
# ======================================================================

melhor_nome = resultados_df.iloc[0]["Modelo"]
melhor_modelo, _, melhor_prob = modelos[melhor_nome]

print(f"\nMelhor modelo (maior AUC): {melhor_nome}")


# ======================================================================
#
# 6.4 AJUSTE DO LIMIAR DE DECISÃO
# Por padrão, o modelo classifica como "diabetes" quando a
# probabilidade é ≥ 0,5. Reduzir esse limiar aumenta a Sensibilidade
# (menos casos perdidos) ao custo de mais falsos positivos. A tabela
# abaixo ajuda a clínica a escolher o ponto de equilíbrio mais adequado.
#
# Observação: aqui a tabela é calculada no conjunto de teste apenas
# para fins de análise. Em um projeto real, o limiar deve ser escolhido
# em um conjunto de validação separado, para não "otimizar no teste".
# ======================================================================

limiares = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

tabela_limiar = []
for t in limiares:
    y_pred_t = (melhor_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    tabela_limiar.append({
        "Limiar": t,
        "Precisão": precision_score(y_test, y_pred_t, zero_division=0),
        "Sensibilidade": recall_score(y_test, y_pred_t, zero_division=0),
        "F1": f1_score(y_test, y_pred_t, zero_division=0),
        "Falsos negativos": fn,
        "Falsos positivos": fp
    })

print(f"\nEfeito do limiar de decisão - {melhor_nome}:")
print((pd.DataFrame(tabela_limiar).round(4)).to_string(index=False))


# ======================================================================
#
# 7. INTERPRETAÇÃO: IMPORTÂNCIA DAS VARIÁVEIS
# Analisamos a importância das variáveis (Feature Importance) do melhor
# modelo, que indica o quanto cada variável contribuiu para as decisões
# do modelo.
# ======================================================================

# Recuperando o pré-processador e o modelo já treinados de dentro do pipeline
preprocessor_fitted = melhor_modelo.named_steps["preprocessor"]
modelo_final = melhor_modelo.named_steps["model"]

# Nomes das variáveis após o OneHotEncoder (removendo os prefixos "num__" e "cat__")
feature_names = [
    nome.replace("num__", "").replace("cat__", "")
    for nome in preprocessor_fitted.get_feature_names_out()
]


def variavel_original(nome):
    """Devolve a variável original de uma coluna criada pelo one-hot."""
    for col in categorical_features:
        if nome.startswith(col + "_"):
            return col
    return nome


def nome_variavel_pt(nome):
    """Traduz o nome de uma coluna do modelo (ex.: 'gender_Female' -> 'gênero = Feminino')."""
    col = variavel_original(nome)
    if col != nome:
        valor = nome[len(col) + 1:]
        return f"{NOMES_PT[col]} = {VALORES_PT.get(valor, valor)}"
    return NOMES_PT.get(nome, nome)


feature_importance = (
    pd.DataFrame({"Feature": feature_names, "Importance": modelo_final.feature_importances_})
    .sort_values(by="Importance", ascending=False)
    .reset_index(drop=True)
)

top_features = feature_importance.head(15)
nomes_top_pt = top_features["Feature"].apply(nome_variavel_pt)

print("\n15 variáveis mais importantes:")
print(pd.DataFrame({
    "variável": nomes_top_pt,
    "importância": top_features["Importance"].round(4)
}).to_string(index=False))


plt.figure(figsize=(10, 7))
plt.barh(nomes_top_pt[::-1], top_features["Importance"][::-1])
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.title(f"15 variáveis mais importantes - {melhor_nome}")
plt.tight_layout()
salvar_e_mostrar("08_feature_importance")


# Importância agregada por variável ORIGINAL (somando as colunas do one-hot de cada uma)
importancia_agrupada = (
    feature_importance
    .assign(Variavel=feature_importance["Feature"].apply(variavel_original))
    .groupby("Variavel")["Importance"].sum()
    .sort_values(ascending=False)
)

print("\nImportância total por variável original:")
print(pd.DataFrame({
    "variável": [NOMES_PT.get(v, v) for v in importancia_agrupada.index],
    "importância": importancia_agrupada.round(4).values
}).to_string(index=False))


# ======================================================================
# Discussão dos resultados (XGBoost, melhor modelo):
#
# - HbA1c_level (~41% da importância) e blood_glucose_level (~26%)
#   concentram cerca de 2/3 das decisões do modelo. Isso faz sentido
#   clinicamente: ambos são justamente os exames usados no diagnóstico
#   de diabetes. Na análise exploratória, a média de HbA1c foi 6,93 nos
#   diabéticos contra 5,40 nos não diabéticos, e a de glicose foi 194
#   contra 133.
# - O atributo derivado "comorbidades" (~10%) foi a 3ª variável mais
#   importante, acima das variáveis originais hypertension e
#   heart_disease isoladas. Isso mostra que a engenharia de atributos
#   agregou valor: a taxa de diabetes sobe de 6,4% (sem comorbidades)
#   para 27,6% (uma) e 39,1% (duas).
# - Idade (~5%) e IMC (somando bmi e bmi_category, ~5%) aparecem em
#   seguida como fatores de risco. Pacientes com 60+ anos têm 20,5% de
#   diabetes, e obesos têm 18,0%.
# - smoking_history (~6%) tem importância puxada principalmente pela
#   categoria "No Info", cuja taxa de diabetes é bem menor (4,1%).
#   Provavelmente, a falta de informação sobre tabagismo está associada
#   a pacientes mais jovens, e não ao tabagismo em si.
# - Gênero tem contribuição pequena (~2%).
#
# Ponto de atenção: como HbA1c e glicose são critérios diagnósticos, o
# modelo é excelente para identificar casos já existentes (possivelmente
# ainda não diagnosticados). Para prever o risco *antes* da alteração
# laboratorial, seriam necessários dados históricos dos pacientes.
# ======================================================================

# ======================================================================
#
# 8. CONCLUSÃO E RECOMENDAÇÃO
#
# Resultados no conjunto de teste (limiar padrão 0,5):
#
#   Modelo              Acurácia  Precisão  Sensibilidade  F1      AUC
#   XGBoost             0,909     0,493     0,897          0,636   0,976
#   Random Forest       0,919     0,523     0,867          0,653   0,973
#   Decision Tree       0,857     0,375     0,924          0,533   0,967
#
# Modelo recomendado: XGBOOST.
# - Obteve a maior AUC no teste (0,976) e na validação cruzada com 5
#   folds (0,978 ± 0,001). O desvio baixo indica que o resultado é
#   estável e não depende de uma divisão específica dos dados.
# - A Decision Tree tem a maior Sensibilidade no limiar 0,5, mas à
#   custa de muitos falsos positivos (2.616, com Precisão de apenas
#   0,375) e da menor AUC. Ela é útil como modelo interpretável de
#   referência, mas separa pior as classes.
# - O Random Forest tem F1 ligeiramente maior no limiar 0,5, mas
#   Sensibilidade menor e AUC inferior. Como a AUC avalia o modelo em
#   todos os limiares, o XGBoost oferece a melhor base para ajustar o
#   ponto de corte conforme a necessidade da clínica.
# - A Acurácia não foi usada como critério: com 91,5% de não
#   diabéticos, um modelo que sempre dissesse "não" teria 91,5% de
#   acerto e seria inútil.
#
# Limiar de decisão recomendado: 0,3.
# - Com ele, a Sensibilidade sobe de 0,897 para 0,951 e os diabéticos
#   não identificados caem de 174 para 83 (menos da metade). Em
#   contrapartida, os falsos positivos sobem de 1.567 para 2.583.
# - Para triagem preventiva esse trade-off é vantajoso: um falso
#   positivo gera apenas um exame confirmatório (barato), enquanto um
#   falso negativo significa um paciente diabético sem acompanhamento.
#   A clínica pode ajustar o limiar conforme sua capacidade de
#   atendimento (tabela 6.4).
#
# Implantação: o pipeline completo (pré-processamento + modelo) é salvo
# com joblib e pode ser aplicado diretamente aos novos pacientes,
# gerando uma lista ordenada por probabilidade de risco, para que as
# equipes priorizem exames e acompanhamento.
#
# Limitações: base secundária do Kaggle, ausência de histórico familiar
# e de dados longitudinais, e necessidade de validar o modelo com dados
# reais da rede de clínicas antes do uso em produção.
# ======================================================================

# Salvando o pipeline final para uso em produção
import joblib
caminho_modelo = pasta_script / "modelo_diabetes.pkl"
joblib.dump(melhor_modelo, caminho_modelo)
print(f"\nModelo '{melhor_nome}' salvo em {caminho_modelo}")

# Exemplo de uso em produção: probabilidade de risco para novos pacientes,
# aplicando o limiar recomendado de 0,3
modelo_carregado = joblib.load(caminho_modelo)
exemplo = X_test.head(5).copy()
exemplo["prob_diabetes"] = modelo_carregado.predict_proba(exemplo)[:, 1].round(3)
exemplo["alerta_risco"] = (exemplo["prob_diabetes"] >= 0.3).astype(int)

saida = exemplo[["gender", "age", "bmi", "HbA1c_level", "blood_glucose_level",
                 "prob_diabetes", "alerta_risco"]].rename(columns={
    **NOMES_PT,
    "prob_diabetes": "probabilidade de diabetes",
    "alerta_risco": "alerta de risco"
})
saida["gênero"] = saida["gênero"].replace(VALORES_PT)
saida["alerta de risco"] = saida["alerta de risco"].map(SIM_NAO)

print("\nExemplo de aplicação do modelo em novos pacientes (limiar 0,3):")
print(saida.to_string(index=False))
