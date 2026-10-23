import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Classificação de cartas por naipe e arcano maior com KNN, comparando acuracias com validação cruzada K-Fold (5)

dataset = '../dataset/Dataset-Sheet.xlsx'
categorias = [ 'planet', 'zodiac', 'element', 'yes_no', 'yes_no_reversed' ]
numericas = [ 'number_numerology' ]
texto = [
    "keywords_upright",
    "keywords_reversed",
    "meaning_upright",
    "meaning_reversed",
    "love",
    "career",
    "mood",
    "spiritual",
]

def carregar(dataset):
   df = pd.read_excel(dataset)
   df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
   return df

def aggregate_col(valor):
    if isinstance(valor, list):
        return " ".join(map(str, valor))
    if pd.isna(valor):
        return ""
    return str(valor)

def preparar(df):
    # arcano maior e naipes 
    y = df["suit"].mask(df["arcana"] == "major", "major")

    cat = [c for c in categorias if c in df.columns]
    num = [c for c in numericas if c in df.columns]
    txt = [c for c in texto if c in df.columns]

    X = pd.DataFrame(index=df.index)
    for c in cat:
        X[c] = df[c].fillna("desconhecido").astype(str)
    for c in num:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["texto"] = df[txt].map(aggregate_col).agg(" ".join, axis=1)
    return X, y, cat, num


def pipeline(cats, nums):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("esc", StandardScaler()),
        ]), nums),
        ("txt", TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2),
            min_df=2, sublinear_tf=True, max_features=500,
        ), "texto"),
    ])
    knn = KNeighborsClassifier(metric="cosine", algorithm="brute")
    return Pipeline([("pre", pre), ("knn", knn)])


grade_testes = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"],
    "pre__transformer_weights": [
        {"cat": 1, "num": 1, "txt": 1},
        {"cat": 2, "num": 1, "txt": 1},
        {"cat": 1, "num": 1, "txt": 2},
    ],
}


def main():
    df = carregar(dataset)
    print("Colunas:", df.columns.tolist())
    X, y, cats, nums = preparar(df)
    print(y.value_counts(), "\n")
    pipe = pipeline(cats, nums)
    interno = StratifiedKFold(5, shuffle=True, random_state=42)
    externo = StratifiedKFold(5, shuffle=True, random_state=7)
    busca = GridSearchCV(pipe, grade_testes, cv=interno, scoring="accuracy", n_jobs=-1)
    notas = cross_val_score(busca, X, y, cv=externo)
    print(f"Acurácia (CV aninhada): {notas.mean():.2%} ± {notas.std():.2%}")
    busca.fit(X, y)
    print("Melhores parâmetros:", busca.best_params_)

    res = pd.DataFrame(busca.cv_results_)
    res["k"] = res["param_knn__n_neighbors"]
    res["pesos"] = res["param_knn__weights"]

    tabela = (
    res.groupby(["k", "pesos"])["mean_test_score"]
    .max()
    .unstack()
    .mul(100)
    .round(2)
   )
    print("\nMelhor acurácia por k (%):")
    print(tabela)
    
if __name__ == "__main__":
    main()