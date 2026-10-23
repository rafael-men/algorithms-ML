from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

dataset = '../dataset/Dataset-Sheet.xlsx'
categorias = ["planet", "zodiac", "element", "yes_no", "yes_no_reversed"]
numericas = ["number_numerology"]
texto = [
    "keywords_upright", "keywords_reversed",
    "meaning_upright", "meaning_reversed",
    "love", "love_reversed",
    "career", "career_reversed",
    "mood", "mood_reversed",
    "spiritual", "spiritual_reversed",
]


def carregar(caminho):
    df = pd.read_excel(caminho)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def juntar(valor):
    if isinstance(valor, list):
        return " ".join(map(str, valor))
    if pd.isna(valor):
        return ""
    return str(valor)


def preparar(df):
    # classe: "major" para arcanos maiores, senão o naipe
    y = df["suit"].mask(df["arcana"] == "major", "major")

    cats = [c for c in categorias if c in df.columns]
    nums = [c for c in numericas if c in df.columns]
    txts = [c for c in texto if c in df.columns]

    X = pd.DataFrame(index=df.index)
    for c in cats:
        X[c] = df[c].fillna("desconhecido").astype(str)
    for c in nums:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["texto"] = df[txts].map(juntar).agg(" ".join, axis=1)

    return X, y, cats, nums


def pipeline_knn(cats, nums):
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


def pipeline_bnb(cats, nums):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore")),
        ]), nums),
        ("txt", CountVectorizer(
            binary=True, stop_words="english",
            ngram_range=(1, 2), min_df=2,
        ), "texto"),
    ])
    nb = BernoulliNB(binarize=None)
    return Pipeline([("pre", pre), ("nb", nb)])


GRADE_KNN = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"],
    "pre__transformer_weights": [
        {"cat": 1, "num": 1, "txt": 1},
        {"cat": 2, "num": 1, "txt": 1},
        {"cat": 1, "num": 1, "txt": 2},
    ],
}

GRADE_BNB = {
    "nb__alpha": [0.1, 0.3, 0.5, 1.0, 2.0],
    "nb__fit_prior": [True, False],
    "pre__txt__max_features": [200, 500, 1000],
}


def main():
    df = carregar(dataset)
    X, y, cats, nums = preparar(df)
    print(y.value_counts(), "\n")

    interno = StratifiedKFold(5, shuffle=True, random_state=42)
    externo = StratifiedKFold(5, shuffle=True, random_state=7)

    modelos = {
        "KNN": GridSearchCV(pipeline_knn(cats, nums), GRADE_KNN,
                            cv=interno, scoring="accuracy", n_jobs=-1),
        "Bernoulli NB": GridSearchCV(pipeline_bnb(cats, nums), GRADE_BNB,
                                     cv=interno, scoring="accuracy", n_jobs=-1),
    }

    linhas = []
    por_fold = {}
    for nome, busca in modelos.items():
        notas = cross_val_score(busca, X, y, cv=externo)
        por_fold[nome] = notas
        linhas.append({
            "Modelo": nome,
            "Acurácia Média": f"{notas.mean():.2%}",
            "Desvio Padrão": f"{notas.std():.2%}",
        })

    print("Resultados da validação cruzada aninhada:")
    print(pd.DataFrame(linhas).to_string(index=False))
    print("\nAcurácia por fold:")
    print((pd.DataFrame(por_fold) * 100).round(2).rename_axis("fold").to_string())
    melhor = max(por_fold, key=lambda n: por_fold[n].mean())
    print(f"\nMelhor: {melhor} com {por_fold[melhor].mean():.2%}")
    print("\nMelhores parâmetros (ajustando no dataset inteiro):")
    for nome, busca in modelos.items():
        busca.fit(X, y)
        print(f"{nome}: {busca.best_params_}")


if __name__ == "__main__":
    main()