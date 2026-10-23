import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder


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


def carregar(dataset):
      df = pd.read_excel(dataset)
      df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
      return df

def juntar(valor):
    if isinstance(valor, list):
        return " ".join(map(str, valor))
    if pd.isna(valor):
        return ""
    else:
     return str(valor)

def preparar(df):
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

def pipeline(cats, nums):
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
    # features binarizados para o classificador 
    nb = BernoulliNB(binarize=None)
    return Pipeline([("pre", pre), ("nb", nb)])

GRADE = {
    "nb__alpha": [0.1, 0.3, 0.5, 1.0, 2.0],
    "nb__fit_prior": [True, False],
    "pre__txt__max_features": [200, 500, 1000],
}


def main():
    df = carregar(dataset)
    X, y, cats, nums = preparar(df)
    print(y.value_counts(), "\n")

    pipe = pipeline(cats, nums)
    interno = StratifiedKFold(5, shuffle=True, random_state=42)
    externo = StratifiedKFold(5, shuffle=True, random_state=7)
    busca = GridSearchCV(pipe, GRADE, cv=interno, scoring="accuracy", n_jobs=-1)
    notas = cross_val_score(busca, X, y, cv=externo)
    print(f"CV aninhada: {notas.mean():.2%} ± {notas.std():.2%}")
    busca.fit(X, y)
    print("Melhores parâmetros:", busca.best_params_)
    res = pd.DataFrame(busca.cv_results_)
    res["alpha"] = res["param_nb__alpha"].astype(float)
    res["fit_prior"] = res["param_nb__fit_prior"]

    tabela = (
        res.groupby(["alpha", "fit_prior"])["mean_test_score"]
        .max()
        .unstack()
        .mul(100)
        .round(2)
    )
    print("\nMelhor acurácia (%):")
    print(tabela)


if __name__ == "__main__":
    main()