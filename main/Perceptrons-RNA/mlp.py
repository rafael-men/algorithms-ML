import itertools

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Usa um peceptron multicamadas para classificar cartas por naipe e arcano maior

dataset = '../dataset/Dataset-Sheet.xlsx'

def sigmoid(z):
   return 1 / (1 + np.exp(-z))

class MLP:
   def __init__(self, camadas, lr, iteracoes, lambd=0.0, seed=2):
      self.camadas = camadas
      self.lr = lr
      self.iteracoes = iteracoes
      self.lambd = lambd
      self.seed = seed
   def _init(self, n_x, n_y):
        rng = np.random.default_rng(self.seed)
        tamanhos = [n_x, *self.camadas, n_y]
        self.W = [rng.standard_normal((tamanhos[i + 1], tamanhos[i])) * np.sqrt(1 / tamanhos[i])
                  for i in range(len(tamanhos) - 1)]
        self.b = [np.zeros((tamanhos[i + 1], 1)) for i in range(len(tamanhos) - 1)]
   def _forward(self, X):
        A = [X]
        for i in range(len(self.W)):
            Z = self.W[i] @ A[-1] + self.b[i]
            A.append(sigmoid(Z) if i == len(self.W) - 1 else np.tanh(Z))
        return A

   def fit(self, X, Y):
        X, Y = X.T, Y.T
        m = X.shape[1]
        self._init(X.shape[0], Y.shape[0])
        for _ in range(self.iteracoes):
            A = self._forward(X)
            dZ = A[-1] - Y
            for i in reversed(range(len(self.W))):
                dW = dZ @ A[i].T / m + self.lambd * self.W[i]
                db = dZ.sum(axis=1, keepdims=True) / m
                if i > 0:
                    dZ = (self.W[i].T @ dZ) * (1 - A[i] ** 2)
                self.W[i] -= self.lr * dW
                self.b[i] -= self.lr * db
        return self

   def predict(self, X):
        return np.argmax(self._forward(X.T)[-1], axis=0)


colunas = ["suit", "arcana"]   # colunas alvo
categorias = ["planet", "zodiac", "element", "yes_no", "yes_no_reversed"]
numerica = ["number_numerology"]
texto = ["keywords_upright", "keywords_reversed", "meaning_upright", "meaning_reversed",
         "love", "love_reversed", "career", "career_reversed",
         "mood", "mood_reversed", "spiritual", "spiritual_reversed"]


df = pd.read_excel(dataset)
df.columns = (df.columns.astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True))

def existentes(cols):
    faltando = [c for c in cols if c not in df.columns]
    if faltando:
        print("Colunas não encontradas (ignoradas):", faltando)
    return [c for c in cols if c in df.columns]

categorias = existentes(categorias)
numerica = existentes(numerica)
texto = existentes(texto)
df[categorias] = df[categorias].fillna("nenhum").astype(str)
df[numerica] = df[numerica].fillna(0)
df["texto"] = df[texto].fillna("").astype(str).agg(" ".join, axis=1)
## classes, y = np.unique(df[colunas].astype(str).agg(" ".join, axis=1), return_inverse=True)
## Y = np.eye(len(classes))[y] 
## print(classes)

def novo_preprocessador():
   return ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorias),
        ("num", StandardScaler(), numerica),
        ("txt", TfidfVectorizer(max_features=200), "texto"),
   ])

def prever(params, dados, Y, idx_tr, idx_va):
    prep = novo_preprocessador()
    X_tr = prep.fit_transform(dados.iloc[idx_tr])
    X_va = prep.transform(dados.iloc[idx_va])
    mlp = MLP(**params).fit(X_tr, Y[idx_tr])
    return mlp.predict(X_va)

def avaliar(params, dados, y, Y, idx_tr, idx_va):
    return accuracy_score(y[idx_va], prever(params, dados, Y, idx_tr, idx_va))


grade = {
    "camadas": [(8,), (16,), (32,)],
    "lr": [0.1],
    "iteracoes": [100, 250, 500, 1000],
    "lambd": [0, 0.01, 0.03, 0.1, 0.3],
}

def rodar(alvo):
    dados = df[df[alvo].notna()].reset_index(drop=True)

    # y e Y calculados por alvo 
    classes, y = np.unique(dados[alvo].astype(str), return_inverse=True)
    Y = np.eye(len(classes))[y]
    print("Classes:", list(classes))
    idx = np.arange(len(dados))
    idx_treino, idx_teste = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    resultados = []

    for valores in itertools.product(*grade.values()):
        params = dict(zip(grade.keys(), valores))
        accs = [avaliar(params, dados, y, Y, idx_treino[tr], idx_treino[va])
                for tr, va in skf.split(idx_treino, y[idx_treino])]
        resultados.append((np.mean(accs), np.std(accs), params))
        print(f"{params} -> {np.mean(accs):.3f} (+/- {np.std(accs):.3f})")

    resultados.sort(key=lambda r: r[0], reverse=True)
    melhor_cv, _, melhor = resultados[0]
    print(f"\nMelhor: {melhor} | CV: {melhor_cv:.3f}")
    acc_teste = avaliar(melhor, dados, y, Y, idx_treino, idx_teste)
    print(f"Acurácia no teste: {acc_teste * 100:.2f}%")
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for tr, va in skf.split(idx_treino, y[idx_treino]):
        pred = prever(melhor, dados, Y, idx_treino[tr], idx_treino[va])
        cm += confusion_matrix(y[idx_treino[va]], pred, labels=range(n))
    print("Matriz de confusão (CV):")
    print(pd.DataFrame(cm, index=classes, columns=classes))

    return melhor, acc_teste


if __name__ == "__main__":
    resumo = {alvo: rodar(alvo) for alvo in colunas}

    print("\n===== Resumo =====")
    for alvo, (params, acc) in resumo.items():
        print(f"{alvo}: {params} | teste {acc * 100:.2f}%")