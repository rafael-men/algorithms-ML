# Cartas de Tarot — Classificação com Machine Learning

Experimento de classificação supervisionada que tenta identificar o **grupo de uma carta de tarot**
(arcano maior ou um dos quatro naipes) a partir dos atributos presentes na base de cartas.
O projeto compara famílias de algoritmos de machine learningusando validação cruzada e diferentes esquemas de
pré-processamento.

- **Tarefa:** classificação multiclasse (5 classes)
- **Amostras:** 78 cartas (22 arcanos maiores + 56 cartas de naipe)
- **Stack:** Python 3.13 + pandas + scikit-learn
- **Idioma dos scripts:** português (comentários, rótulos e relatórios)

---

## Objetivo

Responder a seguinte pergunta experimental:

> Dada uma carta (com número de numerologia, planeta, signo, elemento e indicadores de
> "sim/não" na posição normal e reversa), é possível prever se ela pertence ao arcano maior ou a
> um dos naipes (`wands`, `cups`, `swords`, `pentacles`)?

Cada script do repositório isola uma hipótese: um algoritmo isolado, uma combinação de algoritmos
ou uma escolha específica de pré-processamento.

---

## Dataset

A base fica em `main/dataset/` e existe em dois formatos equivalentes (78 registros × 22 atributos):

| Arquivo | Formato | Formato usado por |
| --- | --- | --- |
| `cards.json` | JSON (array de objetos) | `main/KNN`, `main/Naive-Bayes-KNN` |
| `Dataset-Sheet.xlsx` | Excel (aba `cards`) | `main/decision-tree`, `main/Comparações` |

### Definição do rótulo (target)

```python
if row['arcana'] == 'major':
    label = 'major'
else:
    label = row['suit']
```

### Distribuição das classes

| Classe | Cartas | % do total |
| --- | --- | --- |
| `major` | 22 | 28,2% |
| `wands` | 14 | 17,9% |
| `cups` | 14 | 17,9% |
| `swords` | 14 | 17,9% |
| `pentacles` | 14 | 17,9% |

### Atributos

| Grupo | Colunas |
| --- | --- |
| Identificação | `id`, `name`, `arcana`, `suit` |
| Numérico | `number_numerology` |
| Categóricos | `element`, `planet`, `zodiac`, `yes_no`, `yes_no_reversed` |
| Texto (keywords) | `keywords_upright`, `keywords_reversed` |
| Texto (significados) | `meaning_upright`, `meaning_reversed` |
| Texto (áreas) | `love`, `career`, `mood`, `spiritual` (+ variantes `_reversed`) |

---

## Metodologia

Todos os scripts seguem o mesmo pipeline:

1. **Carga** — `json.load` + `pd.json_normalize` (JSON) ou `pd.read_excel` (planilha).
2. **Rótulo** — `major` quando `arcana == 'major'`, caso contrário `suit`.
3. **Engenharia de features**
   - colunas categóricas: `fillna('unknown')`
   - coluna numérica: `to_numeric` + `fillna(median)`
   - colunas de texto: listas são achatadas com *join* e todas as colunas textuais são
     concatenadas em um único `text_blob`
4. **Pré-processamento** — `ColumnTransformer` combinando:
   - `OneHotEncoder(handle_unknown='ignore')` para categóricos
   - `StandardScaler` / `MinMaxScaler` / `passthrough` para o número de numerologia
   - `TfidfVectorizer(max_features=100, stop_words='english')` para o `text_blob`
     (usado apenas nos scripts de KNN)
5. **Validação**
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` nos scripts de KNN e comparações
   - `train_test_split(test_size=0.25, stratify=y, random_state=42)` nos scripts de árvore/RF

> Os scripts usam `StratifiedKFold` com divisão por índice (`skf.split`) e retreinam um
> `ColumnTransformer` novo a cada fold, evitando vazamento de dados entre treino e teste.

---

## Modelos comparados

| Script | Modelo(s) | Hiperparâmetros | Validação |
| --- | --- | --- | --- |
| `main/KNN/main.py` | KNN | `n_neighbors` = 3, 5, 7 | 5-fold |
| `main/Naive-Bayes-KNN/main.py` | KNN + Naive Bayes | `k` = 3, 5, 7; `MultinomialNB()` | 5-fold |
| `main/decision-tree/main.py` | Árvore de Decisão | `criterion='gini'`, `max_depth=3` | hold-out 25% |
| `main/decision-tree/random -forest/main.py` | Random Forest | `n_estimators=200`, `n_jobs=-1` | hold-out 25% + 5-fold |
| `main/Comparações/knn_dt.py` | Árvore + KNN | `max_depth=3`; `k=5` | hold-out 25% + 5-fold |
| `main/Comparações/knn_dt_nb_trees.py` | NB + Árvore + KNN | `BernoulliNB(alpha=1.0)`; `max_depth=3`; `k=5` | hold-out 25% + 5-fold |
| `main/Comparações/knn_dt__nb_trees_forest.py` | NB + Árvore + RF + KNN | + `n_estimators=200` | hold-out 25% + 5-fold |

Detalhe importante: o **Naive Bayes exige features binárias**, então `knn_dt_nb_trees.py` e
`knn_dt__nb_trees_forest.py` usam um `preprocessador_nb` dedicado com
`StandardScaler` → `Binarizer(threshold=0.0)` na coluna numérica.

`main/decision-tree/tree.py` apenas plota a árvore treinada em `main/decision-tree/main.py`
com `plot_tree` e `matplotlib`.

---

## Como executar

### Pré-requisitos

- Python 3.13 (o ambiente `main/libs` foi criado com 3.13.7)

### Instalação

```bash
git clone https://github.com/rafael-men/tarot-cards-algorithms-machine-learning.git
cd tarot-cards-algorithms-machine-learning

python -m venv main/libs
main\libs\Scripts\activate
pip install -r main/requirements.txt
```

Dependências (`main/requirements.txt`): `pandas`, `scikit-learn`, `numpy`, `matplotlib`,
`seaborn`, `openpyxl`.

### Executando os scripts

> **Atenção:** todos os scripts usam caminhos **relativos** ao diretório de trabalho.
> Execute cada um a partir da pasta indicada na tabela — caso contrário você recebe
> `FileNotFoundError` ao ler o dataset.

| Script | Executar a partir de | Comando |
| --- | --- | --- |
| `KNN/main.py` | `main/` | `python KNN/main.py` |
| `Naive-Bayes-KNN/main.py` | `main/Naive-Bayes-KNN/` | `python main.py` |
| `decision-tree/main.py` | `main/` | `python decision-tree/main.py` |
| `decision-tree/tree.py` | `main/` | `python decision-tree/tree.py` |
| `decision-tree/random -forest/main.py` | `main/decision-tree/random -forest/` | `python main.py` |
| `Comparações/knn_dt.py` | `main/Comparações/` | `python knn_dt.py` |
| `Comparações/knn_dt_nb_trees.py` | `main/Comparações/` | `python knn_dt_nb_trees.py` |
| `Comparações/knn_dt__nb_trees_forest.py` | `main/Comparações/` | `python knn_dt__nb_trees_forest.py` |

`decision-tree/tree.py` importa `main` (`from main import preprocessador, classificador`),
então ele **depende** de `main/decision-tree/main.py` e precisa de um backend interativo
do matplotlib para exibir a figura.

---

## Resultados

Os números abaixo foram medidos neste repositório (Python 3.13.7, scikit-learn 1.9.1,
pandas 3.0.5). Cada tabela corresponde à execução de um script.

### KNN isolado — `main/KNN/main.py`

Acurácia média da validação cruzada 5-fold, com TF-IDF + One-Hot + `passthrough`:

| Modelo | Acurácia média | Desvio padrão |
| --- | --- | --- |
| KNN = 3 | 43,58% | 16,11% |
| **KNN = 5** | **48,83%** | 21,40% |
| KNN = 7 | 47,42% | 20,04% |

### KNN vs. Naive Bayes — `main/Naive-Bayes-KNN/main.py`

| Modelo | Acurácia média | Desvio padrão |
| --- | --- | --- |
| KNN = 3 | 26,92% | 8,24% |
| KNN = 5 | 33,67% | 15,53% |
| **KNN = 7** | **38,58%** | 10,44% |
| Naive Bayes (`MultinomialNB`) | 33,33% | 14,21% |

O `KNN=7` venceu o Naive Bayes por **5,25 pontos percentuais**.

### Árvore de Decisão — `main/decision-tree/main.py`

Acurácia no conjunto de teste (hold-out 25%): **0,35**.
A matriz de confusão mostra concentração forte na coluna `swords`, indicando que a árvore
(max_depth=3) quase não diferencia os naipes.

### Comparação completa — `main/Comparações/knn_dt__nb_trees_forest.py`

Validação cruzada 5-fold sobre a base Excel (One-Hot + StandardScaler):

| Modelo | Acurácia média | Desvio padrão |
| --- | --- | --- |
| Árvore de Decisão | 29,50% | 10,18% |
| KNN (k=5) | 32,08% | 8,11% |
| BernoulliNB | 42,50% | 9,42% |
| **Random Forest** | **48,83%** | 11,00% |

Métricas no hold-out de 25% do mesmo script:

| Modelo | Acurácia | Macro F1 |
| --- | --- | --- |
| BernoulliNB | 0,50 | 0,4754 |
| Árvore de Decisão | 0,35 | 0,2386 |
| Random Forest | 0,50 | 0,4866 |
| KNN (k=5) | 0,35 | 0,3222 |

**Melhor algoritmo: Random Forest, com 48,83% de acurácia média no 5-fold.**

### Importância das features — `main/decision-tree/random -forest/main.py`

A Random Forest (200 árvores, `f1_macro` de 42,86% ± 7,53% no 5-fold interno de treino)
apontou como features mais relevantes:

| Feature | Importância |
| --- | --- |
| `number_numerology` | 0,1571 |
| `yes_no_yes` | 0,0557 |
| `zodiac_Capricorn` | 0,0529 |
| `zodiac_Cancer` | 0,0436 |
| `yes_no_reversed_no` | 0,0407 |

O número de numerologia concentra ~16% da importance — é o único sinal numérico com
valor discriminativo real no dataset.

### Referências de comparação

| Modelo de referência | Acurácia |
| --- | --- |
| Sempre prever a classe majoritária (`major`) | 28,20% |
| Aleatório uniforme entre 5 classes | 20,00% |
| Melhor modelo obtido aqui (Random Forest) | 48,83% |

---

## Conclusões

1. **Random Forest é o algoritmo mais estável** do estudo: lidera tanto no 5-fold (48,83%)
   quanto no hold-out, e ainda é o único com macro F1 acima de 0,48.
2. **KNN é altamente sensível à representação dos dados.** Com TF-IDF + One-Hot + `passthrough`
   o k=5 atinge 48,83%; com One-Hot + StandardScaler o mesmo k=5 cai para 32,08%. Como a
   distância euclidiana é distorcida por vetores esparsos de texto, a escolha do pré-processamento
   importa mais do que o k.
3. **Naive Bayes se beneficia de features binárias** (`Binarizer` na coluna numérica):
   42,50% contra os 33,33% do `MultinomialNB` com features contínuas.
4. **A árvore podada (`max_depth=3`) é insuficiente** — 29,50% fica abaixo do baseline da classe
   majoritária. O ensemble de árvores resolve o problema de viés de alto corte.
5. **O volume de dados é o principal gargalo.** Com 78 amostras e features majoritariamente
   categóricas de baixa cardinalidade, o teto de ~50% de acurácia reflete mais a qualidade
   dos atributos do que a escolha do classificador.

---

## Estrutura do projeto

```
.
├── main/
│   ├── dataset/
│   │   ├── cards.json              # base JSON (78 × 22)
│   │   └── Dataset-Sheet.xlsx      # mesma base em Excel (aba "cards")
│   ├── KNN/
│   │   └── main.py                 # KNN k=3,5,7 com TF-IDF
│   ├── Naive-Bayes-KNN/
│   │   └── main.py                 # KNN vs. MultinomialNB
│   ├── decision-tree/
│   │   ├── main.py                 # Árvore de Decisão (hold-out + confusion matrix)
│   │   ├── tree.py                 # visualização da árvore
│   │   └── random -forest/
│   │       └── main.py             # Random Forest + importâncias de features
│   ├── Comparações/
│   │   ├── knn_dt.py               # Árvore vs. KNN
│   │   ├── knn_dt_nb_trees.py      # + BernoulliNB
│   │   └── knn_dt__nb_trees_forest.py  # + Random Forest (comparação final)
│   ├── libs/                       # ambiente virtual (não versionado)
│   └── requirements.txt
└── .gitignore
```

> `main/libs` (venv) não é versionado. As pastas `Comparações/` e
> `decision-tree/random -forest/` ainda não foram commitadas.



## Referências

- [scikit-learn — KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [scikit-learn — Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scikit-learn — DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [scikit-learn — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [scikit-learn — StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

