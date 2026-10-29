# Algoritmos de Machine Learning

Experimento de classificação supervisionada que tenta identificar o **grupo de uma carta de tarot**
(arcano maior ou um dos quatro naipes) a partir dos atributos presentes na base de cartas.
O projeto compara famílias de algoritmos de ML e Redes Neurais Artificiais usando validação cruzada e diferentes
esquemas de pré-processamento.

- **Tarefa:** classificação multiclasse (5 classes)
- **Amostras:** 78 cartas (22 arcanos maiores + 56 cartas de naipe)
- **Stack:** Python 3.13 + pandas + scikit-learn + numpy

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

| Arquivo | Formato | Usado por |
| --- | --- | --- |
| `cards.json` | JSON (array de objetos) | `main/Naive-Bayes-KNN` |
| `Dataset-Sheet.xlsx` | Excel (aba `cards`) | `main/KNN`, `main/Bernoulli-NB`, `main/decision-tree`, `main/Comparações` |

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

Os scripts seguem a mesma lógica geral. Os de KNN e Bernoulli NB (e a comparação entre eles) foram
reescritos com `Pipeline` e validação aninhada; os demais mantêm o fluxo original, descrito mais abaixo.

### Scripts reescritos: `KNN`, `Bernoulli-NB` e `Comparações/knn_bnb.py`

1. **Carga** — `pd.read_excel` na planilha, com os nomes das colunas padronizados
   (`strip`, `lower`, espaços trocados por `_`).
2. **Rótulo** — `major` quando `arcana == 'major'`, caso contrário `suit`.
3. **Montagem do `X`**
   - categóricas: `fillna('desconhecido')`
   - numérica: `to_numeric` (valores ausentes ficam para o imputador do pipeline)
   - texto: listas são achatadas, células vazias viram string vazia e todas as colunas textuais
     (incluindo as variantes `_reversed`) são concatenadas em uma coluna `texto`
4. **Pré-processamento dentro do `Pipeline`** (ajustado só com o treino de cada fold):

   | Bloco | KNN | Bernoulli NB |
   | --- | --- | --- |
   | Categóricas | `OneHotEncoder` | `OneHotEncoder` |
   | `number_numerology` | `SimpleImputer(median)` + `StandardScaler` | `SimpleImputer(most_frequent)` + `OneHotEncoder` (tratado como categoria) |
   | Texto | `TfidfVectorizer` (1-2 gramas, `min_df=2`, `sublinear_tf`, 500 termos) | `CountVectorizer(binary=True)` (1-2 gramas, `min_df=2`) |

5. **Classificador**
   - KNN: `metric='cosine'`, `algorithm='brute'`
   - Bernoulli NB: `binarize=None`, já que todas as features entram como 0/1
6. **Validação cruzada aninhada**
   - `GridSearchCV` com `StratifiedKFold(5, shuffle=True, random_state=42)` (interno) escolhe os
     hiperparâmetros
   - `cross_val_score` com `StratifiedKFold(5, shuffle=True, random_state=7)` (externo) mede a
     acurácia, de modo que a escolha dos hiperparâmetros nunca vê o fold de teste

Grades de hiperparâmetros:

| Modelo | Parâmetros testados |
| --- | --- |
| KNN | `n_neighbors` = 3, 5, 7, 9; `weights` = uniform, distance; peso dos blocos (`cat`, `num`, `txt`) = 1/1/1, 2/1/1, 1/1/2 |
| Bernoulli NB | `alpha` = 0.1, 0.3, 0.5, 1.0, 2.0; `fit_prior` = True, False; `max_features` do vocabulário = 200, 500, 1000 |

### Demais scripts (fluxo original)

1. **Carga** — `json.load` + `pd.json_normalize` (JSON) ou `pd.read_excel` (planilha).
2. **Rótulo** — igual ao descrito acima.
3. **Engenharia de features**
   - colunas categóricas: `fillna('unknown')`
   - coluna numérica: `to_numeric` + `fillna(median)`
   - colunas de texto: listas são achatadas com *join* e concatenadas em um único `text_blob`
4. **Pré-processamento** — `ColumnTransformer` combinando `OneHotEncoder(handle_unknown='ignore')`
   para categóricos, `StandardScaler` / `MinMaxScaler` / `passthrough` para a numérica e
   `TfidfVectorizer(max_features=100, stop_words='english')` para o `text_blob` (só no KNN).
5. **Validação**
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` em `Naive-Bayes-KNN` e `Comparações`
   - `train_test_split(test_size=0.25, stratify=y, random_state=42)` nos scripts de árvore/RF

> Esses scripts usam `StratifiedKFold` com divisão por índice (`skf.split`) e retreinam um
> `ColumnTransformer` novo a cada fold, evitando vazamento de dados entre treino e teste.

---

## Modelos comparados

| Script | Modelo(s) | Hiperparâmetros | Validação |
| --- | --- | --- | --- |
| `main/KNN/main.py` | KNN | grade (ver acima), cosseno | 5-fold aninhado |
| `main/Bernoulli-NB/main.py` | Bernoulli NB | grade (ver acima) | 5-fold aninhado |
| `main/Comparações/knn_bnb.py` | KNN + Bernoulli NB | grades acima, mesmos folds externos | 5-fold aninhado |
| `main/Naive-Bayes-KNN/main.py` | KNN + Naive Bayes | `k` = 3, 5, 7; `MultinomialNB()` | 5-fold |
| `main/decision-tree/main.py` | Árvore de Decisão | `criterion='gini'`, `max_depth=3` | hold-out 25% |
| `main/decision-tree/random -forest/main.py` | Random Forest | `n_estimators=200`, `n_jobs=-1` | hold-out 25% + 5-fold |
| `main/Comparações/knn_dt.py` | Árvore + KNN | `max_depth=3`; `k=5` | hold-out 25% + 5-fold |
| `main/Comparações/knn_dt_nb_trees.py` | NB + Árvore + KNN | `BernoulliNB(alpha=1.0)`; `max_depth=3`; `k=5` | hold-out 25% + 5-fold |
| `main/Comparações/knn_dt__nb_trees_forest.py` | NB + Árvore + RF + KNN | + `n_estimators=200` | hold-out 25% + 5-fold |

Detalhe importante: o **Naive Bayes exige features binárias**. Nos scripts antigos
(`knn_dt_nb_trees.py` e `knn_dt__nb_trees_forest.py`) isso é resolvido com um `preprocessador_nb`
dedicado, com `StandardScaler` → `Binarizer(threshold=0.0)` na coluna numérica. Na versão reescrita
(`Bernoulli-NB`), a numérica vira categoria com `OneHotEncoder` e o texto usa
`CountVectorizer(binary=True)`.

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
`seaborn`, `openpyxl` (necessário para ler o `.xlsx`).

### Executando os scripts

Os scripts reescritos (`KNN`, `Bernoulli-NB` e `Comparações/knn_bnb.py`) montam o caminho do dataset a
partir da localização do próprio arquivo, então podem ser executados de qualquer pasta.

> **Atenção:** os demais scripts ainda usam caminhos **relativos** ao diretório de trabalho.
> Execute cada um a partir da pasta indicada na tabela — caso contrário você recebe
> `FileNotFoundError` ao ler o dataset.

| Script | Executar a partir de | Comando |
| --- | --- | --- |
| `KNN/main.py` | qualquer pasta | `python main/KNN/main.py` |
| `Bernoulli-NB/main.py` | qualquer pasta | `python main/Bernoulli-NB/main.py` |
| `Comparações/knn_bnb.py` | qualquer pasta | `python main/Comparações/knn_bnb.py` |
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

Os scripts de KNN e Bernoulli NB fazem busca em grade dentro de validação aninhada e podem levar
mais tempo que os demais.

---

## Conclusões

As conclusões 1 a 5 vêm das execuções originais e devem ser revisadas quando os resultados da
versão reescrita forem preenchidos.

1. **Random Forest é o algoritmo mais estável** nas execuções originais: lidera tanto no 5-fold
   (48,83%) quanto no hold-out, e ainda é o único com macro F1 acima de 0,48.
2. **KNN é altamente sensível à representação dos dados.** Com TF-IDF + One-Hot + `passthrough`
   o k=5 atinge 48,83%; com One-Hot + StandardScaler o mesmo k=5 cai para 32,08%. Como a
   distância euclidiana é distorcida por vetores esparsos de texto, a escolha do pré-processamento
   importa mais do que o k. A versão reescrita testa essa hipótese com distância cosseno e
   escala na coluna numérica.
3. **Naive Bayes se beneficia de features binárias** (`Binarizer` na coluna numérica):
   42,50% contra os 33,33% do `MultinomialNB` com features contínuas.
4. **A árvore podada (`max_depth=3`) é insuficiente** — 29,50% fica abaixo do baseline da classe
   majoritária. O ensemble de árvores resolve o problema de viés de alto corte.
5. **O volume de dados é o principal gargalo.** Com 78 amostras e features majoritariamente
   categóricas de baixa cardinalidade, o teto de ~50% de acurácia reflete mais a qualidade
   dos atributos do que a escolha do classificador.
6. **A validação aninhada dá uma estimativa menos otimista.** Escolher `k`, `alpha` e os pesos de
   bloco olhando o próprio resultado do fold inflaria a acurácia; na versão reescrita essa escolha
   acontece só dentro do treino de cada fold externo.

---

## Estrutura do projeto

```
.
├── main/
│   ├── dataset/
│   │   ├── cards.json              # base JSON (78 × 22)
│   │   └── Dataset-Sheet.xlsx      # mesma base em Excel (aba "cards")
│   ├── KNN/
│   │   └── main.py                 # KNN com Pipeline, cosseno e CV aninhada
│   ├── Bernoulli-NB/
│   │   └── main.py                 # Bernoulli NB com features binárias e CV aninhada
│   ├── Naive-Bayes-KNN/
│   │   └── main.py                 # KNN vs. MultinomialNB
│   ├── decision-tree/
│   │   ├── main.py                 # Árvore de Decisão (hold-out + confusion matrix)
│   │   ├── tree.py                 # visualização da árvore
│   │   └── random -forest/
│   │       └── main.py             # Random Forest + importâncias de features
│   ├── Comparações/
│   │   ├── knn_bnb.py              # KNN vs. Bernoulli NB (CV aninhada)
│   │   ├── knn_dt.py               # Árvore vs. KNN
│   │   ├── knn_dt_nb_trees.py      # + BernoulliNB
│   │   └── knn_dt__nb_trees_forest.py # + Random Forest (comparação final)
│   ├── Perceptrons-RNA/
│   │   ├── mlp.py                  # Classificação através de um perceptron multilayer
│   ├── libs/                       # ambiente virtual (não versionado)
│   └── requirements.txt
└── .gitignore
```


