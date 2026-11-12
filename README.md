# Algoritmos de Machine Learning

Experimento de classificação supervisionada que busca identificar o **grupo de uma carta de tarot** (arcano maior ou um dos quatro naipes) a partir dos atributos presentes na base de cartas.

O projeto também contempla atividades complementares de **séries temporais**, utilizando dados históricos de fechamento de ações do Yahoo!, e **clusterização**, aplicando K-Means e DBSCAN sobre a mesma base de cartas.

O objetivo é comparar diferentes famílias de algoritmos de Machine Learning e Redes Neurais Artificiais, utilizando validação cruzada e diferentes estratégias de pré-processamento.

- **Tarefa principal:** classificação multiclasse (5 classes).
- **Amostras:** 78 cartas (22 arcanos maiores + 56 cartas de naipe).
- **Stack:** Python 3.13, pandas, scikit-learn, NumPy e statsmodels.

---

## Objetivo

Responder à seguinte pergunta experimental:

> Dada uma carta, considerando seu número de numerologia, planeta, signo, elemento e indicadores de "sim/não" nas posições normal e reversa, é possível prever se ela pertence ao arcano maior ou a um dos quatro naipes (`wands`, `cups`, `swords`, `pentacles`)?

### Atividades complementares

- **Séries temporais:** dado o histórico de fechamento das ações do Yahoo!, é possível prever os próximos 15 valores?
- **Clusterização:** os grupos encontrados nos dados das cartas correspondem às classes originais?

Cada script do repositório isola uma hipótese experimental: um algoritmo específico, uma combinação de algoritmos ou uma escolha particular de pré-processamento.

---

## Dataset

A base de cartas está localizada em `main/dataset/` e existe em dois formatos equivalentes, contendo 78 registros e 22 atributos.

| Arquivo | Formato | Utilizado por |
|---|---|---|
| `cards.json` | JSON (array de objetos) | `main/Naive-Bayes-KNN` |
| `Dataset-Sheet.xlsx` | Excel (aba `cards`) | `main/KNN`, `main/Bernoulli-NB`, `main/decision-tree`, `main/Comparações`, `main/Regressao/clusterizacao.ipynb` |

A série temporal está localizada em:

| Arquivo | Descrição | Utilizado por |
|---|---|---|
| `yahoo_stock.csv` | Histórico do valor de fechamento das ações do Yahoo! (Yahoo Finance / Kaggle) | `main/Regressao/series-temporais.ipynb` |

### Definição do rótulo (target)

O rótulo é definido a partir das colunas `arcana` e `suit`:

```python
if row['arcana'] == 'major':
    label = 'major'
else:
    label = row['suit']
```

### Distribuição das classes

| Classe | Cartas | Percentual |
|---|---:|---:|
| major | 22 | 28,2% |
| wands | 14 | 17,9% |
| cups | 14 | 17,9% |
| swords | 14 | 17,9% |
| pentacles | 14 | 17,9% |
| **Total** | **78** | **100%** |

### Atributos

| Grupo | Colunas |
|---|---|
| Identificação | `id`, `name`, `arcana`, `suit` |
| Numérico | `number_numerology` |
| Categóricos | `element`, `planet`, `zodiac`, `yes_no`, `yes_no_reversed` |
| Texto (keywords) | `keywords_upright`, `keywords_reversed` |
| Texto (significados) | `meaning_upright`, `meaning_reversed` |
| Texto (áreas) | `love`, `career`, `mood`, `spiritual` e respectivas variantes `_reversed` |

---

## Metodologia

### 1. Classificação

A classificação é realizada por meio de diferentes algoritmos, com estratégias de pré-processamento e validação.

#### Scripts reescritos

Os scripts de KNN, Bernoulli Naive Bayes e a comparação entre ambos foram reescritos utilizando `Pipeline` e validação cruzada aninhada (*Nested Cross-Validation*).

**Carga dos dados**

- Utilização de `pd.read_excel` para carregar a planilha.
- Padronização dos nomes das colunas: remoção de espaços extras (`strip`), conversão para minúsculas (`lower`) e substituição de espaços por `_`.

**Definição do rótulo**

- `major` quando `arcana == 'major'`.
- Caso contrário, utiliza o valor da coluna `suit`.

**Montagem das features (X)**

- Categóricas: valores ausentes preenchidos com `desconhecido`.
- Numérica: conversão com `to_numeric`; valores ausentes tratados pelo imputador do pipeline.
- Texto: listas achatadas, células vazias convertidas em strings vazias e todas as colunas textuais, incluindo variantes `_reversed`, concatenadas em uma única coluna.

**Pré-processamento**

O pré-processamento é executado dentro do `Pipeline`, sendo ajustado exclusivamente com os dados de treinamento de cada fold.

| Bloco | KNN | Bernoulli Naive Bayes |
|---|---|---|
| Categóricas | `OneHotEncoder` | `OneHotEncoder` |
| `number_numerology` | `SimpleImputer(median)` + `StandardScaler` | `SimpleImputer(most_frequent)` + `OneHotEncoder` |
| Texto | `TfidfVectorizer` (1-2 gramas, `min_df=2`, `sublinear_tf`, 500 termos) | `CountVectorizer(binary=True)` (1-2 gramas, `min_df=2`) |

**Classificadores**

- **KNN:** `metric='cosine'` e `algorithm='brute'`.
- **Bernoulli Naive Bayes:** `binarize=None`, pois todas as features são representadas por valores binários.

**Validação cruzada aninhada**

A avaliação utiliza dois níveis de validação:

1. **Validação interna:** `GridSearchCV` com `StratifiedKFold(5, shuffle=True, random_state=42)` para selecionar os melhores hiperparâmetros.
2. **Validação externa:** `cross_val_score` com `StratifiedKFold(5, shuffle=True, random_state=7)` para medir a acurácia em dados não utilizados na seleção dos hiperparâmetros.

Dessa forma, a escolha dos hiperparâmetros nunca tem acesso ao fold de teste externo.

**Grades de hiperparâmetros**

| Modelo | Parâmetros testados |
|---|---|
| KNN | `n_neighbors = 3, 5, 7, 9`; `weights = uniform, distance`; pesos dos blocos (cat, num, txt) = `1/1/1`, `2/1/1`, `1/1/2` |
| Bernoulli NB | `alpha = 0.1, 0.3, 0.5, 1.0, 2.0`; `fit_prior = True, False`; `max_features = 200, 500, 1000` |

#### Demais scripts (fluxo original)

Os demais scripts mantêm o fluxo original de processamento.

**Carga dos dados**

- `json.load` + `pd.json_normalize` para o arquivo JSON.
- `pd.read_excel` para a planilha.

**Definição do rótulo**

- `major` quando `arcana == 'major'`.
- Caso contrário, utiliza o valor da coluna `suit`.

**Engenharia de features**

- Colunas categóricas: `fillna('unknown')`.
- Coluna numérica: `to_numeric` + `fillna(median)`.
- Colunas textuais: listas achatadas com `join` e concatenadas em uma única coluna `text_blob`.

**Pré-processamento**

Utilização de `ColumnTransformer`, combinando:

- `OneHotEncoder(handle_unknown='ignore')` para variáveis categóricas.
- `StandardScaler`, `MinMaxScaler` ou `passthrough` para a variável numérica.
- `TfidfVectorizer(max_features=100, stop_words='english')` para a coluna `text_blob`, exclusivamente no KNN.

**Validação**

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` em `Naive-Bayes-KNN` e nos scripts de comparação.
- `train_test_split(test_size=0.25, stratify=y, random_state=42)` nos scripts de árvore e Random Forest.

Esses scripts utilizam `StratifiedKFold` com divisão por índice (`skf.split`) e criam um novo `ColumnTransformer` a cada fold, evitando vazamento de dados entre treinamento e teste.

---

### 2. Séries temporais

**Notebook:** `main/Regressao/series-temporais.ipynb`

O objetivo é prever o valor de fechamento das ações do Yahoo! em um horizonte de 15 passos, comparando três modelos com o mesmo esquema de avaliação.

**Carga dos dados**

- Utilização de `pd.read_csv` para carregar `yahoo_stock.csv`.
- Remoção das colunas `High`, `Low`, `Open`, `Volume` e `Adj Close`.
- Manutenção exclusiva da coluna `Close`.

**Backtest**

Para cada combinação de parâmetros:

1. Os últimos 15 pontos da série são removidos.
2. O modelo é treinado com os dados restantes.
3. São realizadas previsões para os 15 passos seguintes.
4. As previsões são comparadas com os valores reais.

A métrica utilizada é o **MAPE (Mean Absolute Percentage Error)**.

**Modelos e grades de parâmetros**

| Modelo | Tipo | Grade testada |
|---|---|---|
| ETS (`statsmodels.tsa.exponential_smoothing.ets.ETSModel`) | Estatístico | `error ∈ {add, mul}`, `trend ∈ {add, mul}`, `seasonal ∈ {add, mul}`, `initialization ∈ {estimated, heuristic}`, `seasonal_periods = 18` |
| ARIMA (`statsmodels.tsa.arima.model.ARIMA`) | Estatístico | `order=(p,d,q)`, com `p ∈ {0,1,2}`, `d ∈ {0,1}`, `q ∈ {0,1,2}` |
| SVR (`sklearn.svm.SVR`) | Machine Learning | `kernel ∈ {rbf, linear}`, `C ∈ {1,10,100}`, `epsilon ∈ {0.01,0.1}`, `lag ∈ {5,10,20}` |

**Funcionamento do SVR**

O SVR transforma a série temporal em um problema de aprendizado supervisionado:

- Cada amostra utiliza uma janela de observações anteriores (`lag`) para prever o próximo valor.
- É aplicado `MinMaxScaler` em X e y.
- A previsão é recursiva: cada valor previsto é incorporado à janela utilizada para realizar a próxima previsão.

**Avaliação**

- Gráfico comparativo das previsões dos três modelos.
- Análise dos resíduos utilizando o teste de Jarque-Bera.
- Avaliação da estatística, p-valor, assimetria e curtose.

**Correções de compatibilidade**

- Utilização de indexação posicional com `.iloc`, compatível com pandas 3.x.
- Alinhamento do índice das previsões às datas reais para a geração dos gráficos.

---

### 3. Clusterização

**Notebook:** `main/Regressao/clusterizacao.ipynb`

O objetivo é aplicar K-Means e DBSCAN sobre a base das cartas, sem utilizar a coluna de classe, para verificar se os agrupamentos encontrados correspondem às classes originais.

**Preparação das features**

O pré-processamento segue a mesma estratégia dos scripts de classificação:

- `OneHotEncoder` para variáveis categóricas.
- `StandardScaler` para `number_numerology`.
- `TfidfVectorizer` para o texto concatenado, utilizando 1-2 gramas, `min_df=2`, `sublinear_tf` e 500 termos.

Como o resultado apresenta alta dimensionalidade e representação esparsa (aproximadamente 538 features), são aplicadas:

1. `TruncatedSVD` com 10 componentes.
2. `StandardScaler` antes da clusterização.

**K-Means**

- `n_clusters = 5`, correspondente ao número de classes originais.
- `n_init = 20`.
- `random_state = 42`.

**DBSCAN**

O parâmetro `eps` é escolhido automaticamente por meio da curva *k-distance*:

1. Utilização de `NearestNeighbors`, com `k=5`.
2. Teste de uma grade de percentis.
3. Seleção do primeiro valor que produz entre 2 e 10 clusters, com menos de 40% de ruído.

**Análises gráficas**

- Dispersão 2D utilizando SVD com dois componentes, colorindo os pontos de acordo com os clusters encontrados pelo K-Means e DBSCAN.
- Heatmap de correlação de Pearson entre os dummies de cluster e os dados (componentes SVD + `number_numerology`).
- Heatmap do cruzamento percentual entre cluster e classe original.

**Avaliação**

- Correspondência cluster → classe por meio de atribuição ótima, utilizando `scipy.optimize.linear_sum_assignment`.
- Acurácia de correspondência.
- Adjusted Rand Index (ARI).

**Resultados observados**

| Algoritmo | Correspondência | ARI |
|---|---:|---:|
| K-Means | Aproximadamente 56% | 0,235 |
| DBSCAN | Aproximadamente 40% | 0,160 |

O K-Means apresentou clusters quase puros para `cups` (91%) e `pentacles` (73%).

O DBSCAN encontrou dois clusters densos e classificou aproximadamente 23% das cartas como ruído.

---

## Modelos comparados

| Script | Modelo(s) | Hiperparâmetros | Validação |
|---|---|---|---|
| `main/KNN/main.py` | KNN | Grade descrita anteriormente, distância cosseno | 5-fold aninhado |
| `main/Bernoulli-NB/main.py` | Bernoulli NB | Grade descrita anteriormente | 5-fold aninhado |
| `main/Comparações/knn_bnb.py` | KNN + Bernoulli NB | Grades anteriores, mesmos folds externos | 5-fold aninhado |
| `main/Naive-Bayes-KNN/main.py` | KNN + Naive Bayes | `k = 3, 5, 7`; `MultinomialNB()` | 5-fold |
| `main/decision-tree/main.py` | Árvore de Decisão | `criterion='gini'`, `max_depth=3` | Hold-out 25% |
| `main/decision-tree/random -forest/main.py` | Random Forest | `n_estimators=200`, `n_jobs=-1` | Hold-out 25% + 5-fold |
| `main/Comparações/knn_dt.py` | Árvore + KNN | `max_depth=3`; `k=5` | Hold-out 25% + 5-fold |
| `main/Comparações/knn_dt_nb_trees.py` | NB + Árvore + KNN | `BernoulliNB(alpha=1.0)`; `max_depth=3`; `k=5` | Hold-out 25% + 5-fold |
| `main/Comparações/knn_dt__nb_trees_forest.py` | NB + Árvore + RF + KNN | Parâmetros anteriores + `n_estimators=200` | Hold-out 25% + 5-fold |
| `main/Regressao/series-temporais.ipynb` | ETS + ARIMA + SVR | Grades descritas anteriormente | Backtest 15 passos / MAPE |
| `main/Regressao/clusterizacao.ipynb` | K-Means + DBSCAN | `k=5`; `eps` por k-distance | Atribuição ótima + ARI |

### Observação sobre o Naive Bayes

O Naive Bayes exige features binárias no caso do Bernoulli NB.

Nos scripts antigos (`knn_dt_nb_trees.py` e `knn_dt__nb_trees_forest.py`), isso é resolvido com um `preprocessador_nb` dedicado, que aplica:

`StandardScaler` → `Binarizer(threshold=0.0)` na coluna numérica.

Na versão reescrita (`Bernoulli-NB`), a variável numérica é transformada em categoria com `OneHotEncoder`, enquanto o texto utiliza `CountVectorizer(binary=True)`.

### Visualização da árvore

O arquivo `main/decision-tree/tree.py` apenas plota a árvore treinada em `main/decision-tree/main.py`, utilizando `plot_tree` e `matplotlib`.

---

## Como executar

### Pré-requisitos

- Python 3.13 (o ambiente `main/libs` foi criado com Python 3.13.7).

### Instalação

Clone o repositório:

```bash
git clone https://github.com/rafael-men/algorithms-ML.git
```

Entre na pasta:

```bash
cd tarot-cards-algorithms-machine-learning
```

Crie o ambiente virtual:

```bash
python -m venv main/libs
```

Ative o ambiente virtual.

**Windows:**

```bash
main\libs\Scripts\activate
```

**Linux / WSL:**

```bash
source main/libs/bin/activate
```

Instale as dependências:

```bash
pip install -r main/requirements.txt
```

### Dependências

O arquivo `main/requirements.txt` contém:

- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn
- openpyxl (necessário para leitura de arquivos `.xlsx`)
- statsmodels (séries temporais)
- scipy (clusterização)

### Executando os scripts

Os scripts reescritos (KNN, Bernoulli NB e `Comparações/knn_bnb.py`) montam o caminho do dataset a partir da localização do próprio arquivo. Portanto, podem ser executados de qualquer diretório.

Os demais scripts ainda utilizam caminhos relativos ao diretório de trabalho. Execute-os a partir da pasta indicada na tabela para evitar erros `FileNotFoundError`.

| Script | Executar a partir de | Comando |
|---|---|---|
| KNN | Qualquer pasta | `python main/KNN/main.py` |
| Bernoulli-NB | Qualquer pasta | `python main/Bernoulli-NB/main.py` |
| Comparações/knn_bnb.py | Qualquer pasta | `python main/Comparações/knn_bnb.py` |
| Naive-Bayes-KNN | `main/Naive-Bayes-KNN/` | `python main.py` |
| decision-tree/main.py | `main/` | `python decision-tree/main.py` |
| decision-tree/tree.py | `main/` | `python decision-tree/tree.py` |
| random-forest/main.py | `main/decision-tree/random -forest/` | `python main.py` |
| Comparações/knn_dt.py | `main/Comparações/` | `python knn_dt.py` |
| Comparações/knn_dt_nb_trees.py | `main/Comparações/` | `python knn_dt_nb_trees.py` |
| Comparações/knn_dt__nb_trees_forest.py | `main/Comparações/` | `python knn_dt__nb_trees_forest.py` |
| series-temporais.ipynb | Jupyter / VS Code | Executar células em ordem |
| clusterizacao.ipynb | Jupyter / VS Code | Executar células em ordem |

**Observações importantes**

- `decision-tree/tree.py` importa `main` por meio de `from main import preprocessador, classificador`. Portanto, depende de `main/decision-tree/main.py` e precisa de um backend interativo do Matplotlib para exibir a figura.
- Os notebooks em `Regressao/` utilizam caminhos absolutos para o dataset (`C:\Users\rafaelmg\Downloads\algorithms-ML\main\dataset\...`). Ajuste esses caminhos caso o repositório seja clonado em outra máquina.
- O notebook `series-temporais.ipynb` utiliza statsmodels (ETS e ARIMA) e pode demorar alguns minutos durante a execução da grade de parâmetros do SVR.
- Os scripts de KNN e Bernoulli NB executam busca em grade dentro da validação aninhada e podem levar mais tempo que os demais.

---

## Conclusões

As conclusões de 1 a 5 foram obtidas nas execuções originais e devem ser revisadas quando os resultados da versão reescrita forem preenchidos.

1. **Random Forest apresentou maior estabilidade nas execuções originais.** Liderou tanto na validação 5-fold (48,83%) quanto no hold-out, sendo também o único modelo com macro F1 acima de 0,48.

2. **KNN é altamente sensível à representação dos dados.** Com TF-IDF + One-Hot + passthrough, o `k=5` atingiu 48,83%. Com One-Hot + StandardScaler, o mesmo `k=5` caiu para 32,08%. A distância euclidiana pode ser distorcida por vetores esparsos de texto, demonstrando que a escolha do pré-processamento é tão importante quanto a escolha do valor de `k`. A versão reescrita testa essa hipótese com distância cosseno e escala na coluna numérica.

3. **Naive Bayes se beneficia de features binárias.** A utilização de `Binarizer` na coluna numérica alcançou 42,50%, contra 33,33% do MultinomialNB com features contínuas.

4. **A árvore de decisão podada apresentou desempenho insuficiente.** Com `max_depth=3`, atingiu 29,50%, ficando abaixo do baseline da classe majoritária. O ensemble de árvores ajuda a reduzir o problema de viés associado ao corte excessivo.

5. **O volume de dados é o principal gargalo.** Com apenas 78 amostras e features majoritariamente categóricas de baixa cardinalidade, o teto de aproximadamente 50% de acurácia reflete mais a qualidade dos atributos disponíveis do que a escolha do classificador.

6. **A validação aninhada fornece estimativas menos otimistas.** Escolher `k`, `alpha` e os pesos dos blocos observando o próprio resultado do fold pode inflar a acurácia. Na versão reescrita, essa escolha ocorre exclusivamente dentro do conjunto de treinamento de cada fold externo.

### Séries temporais e clusterização

**Séries temporais**

Séries financeiras são difíceis de prever. O notebook `series-temporais.ipynb` compara ETS, ARIMA e SVR em um backtest de 15 passos, com seleção de parâmetros por MAPE.

A análise de resíduos via Jarque-Bera indica curtose elevada, característica observada em séries financeiras. Nenhum dos modelos captura a volatilidade de forma completamente satisfatória.

**Clusterização**

Os algoritmos não supervisionados não necessariamente recuperam classes definidas previamente.

No notebook `clusterizacao.ipynb`, mesmo utilizando `n_clusters = 5`, a correspondência entre clusters e classes ficou em aproximadamente 56% para K-Means e 40% para DBSCAN.

As classes `cups` e `pentacles` formaram grupos quase puros, enquanto `major`, `wands` e `swords` apresentaram maior mistura. Isso indica que a divisão por naipe não representa necessariamente a estrutura dominante nos atributos textuais das cartas.

---

## Estrutura do projeto

```text
.
├── main/
│   ├── dataset/
│   │   ├── cards.json
│   │   ├── Dataset-Sheet.xlsx
│   │   └── yahoo_stock.csv
│   │
│   ├── KNN/
│   │   └── main.py
│   │
│   ├── Bernoulli-NB/
│   │   └── main.py
│   │
│   ├── Naive-Bayes-KNN/
│   │   └── main.py
│   │
│   ├── decision-tree/
│   │   ├── main.py
│   │   ├── tree.py
│   │   └── random -forest/
│   │       └── main.py
│   │
│   ├── Regressao/
│   │   ├── series-temporais.ipynb
│   │   └── clusterizacao.ipynb
│   │
│   ├── Comparações/
│   │   ├── knn_bnb.py
│   │   ├── knn_dt.py
│   │   ├── knn_dt_nb_trees.py
│   │   └── knn_dt__nb_trees_forest.py
│   │
│   ├── Perceptrons-RNA/
│   │   └── mlp.py
│   │
│   ├── libs/
│   └── requirements.txt
│
└── .gitignore
```

**Descrição dos principais diretórios:**

- `dataset/`: bases de dados utilizadas nos experimentos.
- `KNN/`: implementação do KNN com Pipeline, distância cosseno e validação cruzada aninhada.
- `Bernoulli-NB/`: implementação do Bernoulli Naive Bayes com features binárias.
- `Naive-Bayes-KNN/`: comparação entre KNN e Multinomial Naive Bayes.
- `decision-tree/`: implementação e visualização da árvore de decisão e Random Forest.
- `Regressao/`: experimentos de séries temporais e clusterização.
- `Comparações/`: experimentos comparativos entre diferentes classificadores.
- `Perceptrons-RNA/`: experimentos de classificação utilizando redes neurais artificiais.
- `libs/`: ambiente virtual Python (não versionado).

---

