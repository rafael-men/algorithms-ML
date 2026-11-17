import csv
from pathlib import Path
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

raiz = Path(__file__).resolve().parents[2]
dataset = raiz / "dataset" / "dataset_transacional_de_compras_apriori.txt"

confianca_min = 0.7
min_compras = 50
min_sup = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30]

def carregar_transacoes(caminho):
    transacoes = []
    with open(caminho, encoding="utf-8", newline="") as f:
        for linha in csv.reader(f, skipinitialspace=True):
            itens = [item.strip() for item in linha[1:] if item.strip()]
            if itens:
                transacoes.append(itens)
    return transacoes

transacoes = carregar_transacoes(dataset)
N = len(transacoes)
print(f"Transações carregadas: {N}")

te = TransactionEncoder()
df_encoded = pd.DataFrame(
    te.fit(transacoes).transform(transacoes),
    columns=te.columns_
)

print("\n Dados binários (primeiras linhas, primeiras colunas)")
print(df_encoded.iloc[:5, :5])

print("\nComparação de min_sup")
resultados = []
for ms in min_sup:
    fi = apriori(df_encoded, min_support=ms, use_colnames=True)
    regras = (
        association_rules(fi, metric="confidence", min_threshold=confianca_min)
        if not fi.empty
        else pd.DataFrame()
    )
    n_regras = len(regras)
    if n_regras:
        menor_apoio = int((regras["support"] * N).round().min())
        valido = menor_apoio >= min_compras
    else:
        menor_apoio = None
        valido = False

    resultados.append({
        "min_support": ms,
        "itemsets": len(fi),
        "regras": n_regras,
        "menor_apoio": menor_apoio,
        "valido": valido,
    })
    print(
        f"min_support={ms:.2f} | itemsets={len(fi)} | regras={n_regras} | "
        f"menor apoio de regra={menor_apoio} | válido={valido}"
    )

validos = [r for r in resultados if r["valido"]]
if not validos:
    raise SystemExit(
        f"Nenhum min_support gera regras com pelo menos {min_compras} transações de apoio."
    )

melhor = max(validos, key=lambda r: (r["regras"], r["min_support"]))
min_sup = melhor["min_support"]

# Justificativa
print(
    f"\nmin_support escolhido: {min_sup:.2f}\n"
    f"Justificativa: é o valor com mais regras válidas ({melhor['regras']}). "
    f"A menor regra é apoiada por {melhor['menor_apoio']} de {N} transações "
    f"(mínimo exigido: {min_compras}). "
    f"Entre os valores empatados, é o maior, o que gera {melhor['itemsets']} itemsets "
    f"e evita combinações raras."
)

frequent_itemsets = apriori(
    df_encoded,
    min_support=min_sup,
    use_colnames=True
)
frequent_itemsets["tamanho"] = frequent_itemsets["itemsets"].apply(len)

print(f"\n Itemsets frequentes (min_support={min_sup:.2f})")
print(frequent_itemsets.sort_values("support", ascending=False).to_string())

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=confianca_min
)

for coluna in ["antecedents", "consequents"]:
    rules[coluna] = rules[coluna].apply(lambda s: ", ".join(sorted(s)))

rules = rules.sort_values("lift", ascending=False)

print(f"\nRegras encontradas (confidence >= {confianca_min})")
print(
    rules[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].to_string(index=False)
)
