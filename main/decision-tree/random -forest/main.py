import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

dataset = '../../dataset/Dataset-Sheet.xlsx'
df = pd.read_excel(dataset)
categorias = ['planet', 'zodiac', 'yes_no', 'yes_no_reversed']
numericas = ['number_numerology']

def classes(row):
      if row.get('arcana') == 'major':
         return 'major'
      return row.get('suit')

df['label'] = df.apply(classes, axis=1)

features = df[categorias + numericas].copy()
y = df['label'].values

for c in categorias:
    features[c] = features[c].fillna('unknown')
for c in numericas:
    features[c] = pd.to_numeric(features[c], errors='coerce')
    features[c] = features[c].fillna(features[c].median())

cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
preprocessador= ColumnTransformer([('cat', cat_pipeline, categorias),('num', num_pipeline, numericas)])
forest = Pipeline([('preproc', preprocessador),('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])

X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.25, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acuracia_cv = cross_val_score(forest, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
macro_cv = cross_val_score(forest, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)

print("Acurácia da random forest: %.4f ± %.4f" % (acuracia_cv.mean(), acuracia_cv.std()))
print("Acurácia f1 Macro: %.4f ± %.4f" % (macro_cv.mean(), macro_cv.std()))

forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

print("\nResultados:")
print("Accuracy: %.4f" % accuracy_score(y_test, y_pred))
print("F1 macro: %.4f" % f1_score(y_test, y_pred, average='macro'))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

ohe = forest.named_steps['preproc'].named_transformers_['cat'].named_steps['onehot']
ohe_features = ohe.get_feature_names_out(categorias)
todas_features = list(ohe_features) + numericas

importances = forest.named_steps['model'].feature_importances_
if len(importances) == len(todas_features):
    feat_imp = sorted(zip(todas_features, importances), key=lambda x: x[1], reverse=True)
    print("\nimportâncias das features:")
    for name, imp in feat_imp[:20]:
        print(f"{name}: {imp:.4f}")
else:
    print("\nNão foi possível mapear importâncias às features pois as dimensões divergem.")