import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

base = pd.read_excel('dataset/Dataset-Sheet.xlsx')
categorias = [
    'planet',
    'zodiac',
    'yes_no',
    'yes_no_reversed'
]
numericas = [
    'number_numerology'
]
textuais = [
    'keywords_upright',
    'keywords_reversed',
    'love',
    'career',
    'mood'
]

def classes(row):
    if row['arcana'] == 'major':
        return 'major'
    return row['suit']

base['label'] = base.apply(classes, axis=1)

features = base[categorias + numericas + textuais].copy()
y=base['label']
print('features: ', len(categorias + numericas + textuais))
print('\nClasses: ', y.value_counts())

for coluna in textuais:
   features[coluna] = (
   features[coluna].fillna('').astype(str)
   )

pipeline = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='most_frequent'
            )
        ),

        (
            'onehot',
            OneHotEncoder(
                handle_unknown='ignore'
            )
        )
    ]
)

pipeline_num = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='median'
            )
        ),

        (
            'scaler',
            StandardScaler()
        )
    ]
)

preprocessador = ColumnTransformer(
    transformers=[
        (
            'cat',
            pipeline,
            categorias
        ),

        (
            'num',
            pipeline_num,
            numericas
        ),

        (
            'keywords_up',
            TfidfVectorizer(
                stop_words='english',
                max_features=20
            ),
            'keywords_upright'
        ),

        (
            'keywords_rev',
            TfidfVectorizer(
                stop_words='english',
                max_features=20
            ),
            'keywords_reversed'
        ),

        (
            'love',
            TfidfVectorizer(
                stop_words='english',
                max_features=20
            ),
            'love'
        ),

        (
            'career',
            TfidfVectorizer(
                stop_words='english',
                max_features=20
            ),
            'career'
        ),

        (
            'mood',
            TfidfVectorizer(
                stop_words='english',
                max_features=20
            ),
            'mood'
        )
    ]
)
X_train, X_test, y_train, y_test = train_test_split(features, y,test_size=0.25, random_state=42, stratify=y)

svc = Pipeline(
    steps=[
         (
               'preprocessor',
               preprocessador
         ),
   
         (
               'modelo',
               SVC()
         )
    ]
)

cross = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
parametros = {
    'modelo__C': [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    'modelo__kernel': [
        'linear',
        'rbf',
        'poly',
        'sigmoid'
    ]
}

gs= GridSearchCV(
    estimator=svc,
    param_grid=parametros,
    scoring='accuracy',
    cv=cross,
    n_jobs=-1
)

gs.fit(X_train, y_train)

print('\nMelhores parâmetros:')
print( 'C:', gs.best_params_['modelo__C'])
print('Kernel:', gs.best_params_['modelo__kernel'])
print('\nMelhor acurácia: %.2f%%'% (gs.best_score_ * 100))

melhor = gs.best_estimator_
previsoes = melhor.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)
print('\nAcurácia de teste: %.2f%%' % ( acuracia * 100 ))

print(
    '\nClassification Report:'
)
print( classification_report(  y_test,  previsoes, zero_division=0 ))

classes_svc = (
    melhor
    .named_steps['modelo']
    .classes_
)
print('Matriz de confusão:\n',confusion_matrix( y_test, previsoes, labels=classes_svc))
