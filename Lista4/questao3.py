import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Carregue os dados do arquivo CSV
df = pd.read_csv('dados.csv')

# Converta variáveis categóricas em numéricas
label_encoder = LabelEncoder()
df['Aparência'] = label_encoder.fit_transform(df['Aparência'])
df['Temperatura'] = label_encoder.fit_transform(df['Temperatura'])
df['Umidade'] = label_encoder.fit_transform(df['Umidade'])
df['Ventando'] = label_encoder.fit_transform(df['Ventando'])
df['Jogar'] = label_encoder.fit_transform(df['Jogar'])

# Separe os atributos e a variável de destino
X = df.drop('Jogar', axis=1)
y = df['Jogar']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie e treine o modelo Naive Bayes
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

# Crie e treine o modelo Árvore de Decisão
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Crie e treine o modelo Random Forest
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Faça previsões nos dados de teste
naive_bayes_predictions = naive_bayes_model.predict(X_test)
decision_tree_predictions = decision_tree_model.predict(X_test)
random_forest_predictions = random_forest_model.predict(X_test)

# Calcule a acurácia dos modelos
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)

print(f'Acurácia do Naive Bayes: {naive_bayes_accuracy}')
print(f'Acurácia da Árvore de Decisão: {decision_tree_accuracy}')
print(f'Acurácia do Random Forest: {random_forest_accuracy}')

# Define os hiperparâmetros que deseja ajustar
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearch
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_random_forest = grid_search.best_estimator_

# RandomSearch
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=10, cv=5)
random_search.fit(X_train, y_train)
best_random_forest_random = random_search.best_estimator_

# Avalie o desempenho dos melhores modelos após a busca de hiperparâmetros
best_random_forest_accuracy = best_random_forest.score(X_test, y_test)
best_random_forest_random_accuracy = best_random_forest_random.score(X_test, y_test)

print(f'Melhor Acurácia do Random Forest (GridSearch): {best_random_forest_accuracy}')
print(f'Melhor Acurácia do Random Forest (RandomSearch): {best_random_forest_random_accuracy}')


