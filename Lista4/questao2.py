import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('dados.csv')
df['Aparência'] = df['Aparência'].astype('category').cat.codes
df['Temperatura'] = df['Temperatura'].astype('category').cat.codes
df['Umidade'] = df['Umidade'].astype('category').cat.codes
df['Ventando'] = df['Ventando'].astype('category').cat.codes

gnb = GaussianNB()

X = df.drop('Jogar', axis=1)

y = df['Jogar']

gnb.fit(X, y)

novo_registro = [[2, 1, 1, 1]]  # Chuva, Fria, Normal, Sim

previsao = gnb.predict(novo_registro)

probabilidades = gnb.predict_proba(novo_registro)

resultado = 'Jogar' if previsao[0] == 1 else 'Não Jogar'

print(f'Probabilidade de Jogar: {probabilidades[0][1]}')
print(f'Probabilidade de Não Jogar: {probabilidades[0][0]}')
