# Projeto de IA para previsão da duração de um processo judicial no STJ
#Autor: Djacy Júnior
# Mestrado na UNB
#Abaixo o código utilizado para treinar o modelo no Google Colab

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Carregar os dados da planilha CSV
data = pd.read_csv('XXXXXXXXXXXX',
                   encoding='ISO-8859-1',
                   delimiter=';',
                   on_bad_lines='skip')  # Ignora linhas com problemas

# Exibir informações sobre o DataFrame
print("Dados carregados com sucesso.")
print(f"Total de linhas: {len(data)}")
print("Estrutura inicial dos dados:")
print(data.info())  # Mostra a estrutura do DataFrame

# Converter as colunas de data antes de qualquer operação
data['Data de inicio'] = pd.to_datetime(data['Data de inicio'])
data['Data de Referencia'] = pd.to_datetime(data['Data de Referencia'])

# Lista de colunas a serem excluídas
colunas_a_excluir = ['Tribunal', 'Recurso', 'Municipio', 'id_municipio', 'UF', 'Grau',
                     'Nome da ultima classe CN', 'ï»¿Ano', 'Mes', 'Codigo da Ultima classe CN']

# Excluir as colunas
data = data.drop(columns=colunas_a_excluir)

# Usar str.get_dummies() para criar as colunas one-hot
codigos_assuntos_encoded = data['Codigos assuntos'].str.strip('{}').str.split(',').str.join('|').str.get_dummies()
codigos_classes_encoded = data['Codigos Classes'].str.strip('{}').str.split(',').str.join('|').str.get_dummies()

# Concatenar as colunas codificadas ao DataFrame original
data_expanded = pd.concat([data, codigos_assuntos_encoded, codigos_classes_encoded], axis=1)

# Remover as colunas originais
data_expanded.drop(columns=['Codigos assuntos', 'Codigos Classes'], inplace=True)

# Calcular a duração do processo
data_expanded['Dias até Data de Referencia'] = (data_expanded['Data de Referencia'] - data_expanded['Data de inicio']).dt.days

# Armazenar o número inicial de linhas
num_linhas_iniciais = data_expanded.shape[0]

# Excluir linhas com valores ausentes na coluna 'Dias até Data de Referencia'
data_expanded = data_expanded.dropna(subset=['Dias até Data de Referencia'])

# Calcular o número de linhas excluídas
num_linhas_excluidas = num_linhas_iniciais - data_expanded.shape[0]
print(f"Número de linhas excluídas: {num_linhas_excluidas}")

# Analisando as durações extremas
extreme_cases = data_expanded[data_expanded['Dias até Data de Referencia'] > 1000]
print("Casos extremos com mais de 1000 dias:")
print(extreme_cases)

# Visualizando a distribuição das durações
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(x=data_expanded['Dias até Data de Referencia'])
plt.title('Boxplot da Duração dos Processos Judiciais')
plt.xlabel('Dias até Data de Referencia')
plt.grid(True)
plt.show()

# Extração de features sazonais e categóricas
data_expanded['Ano_Inicio'] = data_expanded['Data de inicio'].dt.year
data_expanded['Mes_Inicio'] = data_expanded['Data de inicio'].dt.month
data_expanded['Dia_Semana_Inicio'] = data_expanded['Data de inicio'].dt.dayofweek

# Criar a nova coluna de combinação
data_expanded['Codigo orgao_Procedimento'] = data_expanded['Codigo orgao'].astype(str) + "_" + data_expanded['Procedimento']
data_expanded['Frequencia_Tribunal'] = data_expanded['Codigo orgao'].map(data_expanded['Codigo orgao'].value_counts(normalize=True))

# Definir X e y
X = data_expanded.drop(columns=['Dias até Data de Referencia', 'Nome orgao', 'Processo', 'Data de Referencia'])
y = data_expanded['Dias até Data de Referencia']

# Verificar a distribuição de 'Dias até Data de Referencia'
print(data_expanded['Dias até Data de Referencia'].describe())

# Contar valores únicos em 'Dias até Data de Referencia'
print("Valores únicos de 'Dias até Data de Referencia':")
print(data_expanded['Dias até Data de Referencia'].value_counts())

# Exibir o número total de linhas antes da amostragem
print(f"Número total de linhas antes da amostragem: {X.shape[0]}")

# Selecionar aleatoriamente um percentual do DataFrame para treinar o modelo
percentual = 0.10  # 10%
X = X.sample(frac=percentual, random_state=42)
y = y[X.index]  # Manter as mesmas amostras de y

# Informar o número de linhas após a amostragem
print(f"Número de linhas após a amostragem: {X.shape[0]}")

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Informações sobre o número de linhas e colunas
print(f"Número de linhas em X_train: {X_train.shape[0]}")
print(f"Número de colunas em X_train: {X_train.shape[1]}")

# Pipeline para tratamento de dados
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Definir o pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features + ['Codigo orgao_Procedimento'])  # Adicionando a nova coluna
    ])

# Criar um pipeline que inclui o pré-processador e o modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),  # Normalizar os dados após o pré-processamento
    ('model', MLPRegressor(random_state=42))
])

# Parâmetros a serem testados pelo RandomizedSearchCV
param_dist = {
    'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (50, 100), (100, 100)],  # Algumas combinações de camadas ocultas
    'model__alpha': [0.001, 0.01],  # Regularização L2
    'model__learning_rate_init': [0.001, 0.01],  # Taxa de aprendizado inicial
    'model__max_iter': [200, 300, 500, 1000]  # Iterações
}

# RandomizedSearchCV: Testa um número fixo de combinações aleatórias
random_search = RandomizedSearchCV(estimator=pipeline,
                                   param_distributions=param_dist,
                                   n_iter=10,  # Número de combinações a serem testadas
                                   cv=5,  # Validação cruzada
                                   n_jobs=-1,  # Usar todos os núcleos disponíveis
                                   scoring='r2',
                                   random_state=42)

# Executar o RandomizedSearchCV
random_search.fit(X_train, y_train)

# Exibir os melhores parâmetros
print("Melhores hiperparâmetros encontrados:")
print(random_search.best_params_)

# Criar um DataFrame com os resultados do RandomizedSearchCV
results = pd.DataFrame(random_search.cv_results_)

# Selecionar apenas as colunas relevantes para exibição
results_display = results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]

# Renomear as colunas para melhor clareza
results_display.columns = ['Combinação de Hiperparâmetros', 'Média do R²', 'Desvio Padrão do R²', 'Ranking']

# Exibir a tabela de desempenho completa
print("Desempenho de cada combinação feita pelo RandomizedSearchCV:")
print(results_display.to_string(index=False))  # Exibir a tabela completa

# Modelo otimizado
best_model = random_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)

# Avaliação do modelo otimizado
r2_optimized = r2_score(y_test, y_pred_optimized)
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
print(f'R² Otimizado: {r2_optimized}')
print(f'MAE Otimizado: {mae_optimized}')

# Calcular o erro absoluto
erro_absoluto = np.abs(y_test - y_pred_optimized)

# Calcular a precisão
precisao = (y_test - y_pred_optimized) / y_test * 100

# Exibir métricas de desempenho
metrics_df = pd.DataFrame({
    'Métrica': ['R² Otimizado', 'MAE Otimizado'],
    'Valor': [r2_optimized, mae_optimized]
})
print(metrics_df)

# Salvando o modelo otimizado
joblib.dump(best_model, 'modelo_otimizado.pkl')
