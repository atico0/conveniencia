from sklearn.model_selection import train_test_split
from conveniencia.eng_dados import pega_tipos
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



#modelos
def treinar_todos(x, y, modelos, funcs_perda, test_size, plot=True):
  """
  x: dataframe com as variáveis independentes do modelo

  y: dataframe com a variável resposta do modelo
  modelos: dicionário contendo o nome dos modelos a serem usados 
  (o nome é o que vai aparecer no dataframe no final) e os objetos dos modelos

  funcs_perda: dicionário contendo o nome das funções de perda a serem usados 
  (o nome é o que vai aparecer no dataframe no final) e as funções de perda
  test_size: valor real que contém a proporção de dados que será usada no teste dos dados

  plot: booleano indicando se a função deve plotar gráficos de barras com os desempenhos de cada modelo

  Returns:
  Retorna um dataframe onde cada linha corresponde a um modelo treinado e cada coluna
  corresponde a uma função de perda
  """
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
  perda = {}

  # Treinando os modelos
  for i in modelos.keys():
    modelos[i].fit(x_train, y_train)
    y_pred = modelos[i].predict(x_test)

    # Calculando as perdas para cada modelo
    lista = []
    for j in funcs_perda.keys():
      lista.append(funcs_perda[j](y_test, y_pred))

    perda[i] = lista
  metricas = pd.DataFrame(perda)
  metricas.index = list(funcs_perda.keys())

  metricas = metricas.transpose()
  metricas_semindex = metricas.reset_index()
  quant_colunas = metricas_semindex.shape[1]
  if plot:
    barras_y(metricas_semindex, list(1, range(quant_colunas)), "index" (1,quant_colunas-1))

  return metricas



def pcfacil(dados, n_comp, colunas=[], numerica=True):
  """
  reliza uma PCA e retorna um dataframe com as componentes principais e
  retorna um df com os componentes principais e as variáveis não incluidas

  df: dataframe em que será realizado o PCA
  n_comp: inteiro, quantidade de componentes
  colunas: lista númerica representando as coordenadas das variáveis ou strings com os nomes delas
  (caso nada seja imputado, usa a função pega_tipos)
  numerica: Booleano indicando se a lista de variáveis é numerica ou strings
  
  test: dataframe, dataframe de teste para caso o argumento df seja o dataframe de treino

  Returns:
        pandas.DataFrame: O dataframe original mas com as variáveis que foram usadas no PCA 
        retiradas e os componentes principais incluidos

        PCA: O objeto usado para o PCA

  """

  # checando se foram passadas variáveis para o PCA ou se devo usar a função pega_tipos
  # para escolher quais variáveis vão ser usadas no PCA

  df = dados.copy()

  if len(colunas)==0:
    cat, numericas = pega_tipos(df)
    lista_cat = list(cat.keys())
    lista_numericas = list(numericas.keys())

  else:

    if numerica:
      lista_numericas = df.columns[colunas]
    else:
      lista_numericas = colunas

    lista_cat = []

    for j in range(df.shape[1]):
      if df.columns[j] not in lista_numericas:
        lista_cat.append(df.columns[j])

  # Fazeno o PCA
  pca = PCA(n_components=n_comp)
  df_pca = pca.fit_transform(df.loc[:, lista_numericas])
  lista_nomes = []

  #Atribuindo nomes os componentes
  for i in range(1, n_comp+1):
    lista_nomes.append(f"componente_{i}")

  # Criando data frame com os componentes e as demais variáveis
  df_pca = pd.DataFrame(df_pca, columns=lista_nomes)
  df_pca = pd.concat([df.loc[:, lista_cat], df_pca], axis=1)

  plt.plot(pca.explained_variance_ratio_)

  return(df_pca, pca)


