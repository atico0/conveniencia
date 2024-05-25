from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


#modelos
def treinar_todos(x, y, modelos, funcs_perda):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
  perda = {}

  for i in modelos.keys():
    modelos[i].fit(x_train, y_train)
    y_pred = modelos[i].predict(x_test)

    lista = []
    for j in funcs_perda.keys():
      lista.append(funcs_perda[j](y_test, y_pred))

    perda[i] = lista
  metricas = pd.DataFrame(perda)
  metricas.index = list(funcs_perda.keys())
  return(metricas.transpose())



def pcfacil(df, n_comp, variaveis=[]):
  """
  reliza uma PCA e retorna um dataframe com as componentes principais e
  retorna um df com os componentes principais e as variáveis não incluidas

  df: dataframe
  n_comp: inteiro, quantidade de componentes
  variáveis: lista de variáveis para serem incluidas no PCA 
  (caso nada seja imputado, usa a função pega_tipos)
  test: dataframe, dataframe de teste para caso o argumento df seja o dataframe de treino

  """


  if len(variaveis)==0:
    cat, numericas = pega_tipos(df)
    lista_cat = list(cat.keys())
    lista_numericas = list(numericas.keys())
    
  else:
    lista_numericas = variaveis
    lista_cat = []
    for j in range(df.shape[1]):
      if j not in lista_numericas:
        lista_cat.append(j)

  pca = PCA(n_components=n_comp)
  df_pca = pca.fit_transform(df.iloc[:, lista_numericas])
  lista_nomes = []
  for i in range(1, n_comp+1):
    lista_nomes.append(f"componente_{i}")
  df_pca = pd.DataFrame(df_pca, columns=lista_nomes)
  df_pca = pd.concat([df.iloc[:, lista_cat], df_pca], axis=1)

  return(df_pca, pca)


