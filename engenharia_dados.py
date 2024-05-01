import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


#dados ordinais
def labelencoder(df, lista):
  for i in lista:
    df[:,i] = LabelEncoder().fit_transform(df.iloc[:,i])
  return(df)

#dados nominais
def onehotencoder(df, lista):
  for i in lista:
    novo = pd.get_dummies(df.loc[:, i], prefix=i).astype(int)
    df = pd.concat([df, novo], axis=1)
    df.drop([i], axis=1, inplace=True)
  return df



#dados quantitativos
from sklearn.preprocessing import StandardScaler
def padronizar(df, lista):
  for i in lista:
    df.loc[:,i] = StandardScaler().fit_transform(df.loc[:,i].values.reshape(-1,1))
  return df

#pega tipo das variáveis
def pega_tipos(df):
  numericos = {}
  categoricos = {}
  for i in range(df.shape[1]):
    if str(df.iloc[:,i].dtype) == "object":
      categoricos[i] = df.columns[i]
    else:
      numericos[i] = df.columns[i]

  return (categoricos, numericos)



def pcfacil(df, n_comp, variaveis=[], test=[]):

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

  if len(test) != 0:
    test_pca = pca.transform(test.iloc[:, lista_numericas])
    test_pca = pd.DataFrame(test_pca, columns=lista_nomes)
    df_pca = pd.concat([test.iloc[:, lista_cat], test_pca], axis=1)
    return([df_pca, test, pca])
  return(df_pca, pca)

