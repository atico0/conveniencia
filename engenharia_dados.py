import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#dados ordinais
def labelencoder(df, lista):
  """
  realiza o label encoder num conjunto de variáveis num dataframe

  df: dataframe
  lista: lista númerica de variáves
  """
  for i in lista:
    df.iloc[:,i] = LabelEncoder().fit_transform(df.iloc[:,i])
  return(df)

#dados nominais
def onehotencoder(df, lista, drop_first=True):
  """
  realiza o one hot encoder num conjunto de variáveis num dataframe

  df: dataframe
  lista: lista númerica de variáves
  """
  cols = df.columns[lista]
  for i in cols:
    novo = pd.get_dummies(df.loc[:, i], prefix=i, drop_first=drop_first).astype(int)
    df = pd.concat([df, novo], axis=1)
    df.drop([i], axis=1, inplace=True)
  return df



#dados quantitativos
from sklearn.preprocessing import StandardScaler
def padronizar(df, lista):
  """
  realiza a padrinização num conjunto de variáveis num dataframe

  df: dataframe
  lista: lista númerica de variáves
  """
  for i in lista:
    df.iloc[:,i] = StandardScaler().fit_transform(df.iloc[:,i].values.reshape(-1,1))
  return df

#pega tipo das variáveis
def pega_tipos(df):
  """
  retorna um dois dicionarios contendo as variáveis categoricas no primeiro
  e as numéricas no segundo, as chaves são as posições da variável no df
  e os valores não os nomes das variáveis

  df: dataframe
  """
  numericos = {}
  categoricos = {}
  for i in range(df.shape[1]):
    if str(df.iloc[:,i].dtype) == "object":
      categoricos[i] = df.columns[i]
    else:
      numericos[i] = df.columns[i]

  return (categoricos, numericos)




