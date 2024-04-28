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
  for i in range(dados.shape[1]):
    if str(dados.iloc[:,i].dtype) == "object":
      categoricos[i] = df.columns[i]
    else:
      numericos[i] = df.columns[i]

  return (categoricos, numericos)



