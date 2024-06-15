import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#dados ordinais
def label(dados, colunas, numerica=True):
  """
  realiza o label encoder num conjunto de variáveis num dataframe

  df: dataframe
  colunas: lista númerica representando as coordenadas das variáveis ou strings com os nomes delas
  numerica: Booleano indicando se a lista de variáveis é numerica ou strings
  """
  df = dados.copy()

  if numerica:
    cols = df.columns[colunas]
  else:
    cols = colunas

  transformadores = {}
  for i in cols:
    transformador = LabelEncoder() 
    df.loc[:,i] = transformador.fit_transform(df.loc[:,i])
    transformadores[i] = transformador

  return (df, transformadores)

def desfaz_label(df, transformadores):
  

  for i in transformadores.keys():
    df.loc[:, i] = transformadores[i].inverse_transform(df.loc[:, i].astype(int))
  return df



#dados nominais

def onehote(dados, colunas, numerica=True, prefix_sep="_", drop_first=True):
  """
  realiza o one hot encoder num conjunto de variáveis num dataframe

  dados: dataframe
  colunas: lista númerica representando as coordenadas das variáveis ou strings com os nomes delas
  numerica: Booleano indicando se a lista de variáveis é numerica ou strings
  prefix_sep: string usada no get_dummies
  drop_first: Booleano para dizer se alguma coluna vai ser apagada no processo

  Return:
  um dataframe com as variáveis originais mais as variáveis novas do encoding mas sem as variáveis
  usadas no processo  de encoding. Um dicionário contendo os nomes das variáveis apagas no processo
  como chaves e a categoria cuja coluna foi apagada como valor (usar na função desfaz_onehote)


  """

  df = dados.copy()
  
  if numerica:
    cols = df.columns[colunas]
  else:
    cols = colunas

  valores_dropados = {}
  for i in cols:
    novo = pd.get_dummies(df.loc[:, i], prefix=i,
      prefix_sep=prefix_sep).astype(int)

    for coluna in novo.columns:
      elemento_coluna = coluna.replace(i+prefix_sep,"")

    if drop_first:
      valores_dropados[i] = novo.columns[0].replace(i+prefix_sep,"")
      novo = novo.iloc[:, 1:]
    else:
      valores_dropados[i] = ""
    df = pd.concat([df, novo], axis=1)
    df.drop([i], axis=1, inplace=True)
  return (df, valores_dropados)



def desfaz_1onehote(dados, var_org, valor_dropado="", prefix_sep="_"):

  df = dados.copy()
  #checando quais variáveis do dataframe original vão ser usadas
  lista_variaveis = []
  lista_valores = []
  for j in df.columns:
    if var_org+prefix_sep in j:
      lista_variaveis.append(j)
  
  #percorrendo as linhas do dataframe e checando em qual coluna está o valor 1
  for n in df.index:
    local = np.where(df.loc[n, lista_variaveis] == 1)[0]
    if local.size > 0:
      lista_valores.append((lista_variaveis[local[0]]).replace(var_org+prefix_sep,""))
    else:
      #adicionando o valor dropado caso nenhua das colunas possua o valor 1
      lista_valores.append(valor_dropado)
        
  df[var_org] = lista_valores
  df.drop(lista_variaveis, axis=1, inplace=True)
  print(lista_variaveis)
  return df


def desfaz_onehote(dados, valores_dropados, prefix_sep="_"):
  
  df = dados.copy()
  for i in valores_dropados.keys():
    df = desfaz_1onehote(df, i, valores_dropados[i], prefix_sep)
  return df




#dados quantitativos
from sklearn.preprocessing import StandardScaler
def padronizar(df, colunas, numerica=True):
  """
  realiza a padrinização num conjunto de variáveis num dataframe

  df: dataframe
  colunas: lista númerica representando as coordenadas das variáveis ou strings com os nomes delas
  numerica: Booleano indicando se a lista de variáveis é numerica ou strings
  """
  df = dados.copy()
  
  if numerica:
    cols = df.columns[colunas]
  else:
    cols = colunas

  for i in cols:
    df.loc[:,i] = StandardScaler().fit_transform(df.loc[:,i].values.reshape(-1,1))
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



