#gráficos

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def histogramas(df, colunas, numerica=True, forma=(1,1), figsize= (20,10)):
  """
  Plota vários histogramas de uma vez

  df: Dataframe contendo as variáveis a serem plotadas

  colunas: lista númerica representando as coordenadas das variáveis ou strings com os nomes delas
  numerica: Booleano indicando se o paramêtro colunas é numerico ou strings

  forma: tupla numérica representando a quantidade de linhas e colunas dos histogramas

  figsize: tupla numérica representando o tamanho de cada histograma

  returns:
  Nda
  """
  if numerica:
    cols = df.columns[colunas]
  else:
    cols = colunas


  fig, ax = plt.subplots(forma[0], forma[1], figsize=figsize)

  for col, ax in zip(cols, ax.flatten()):
    sns.histplot(df[col], ax=ax, kde=True)

  plt.tight_layout()
  plt.show()


def boxplots(df, colunas, numerica=True, forma=(1,1), figsize= (20, 10)):
  """
  Plota vários boxplots de uma vez

  df: Dataframe contendo as variáveis a serem plotadas

  colunas: lista númerica representando as coordenadas das variáveis ou strings com os nomes delas
  numerica: Booleano indicando se o paramêtro colunas é numerico ou strings

  forma: tupla numérica representando a quantidade de linhas e colunas dos plots

  figsize: tupla numérica representando o tamanho de cada plot

  returns:
  Nda
  
  """
  if numerica:
    cols = df.columns[colunas]
  else:
    cols = colunas

  fig, ax = plt.subplots(forma[0], forma[1], figsize=figsize)

  for col, ax in zip(cols, ax.flatten()):
    sns.boxplot(x=df[col], ax=ax, orient="h")

  plt.tight_layout()
  plt.show()

import ptitprince as pt

def rainclouds(df, colunas, numerica=True, forma=(1,1), figsize= (20, 10), x=None, orient="h"):
  
  """
  Plota vários rainclouds de uma vez]

  df: Dataframe contendo as variáveis a serem plotadas

  colunas: lista númerica representando as coordenadas das variáveis ou strings com os nomes delas
  numerica: Booleano indicando se o paramêtro colunas é numerico ou strings

  forma: tupla numérica representando a quantidade de linhas e colunas dos plots

  figsize: tupla numérica representando o tamanho de cada plot

  x: string representando a variável categorica a ser usada para dividir os rainclouds (basicamente um hue)

  returns:
  Nda
  """
  if numerica:
    cols = df.columns[colunas]
  else:
    cols = colunas

  fig, ax = plt.subplots(forma[0], forma[1], figsize=figsize)

  for col, ax in zip(cols, ax.flatten()):
    pt.RainCloud(data=df, y=col, x=x, orient=orient, ax=ax)
  plt.tight_layout()
  plt.show()


def barras_x(df, colunas, y, forma, numerica=True, figsize= (20, 10), hue=None):
  """
  Plota várias barras de uma vez alterando o eixo X
  
  df: Dataframe contendo as variáveis a serem plotadas

  colunas: lista númerica representando as coordenadas das variáveis (eixos dos x)
  ou strings com os nomes delas
  numerica: Booleano indicando se o paramêtro colunas é numerico ou strings

  y: string representando a variável do eixo y das barras

  forma: tupla numérica representando a quantidade de linhas e colunas dos plots

  figsize: tupla numérica representando o tamanho de cada plot

  returns:
  Nda
  """

  if numerica:
    cols = df.columns[colunas]
  else:
    cols = colunas

  fig, ax = plt.subplots(forma[0], forma[1], figsize=figsize)

  for col, ax in zip(cols, ax.flatten()):
    sns.barplot(x=col, y=y, hue=hue, data=df, palette="Paired")
  plt.tight_layout()
  plt.show()


def barras_y(df, colunas, x, forma, numerica=True, figsize= (20, 10), hue=None):
  """
  Plota várias barras de uma vez alterando o eixo Y
  
  df: Dataframe contendo as variáveis a serem plotadas

  colunas: lista númerica representando as coordenadas das variáveis (eixos dos y)
  ou strings com os nomes delas
  numerica: Booleano indicando se o paramêtro colunas é numerico ou strings

  x: string representando a variável do eixo x das barras

  forma: tupla numérica representando a quantidade de linhas e colunas dos plots

  figsize: tupla numérica representando o tamanho de cada plot

  returns:
  Nda
  """
  if numerica:
    cols = df.columns[colunas]
  else:
    cols = colunas

  fig, ax = plt.subplots(forma[0], forma[1], figsize=figsize)

  for col, ax in zip(cols, ax.flatten()):
    sns.barplot(x=x, y=col, hue=hue, data=df, palette="Paired")
  plt.tight_layout()
  plt.show()



def dispersoes(df, colunas, y, numerica=True, hue=None, style=None):
  """
  Gera vários gráficos de dispersão

  df: Dataframe contendo as variáveis

  colunas: lista númerica representando as coordenadas das variáveis (eixos dos x)
  ou strings com os nomes delas
  numerica: Booleano indicando se o paramêtro colunas é numerico ou strings

  valor: string contendo a coluna no eixo y
  
  """

  if numerica:
    cols = df.columns[colunas]
  else:
    cols = colunas

  for col in cols:
    sns.relplot(data=df, x=col, y=y,
                    hue=hue, style=style)




