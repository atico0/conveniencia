#gráficos

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def histogramas(df, colunas, forma, figsize= (20, 10)):
  num_cols = df.columns[colunas]
  fig, ax = plt.subplots(forma[0], forma[1], figsize=figsize)

  for col, ax in zip(num_cols, ax.flatten()):
    sns.histplot(df[col], ax=ax, kde=True)

  plt.tight_layout()
  plt.show()


def boxplots(df, colunas, forma, figsize= (20, 10)):
  cols = df.columns[colunas]
  fig, ax = plt.subplots(forma[0], forma[1], figsize=figsize)

  for col, ax in zip(cols, ax.flatten()):
    sns.boxplot(x=df[col], ax=ax, orient="h")

  plt.tight_layout()
  plt.show()

import ptitprince as pt

def rainclouds(df, colunas, forma, figsize= (20, 10)):
  cols = df.columns[colunas]
  fig, ax = plt.subplots(forma[0], forma[1], figsize=figsize)

  for col, ax in zip(cols, ax.flatten()):
    pt.RainCloud(data=df, y=col, orient="h", ax=ax)
  plt.tight_layout()
  plt.show()


def barras(df, colunas, valor, forma, figsize= (20, 10), hue=None):
  cols = df.columns[colunas]
  fig, ax = plt.subplots(forma[0], forma[1], figsize=figsize)

  for col, ax in zip(cols, ax.flatten()):
    sns.barplot(x=col, y=valor, hue=hue, data=df, palette="Paired")
  plt.tight_layout()
  plt.show()


def dispersoes(df, colunas, valor,
               hue=None, style=None):
  """
  Gera vários gráficos de dispersão

  df: Dataframe contendo as variáveis
  colunas: lista de números com as colunas usadas no eixo x
  valor: string contendo a coluna no eixo y
  
  """

  cols = df.columns[colunas]

  for col in cols:
    sns.relplot(data=df, x=col, y=valor,
                    hue=hue, style=style)