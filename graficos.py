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

