#gráficos

import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yfinance as yf




def retorno_simples(serie, inicio, final, quant=1, dividendo=0, taxa_compra=0, taxa_venda=0):
  rs = ((serie[final] + dividendo - serie[inicio]) * quant - taxa_compra - taxa_venda)/serie[inicio]
  return(rs*100)


def lucro(serie, inicio, final, quant=1, dividendo=0, taxa_compra=0, taxa_venda=0):
  l = (serie[final] + dividendo - serie[inicio]) * quant - taxa_compra - taxa_venda
  return l

def baixa_acoes(acoes, variavel, inicio, fim=None):
  acoes_df = pd.DataFrame()
  for acao in acoes:
    acoes_df[acao] = yf.download(acao, start=inicio, end=fim).loc[:, variavel]
  return acoes_df


def normalizador(df, variaveis):
  acoes_df_normalizado = df.copy()
  for i in acoes_df_normalizado.columns[1:]:
    acoes_df_normalizado[i] = acoes_df_normalizado[i] / acoes_df_normalizado[i][0]
  return acoes_df_normalizado

def historico(df, variaveis, x=None, normalizar=False, figsize = (15,7),
  title = 'Histórico do preço das ações'):
  if normalizar:
    acoes_df_normalizado = normalizador(df=df, variaveis=variaveis)
    acoes_df_normalizado.loc[:, variaveis].plot(x = x, figsize = figsize,
      title = 'Histórico do preço das ações')
  else:
    df.loc[:, variaveis].plot(x = x, figsize = figsize, title = 'Histórico do preço das ações')


