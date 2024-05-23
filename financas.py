#gráficos

import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yfinance as yf




def retorno_simples(serie, inicio, final, quant=1, dividendo=0, taxa_compra=0, taxa_venda=0):
  rs = ((serie[inicio] + dividendo - serie[final]) * quant - taxa_compra - taxa_venda)/serie[inicio]
  return(rs*100)


def lucro(serie, inicio, final, quant=1, dividendo=0, taxa_compra=0, taxa_venda=0):
  l = (serie[inicio] + dividendo - serie[final]) * quant - taxa_compra - taxa_venda
  return l

def baixa_acoes(acoes, variavel, inicio, fim=None):
  acoes_df = pd.DataFrame()
  for acao in acoes:
    acoes_df[acao] = yf.download(acao, start=inicio, end=fim).loc[:, variavel]
  return acoes_df


