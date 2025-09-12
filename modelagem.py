from sklearn.model_selection import train_test_split
from conveniencia.eng_dados import pega_tipos
from conveniencia import graficos
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor





# Scikit-learn
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

# XGBoost e LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.metrics import median_absolute_error, explained_variance_score








def treinar_bin(dados, target,  modelos={}, funcs_perda={}, test_size=0.25, plot=True):
  """
  x: dataframe com as variáveis independentes do modelo

  y: dataframe com a variável resposta do modelo
  modelos: dicionário contendo o nome dos modelos a serem usados 
  (o nome é o que vai aparecer no dataframe no final) e os objetos dos modelos

  funcs_perda: dicionário contendo o nome das funções de perda a serem usados 
  (o nome é o que vai aparecer no dataframe no final) e as funções de perda
  test_size: valor real que contém a proporção de dados que será usada no teste dos dados

  plot: booleano indicando se a função deve plotar gráficos de barras com os desempenhos de cada modelo

  Returns:
  Retorna um dataframe onde cada linha corresponde a um modelo treinado e cada coluna
  corresponde a uma função de perda
  """
  y = dados[[target]]
  x = dados.drop(columns=[target])

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
  

  if (modelos == {}):
    modelos = {
    "LogisticRegression": LogisticRegression(),
    "Perceptron": Perceptron(),
    "RidgeClassifier": RidgeClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "SVC": SVC(probability=True),  # probabilidade ativada para log_loss/AUC
    "LinearSVC": LinearSVC(),
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB(),
    "KNeighbors": KNeighborsClassifier(),
    "MLPClassifier": MLPClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier()
    }

  if (funcs_perda == {}):
    funcs_perda = {
    "LogLoss": log_loss,  # cross-entropy
    "HingeLoss": hinge_loss,  # usado em SVMs
    "ZeroOneLoss": zero_one_loss,  # taxa de erro
    "BrierScore": brier_score_loss,  # calibração de probabilidades
    "Accuracy": accuracy_score,  # acurácia
    "Precision": precision_score,  # precisão (positivos corretos / preditos)
    "Recall": recall_score,  # sensibilidade
    "F1Score": f1_score,  # média harmônica entre precisão e recall
    "ROC_AUC": roc_auc_score,  # área sob curva ROC
    "MatthewsCorrCoef": matthews_corrcoef,  # MCC
    "CohensKappa": cohen_kappa_score  # concordância entre classes
}


  perda = {}


  # Treinando os modelos
  for i in modelos.keys():
    modelos[i].fit(x_train, y_train)
    y_pred = modelos[i].predict(x_test)

    # Calculando as perdas para cada modelo
    lista = []
    for j in funcs_perda.keys():
      lista.append(funcs_perda[j](y_test, y_pred))

    perda[i] = lista
  metricas = pd.DataFrame(perda)
  metricas.index = list(funcs_perda.keys())

  metricas = metricas.transpose()
  if plot:
    metricas_semindex = metricas.reset_index()
    quant_colunas = metricas_semindex.shape[1]
    colunas = list(funcs_perda.keys())
    graficos.barras_y(metricas_semindex, colunas, "index" (1,quant_colunas-1))

  return metricas



def treinar_class(dados, target,  modelos={}, funcs_perda={}, test_size=0.25, plot=True):
  """
  x: dataframe com as variáveis independentes do modelo

  y: dataframe com a variável resposta do modelo
  modelos: dicionário contendo o nome dos modelos a serem usados 
  (o nome é o que vai aparecer no dataframe no final) e os objetos dos modelos

  funcs_perda: dicionário contendo o nome das funções de perda a serem usados 
  (o nome é o que vai aparecer no dataframe no final) e as funções de perda
  test_size: valor real que contém a proporção de dados que será usada no teste dos dados

  plot: booleano indicando se a função deve plotar gráficos de barras com os desempenhos de cada modelo

  Returns:
  Retorna um dataframe onde cada linha corresponde a um modelo treinado e cada coluna
  corresponde a uma função de perda
  """
  y = dados[[target]]
  x = dados.drop(columns=[target])

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
  

  if (modelos == {}):
    modelos = {
    "LogisticRegression": LogisticRegression(multi_class="multinomial", solver="lbfgs"),
    "Perceptron": Perceptron(),
    "RidgeClassifier": RidgeClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "SVC": SVC(probability=True, decision_function_shape="ovo"),  # OvO para multi-classe
    "LinearSVC": LinearSVC(),
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "KNeighbors": KNeighborsClassifier(),
    "MLPClassifier": MLPClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    "LightGBM": LGBMClassifier()
}

  if (funcs_perda == {}):
    funcs_perda = {
    "LogLoss": log_loss,  # cross-entropy multiclasse
    "ZeroOneLoss": zero_one_loss,  # taxa de erro
    "Accuracy": accuracy_score,
    "F1Score_Macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    "F1Score_Weighted": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
    "Precision_Macro": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro"),
    "Precision_Weighted": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
    "Recall_Macro": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro"),
    "Recall_Weighted": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
    "MatthewsCorrCoef": matthews_corrcoef,  # MCC para multi-classe
    "CohensKappa": cohen_kappa_score  # Kappa para multi-classe
}


  perda = {}


  # Treinando os modelos
  for i in modelos.keys():
    modelos[i].fit(x_train, y_train)
    y_pred = modelos[i].predict(x_test)

    # Calculando as perdas para cada modelo
    lista = []
    for j in funcs_perda.keys():
      lista.append(funcs_perda[j](y_test, y_pred))

    perda[i] = lista
  metricas = pd.DataFrame(perda)
  metricas.index = list(funcs_perda.keys())

  metricas = metricas.transpose()
  if plot:
    metricas_semindex = metricas.reset_index()
    quant_colunas = metricas_semindex.shape[1]
    colunas = list(funcs_perda.keys())
    graficos.barras_y(metricas_semindex, colunas, "index" (1,quant_colunas-1))

  return metricas





def treinar_reg(dados, target,  modelos={}, funcs_perda={}, test_size=0.25, plot=True):
  """
  x: dataframe com as variáveis independentes do modelo

  y: dataframe com a variável resposta do modelo
  modelos: dicionário contendo o nome dos modelos a serem usados 
  (o nome é o que vai aparecer no dataframe no final) e os objetos dos modelos

  funcs_perda: dicionário contendo o nome das funções de perda a serem usados 
  (o nome é o que vai aparecer no dataframe no final) e as funções de perda
  test_size: valor real que contém a proporção de dados que será usada no teste dos dados

  plot: booleano indicando se a função deve plotar gráficos de barras com os desempenhos de cada modelo

  Returns:
  Retorna um dataframe onde cada linha corresponde a um modelo treinado e cada coluna
  corresponde a uma função de perda
  """
  y = dados[[target]]
  x = dados.drop(columns=[target])

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
  

  if (modelos == {}):
    linear_reg = LinearRegression()
    ridge_reg = Ridge()
    lasso_reg = Lasso()
    elasticnet_reg = ElasticNet()
    bayesian_reg = BayesianRidge()
    logistic_reg = LogisticRegression()

    decision_tree_reg = DecisionTreeRegressor()
    random_forest_reg = RandomForestRegressor()
    gradient_boosting_reg = GradientBoostingRegressor()
    xgb_reg = XGBRegressor()
    lgbm_reg = LGBMRegressor()
    svr_reg = SVR()

   
    # Agrupando todos em um dicionário para fácil acesso
    modelos = {
        "Linear": linear_reg,
        "Ridge": ridge_reg,
        "Lasso": lasso_reg,
        "ElasticNet": elasticnet_reg,
        "BayesianRidge": bayesian_reg,
        "Logistic": logistic_reg,
        "DecisionTree": decision_tree_reg,
        "RandomForest": random_forest_reg,
        "GradientBoosting": gradient_boosting_reg,
        "XGBoost": xgb_reg,
        "LightGBM": lgbm_reg,
        "SVR": svr_reg
    }

  if (funcs_perda == {}):
    funcs_perda = {
    "MSE": mean_squared_error,  # erro quadrático médio
    "RMSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),  # raiz do MSE
    "MAE": mean_absolute_error,  # erro absoluto médio
    "MedAE": median_absolute_error,  # erro absoluto mediano
    "MSLE": mean_squared_log_error,  # erro quadrático médio logarítmico
    "R2": r2_score,  # coeficiente de determinação
    "ExplainedVariance": explained_variance_score  # variância explicada pelo modelo
}


  perda = {}


  # Treinando os modelos
  for i in modelos.keys():
    modelos[i].fit(x_train, y_train)
    y_pred = modelos[i].predict(x_test)

    # Calculando as perdas para cada modelo
    lista = []
    for j in funcs_perda.keys():
      lista.append(funcs_perda[j](y_test, y_pred))

    perda[i] = lista
  metricas = pd.DataFrame(perda)
  metricas.index = list(funcs_perda.keys())

  metricas = metricas.transpose()
  if plot:
    metricas_semindex = metricas.reset_index()
    quant_colunas = metricas_semindex.shape[1]
    colunas = list(funcs_perda.keys())
    graficos.barras_y(metricas_semindex, colunas, "index" (1,quant_colunas-1))

  return metricas







def pcfacil(dados, n_comp, colunas=[], numerica=True):
  """
  reliza uma PCA e retorna um dataframe com as componentes principais e
  retorna um df com os componentes principais e as variáveis não incluidas

  df: dataframe em que será realizado o PCA
  n_comp: inteiro, quantidade de componentes
  colunas: lista númerica representando as coordenadas das variáveis ou strings com os nomes delas
  (caso nada seja imputado, usa a função pega_tipos)
  numerica: Booleano indicando se a lista de variáveis é numerica ou strings
  
  test: dataframe, dataframe de teste para caso o argumento df seja o dataframe de treino

  Returns:
        pandas.DataFrame: O dataframe original mas com as variáveis que foram usadas no PCA 
        retiradas e os componentes principais incluidos

        PCA: O objeto usado para o PCA

  """

  # checando se foram passadas variáveis para o PCA ou se devo usar a função pega_tipos
  # para escolher quais variáveis vão ser usadas no PCA

  df = dados.copy()

  if len(colunas)==0:
    cat, numericas = pega_tipos(df)
    lista_cat = list(cat.keys())
    lista_numericas = list(numericas.keys())

  else:

    if numerica:
      lista_numericas = df.columns[colunas]
    else:
      lista_numericas = colunas

    lista_cat = []

    for j in range(df.shape[1]):
      if df.columns[j] not in lista_numericas:
        lista_cat.append(df.columns[j])

  # Fazeno o PCA
  pca = PCA(n_components=n_comp)
  df_pca = pca.fit_transform(df.loc[:, lista_numericas])
  lista_nomes = []

  #Atribuindo nomes os componentes
  for i in range(1, n_comp+1):
    lista_nomes.append(f"componente_{i}")

  # Criando data frame com os componentes e as demais variáveis
  df_pca = pd.DataFrame(df_pca, columns=lista_nomes)
  df_pca = pd.concat([df.loc[:, lista_cat], df_pca], axis=1)

  plt.plot(pca.explained_variance_ratio_)

  return(df_pca, pca)





