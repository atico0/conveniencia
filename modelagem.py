
#modelos
def treinar_todos(x, y, modelos, funcs_perda):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
  perda = {}

  for i in modelos.keys():
    modelos[i].fit(x_train, y_train)
    y_pred = modelos[i].predict(x_test)

    lista = []
    for j in funcs_perda.keys():
      lista.append(funcs_perda[j](y_test, y_pred))

    perda[i] = lista
  metricas = pd.DataFrame(perda)
  metricas.index = list(funcs_perda.keys())
  return(metricas.transpose())