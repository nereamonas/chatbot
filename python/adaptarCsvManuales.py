#https://tacosdedatos.com/texto-vectores

#https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments


import pandas as pd

#1- VIDEOCONFERENCIAS
#1- convertirlo en csv
df = pd.read_csv("txt/6-Videoconferencia_ES.txt", delimiter = '\n')
df.to_csv('csv/Videoconferenciaa_ES.csv')

#Lo editamos. borramos la priera columna y agregamos una nueva columna llamada class
df2 = pd.read_csv('csv/Videoconferencia_ES.csv',names=["Number", "Text", "Class"])
df2.drop('number', axis = 1, inplace = True)
df2['class']='videoconferencia'
print(df2)
df2.to_csv('csv/Videoconferencia_ES.csv', index = False)

#2- CUESTIONARIOS
#1- convertirlo en csv
df = pd.read_csv("txt/8-Cuestionarios2_ES.txt", delimiter = '\n')
df.to_csv('csv/Cuestionarios_ES.csv')


#en las primeras lineas hay q hacer unas pequeñas modificaciones. xq no lo coge bien. eso hay q tratarlo aparte
df2 = pd.read_csv('csv/Cuestionarios_ES.csv',names=["Number","Text", "Class"])
df2.drop('number', axis = 1, inplace = True)
df2['class']='cuestionarios'
print(df2)
df2.to_csv('csv/Cuestionarios_ES.csv', index = False)


#unir todos en un solo csv para meter al clasificador: Sin poner la primera linea
df1 = pd.read_csv('csv/Videoconferencia_ES.csv')
df2 = pd.read_csv('csv/Cuestionarios_ES.csv')

out = df1.append(df2)
out.to_csv('csv/concatenate.csv', index = False)

