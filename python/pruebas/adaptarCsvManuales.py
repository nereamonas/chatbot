#https://tacosdedatos.com/texto-vectores

#https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments


import pandas as pd
from stop_words import get_stop_words
import re
import spacy
import nltk
from autocorrect import Speller
from unicodedata import normalize

"""
#1- VIDEOCONFERENCIAS
#1- convertirlo en csv
df = pd.read_csv("../archivos/txt/6-Videoconferencia_ES.txt", delimiter = '\n')
df.to_csv('../archivos/csvIntermedios/Videoconferenciaa_ES.csv')

#Lo editamos. borramos la priera columna y agregamos una nueva columna llamada class
df2 = pd.read_csv('../archivos/csvIntermedios/Videoconferencia_ES.csv',names=["Number", "Text", "Class"])
df2.drop('number', axis = 1, inplace = True)
df2['class']='videoconferencia'
print(df2)
df2.to_csv('../archivos/csvIntermedios/Videoconferencia_ES.csv', index = False)

#2- CUESTIONARIOS
#1- convertirlo en csv
df = pd.read_csv("../archivos/txt/8-Cuestionarios2_ES.txt", delimiter = '\n')
df.to_csv('../archivos/csvIntermedios/Cuestionarios_ES.csv')


#en las primeras lineas hay q hacer unas pequeñas modificaciones. xq no lo coge bien. eso hay q tratarlo aparte
df2 = pd.read_csv('../archivos/csvIntermedios/Cuestionarios_ES.csv',names=["Number","Text", "Class"])
df2.drop('number', axis = 1, inplace = True)
df2['class']='cuestionarios'
print(df2)
df2.to_csv('../archivos/csvIntermedios/Cuestionarios_ES.csv', index = False)

# 3- CONCATENAR LOS DOS
#unir todos en un solo csv para meter al clasificador: Sin poner la primera linea
df1 = pd.read_csv('../archivos/csvIntermedios/Videoconferencia_ES.csv')
df2 = pd.read_csv('../archivos/csvIntermedios/Cuestionarios_ES.csv')

out = df1.append(df2)
out.to_csv('../archivos/csvIntermedios/concatenate.csv', index = False)

#-------------------------------------------------

"""

stop_words = get_stop_words('spanish')
stop_words.remove("no")
stop_words.remove("sin")

spanish_stemmer = nltk.stem.SnowballStemmer('spanish')

nlp = spacy.load('es_core_news_sm')

def capar(text):
    text=" "+text+" "
    # 1- Poner to do en minusculas
    text = text.lower()
    # 2- quitar las url
    text = re.sub("https?://[^\s]+", "", text)  # quitar las url.
    # 3- quitar simbolos
    symbols = "'!\"#$%&()*+-./:;<=,>?@[\]^_`{|}~“”¿¡ºª·"  # \n /// no puedo quitarle los " jaajaj text = text.replace('"', "")
    for i in symbols:
        text = text.replace(i, "")
    text=text.replace('"', '')
    # 4- quitar numeros
    text = re.sub("\d+", "", text)
    # 5- quitar get_stop_words('spanish')
    for word in stop_words:
        text = text.replace(" " + word + " ", " ")  # a, al, algo, algunas...
    # 6- Stemming en español- SnowballStemmer('spanish')
    text_words = text.split(" ")
    for palabra in text_words:
        text = text.replace(palabra, spanish_stemmer.stem(palabra))
    # 7- corrige un poco las palabras. pero ns si hace falta ahora...
    doc= nlp(text)
    for token in doc:
        if (token.text != token.lemma_):
            text.replace(token.text,token.lemma_)
    # 9- eliminar tildes y simbolos raros. pero sin eliminar la ñ
    # -> NFD y eliminar diacríticos
    text = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1",normalize("NFD", text), 0, re.I)
    # -> NFC
    text = normalize('NFC', text)

    return text



#7.Estandarizar palabras
# frases como "estoy muuuuuy feliz" las convertimos en "estoy muy feliz"
# texto = ''.join(''.join(s)[:2] for _, s in itertools.groupby(texto))


"""
df3 = pd.read_csv('../archivos/csvIntermedios/concatenate.csv')
clase=df3["Text"]
count=0
for x in clase:
    clase[count]=capar(x)
    count+=1

df3["Text"]=clase
df3.to_csv('../archivos/csv/concatenateCapado.csv', index = False)

"""
df4 = pd.read_csv('../../archivos/csvIntermedios/VideoconferenciaXTemas_ES.csv')
clase=df4["Text"]
count=0
for x in clase:
    clase[count]=capar(x)
    count+=1
df4["Text"]=clase
df4.to_csv('../archivos/csv/VideoconferenciaXTemas_ESCapado.csv', index = False)


"""
df5 = pd.read_csv('../archivos/csvIntermedios/CuestionariosXTemas_ES.csv')
clase=df5["Text"]
count=0
for x in clase:
    clase[count]=capar(x)
    count+=1
df5["Text"]=clase
df5.to_csv('../archivos/csv/CuestionariosXTemas_ESCapado.csv', index = False)
"""
#http://datascience.esy.es/limpieza-de-texto-utilizando-python/
#https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
#https://es.stackoverflow.com/questions/135707/c%C3%B3mo-puedo-reemplazar-las-letras-con-tildes-por-las-mismas-sin-tilde-pero-no-l
#https://necronet.github.io/Spacy-getting-started-in-spanish/
#http://datascience.esy.es/limpieza-de-texto-utilizando-python/

