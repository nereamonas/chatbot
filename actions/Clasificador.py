import os
import re

import nltk
import pandas as pd
from autocorrect import Speller
from unicodedata import normalize
import re
import spacy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from stop_words import get_stop_words
from googletrans import Translator
import googletrans
print(os.getcwd())
pathAbs=os.getcwd()


def capar(text):
    # 1- Poner to do en minusculas
    text = text.lower()
    # 2- quitar las url
    text = re.sub("https?://[^\s]+", "", text)  # quitar las url.
    # 3- quitar get_stop_words('spanish')
    stop_words = get_stop_words('spanish')
    for word in stop_words:
        if (word!='no'):
            text = text.replace(" " + word + " ", " ")  # a, al, algo, algunas...
    # 4- quitar simbolos
    symbols = "'!\"#$%&()*+-./:;<=,>?@[\]^_`{|}~“”"  # \n /// no puedo quitarle los " jaajaj text = text.replace('"', "")
    for i in symbols:
        text = text.replace(i, "")
    text=text.replace('"', '')
    # 5- quitar numeros
    text = re.sub("\d+", "", text)
    # 7- Stemming en español- SnowballStemmer('spanish')
    spanish_stemmer = nltk.stem.SnowballStemmer('spanish')
    text_words = text.split(" ")
    for palabra in text_words:
        text = text.replace(palabra, spanish_stemmer.stem(palabra))
    # 8- autocorreccion de palabras
    spell = Speller('es')
    text=spell(text)
    # 9- corrige un poco las palabras. pero ns si hace falta ahora...
    nlp = spacy.load('es_core_news_sm')
    doc= nlp(text)
    for token in doc:
        if (token.text != token.lemma_):
            #print(token.text, "|", token.lemma_, '|', token.pos_)
            text.replace(token.text,token.lemma_)
    # 10- eliminar tildes y simbolos raros. pero sin eliminar la ñ
    # -> NFD y eliminar diacríticos
    text = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1",normalize("NFD", text), 0, re.I)
    # -> NFC
    text = normalize('NFC', text)
    return text


#1- ENTRENAMIENTO CLASES
#2- Cargamos el dataset
df=pd.read_csv(pathAbs+'/archivos/csv/concatenateCapado.csv',sep=',')  # no se porq solo va con ruta absoluta
#3- imprimimos las longitudes
print("len total: ",len(df),"\nLen videoconferencia: ",len(df[df.Class=='videoconferencia']),"\nLen cuestionarios: ",len(df[df.Class=='cuestionarios']))
#3- TfIdf Vectorized. Y separamos para train y test
cvClase = TfidfVectorizer(min_df=1,stop_words='english')
x_trainClase=df["Text"].values.astype('U')
y_trainClase=df["Class"]
#4- fit transform
x_traincvClase=cvClase.fit_transform(x_trainClase)
# 6- Clasificador
svcClase=LinearSVC(random_state=0, tol=1e-5)
mnbClase = CalibratedClassifierCV(svcClase)
mnbClase.fit(x_traincvClase, y_trainClase)  # lo entreno con el train

#2- ENTRENAMIENTO videoconferencia
df = pd.read_csv(pathAbs+'/archivos/csv/VideoconferenciaXTemas_ESCapado.csv', sep=',')  # no se porq solo va con ruta absoluta
# 3- TfIdf Vectorized. Y separamos para train y test
cvVideoconferencia = TfidfVectorizer(min_df=1, stop_words='english')
x_trainVideoconferencia = df["Text"].values.astype('U')
y_trainVideoconferencia = df["Class"]
# 4- fit transform
x_traincvVideoconferencia = cvVideoconferencia.fit_transform(x_trainVideoconferencia)
# 6- Clasificador
svcVideoconferencia=LinearSVC(random_state=0, tol=1e-5)
mnbVideoconferencia = CalibratedClassifierCV(svcVideoconferencia)
mnbVideoconferencia.fit(x_traincvVideoconferencia, y_trainVideoconferencia)  # lo entreno con el train

#3- ENTRENAMIENTO CUESTIONARIOS
df = pd.read_csv(pathAbs+'/archivos/csv/CuestionariosXTemas_ESCapado.csv', sep=',')  # no se porq solo va con ruta absoluta
# 3- TfIdf Vectorized. Y separamos para train y test
cvCuestionarios = TfidfVectorizer(min_df=1, stop_words='english')
x_trainCuestionarios = df["Text"].values.astype('U')
y_trainCuestionarios = df["Class"]
# 4- fit transform
x_traincvCuestionarios = cvCuestionarios.fit_transform(x_trainCuestionarios)
# 6- Clasificador
svcCuestionarios=LinearSVC(random_state=0, tol=1e-5)
mnbCuestionarios = CalibratedClassifierCV(svcCuestionarios)
mnbCuestionarios.fit(x_traincvCuestionarios, y_trainCuestionarios)  # lo entreno con el train

def traducirCastellano(text):
    traductor = Translator()
    traduccion = traductor.translate(text, dest='es')
    print(f"{traduccion.origin} ({traduccion.src}) --> {traduccion.text} ({traduccion.dest})")
    return traduccion.text, traduccion.src

def clasificador(clase,text):
    if (clase == 'clase'):
        mnb=mnbClase
        cv=cvClase
    elif (clase == 'videoconferencia'):
        mnb=mnbVideoconferencia
        cv=cvVideoconferencia
    elif (clase == 'cuestionarios'):
        mnb=mnbCuestionarios
        cv=cvCuestionarios
    x_test = [capar(text)]
    x_testcv = cv.transform(x_test)
    # Sacamos la predicción
    predicion = mnb.predict(x_testcv)
    # nos dará el threshold de cada clase
    threshold= mnb.predict_proba(x_testcv)
    return predicion[0], max(threshold[0])

def conseguirPaginaCorrespondiente(clase, tema):
    # Cargamos el csv que nos dice que pagina del manual corresponde con el tema
    df = pd.read_csv(pathAbs+'/archivos/csv/paginaCorrespondiente.csv', sep=',')
    clasee = df.loc[:, 'Class'] == clase
    df_1 = df.loc[clasee]
    subclase = df_1.loc[:, 'SubClass'] == int(tema)
    df_2 = df_1.loc[subclase]
    url = df_2.values[0][2]
    return url

def clasificarPregunta(pregunta):
    print("Pregunta: ",pregunta)
    preguntaCastellano, idioma=traducirCastellano(pregunta)
    clasePredecida, threshold = clasificador('clase',preguntaCastellano)
    if threshold > 0.50:  # si el threshold es >50% clasificaremos como que será correcta la clase que predice
        # mirar el foro de la clase
        # si no se en cuentra mirar en el manual de la clase
        temaPredecido, thresholdTema = clasificador(clasePredecida, preguntaCastellano)

        # Habria que mirar el threshold. pero de momento que lo haga siempre
        # Conseguimos la url que le corresponde al tema y clase
        url = conseguirPaginaCorrespondiente(clasePredecida, temaPredecido)
    print("Clase predecida: ", clasePredecida, "  Threshold clase: ", threshold, "\nTema predecido: ", temaPredecido, "  Threshold tema: ",thresholdTema,"\n")
    return clasePredecida,temaPredecido,url,idioma
