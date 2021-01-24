import os
import re
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from stop_words import get_stop_words
print(os.getcwd())
pathAbs=os.getcwd()


def capar(text):
    # 1- Poner to do en minusculas
    text = text.lower()
    # 2- quitado las url
    text = re.sub("https?://[^\s]+", "", text)  # quitar las url.
    # 3- quitar get_stop_words('spanish')
    stop_words = get_stop_words('spanish')
    for word in stop_words:
        text = text.replace(" " + word + " ", " ")  # a, al, algo, algunas...
    # 4- quitar simbolos
    symbols = "'!\"#$%&()*+-./:;<=,>?@[\]^_`{|}~“”"  # \n /// no puedo quitarle los " jaajaj text = text.replace('"', "")
    for i in symbols:
        text = text.replace(i, "")
    text=text.replace('"', '')
    # 5- quitar numeros
    text = re.sub("\d+", "", text)
    return text

#1- ENTRENAMIENTO CLASES
#2- Cargamos el dataset
df=pd.read_csv(pathAbs+'/archivos/csv/concatenate.csv',sep=',')  # no se porq solo va con ruta absoluta
#3- imprimimos las longitudes
print("len total: ",len(df),"\nLen videoconferencia: ",len(df[df.Class=='videoconferencia']),"\nLen cuestionarios: ",len(df[df.Class=='cuestionarios']))
#3- TfIdf Vectorized. Y separamos para train y test
cvClase = TfidfVectorizer(min_df=1,stop_words='english')
x_trainClase=df["Text"]
y_trainClase=df["Class"]
count=0
for x in x_trainClase:
    x_trainClase[count]=capar(x)
    count+=1
#4- fit transform
x_traincvClase=cvClase.fit_transform(x_trainClase)
# 6- Clasificador
mnbClase = MultinomialNB(fit_prior=False)  # 0.930693
mnbClase.fit(x_traincvClase, y_trainClase)  # lo entreno con el train

#2- ENTRENAMIENTO videoconferencia
df = pd.read_csv(pathAbs+'/archivos/csv/VideoconferenciaXTemas_ES.csv', sep=',')  # no se porq solo va con ruta absoluta
# 3- TfIdf Vectorized. Y separamos para train y test
cvVideoconferencia = TfidfVectorizer(min_df=1, stop_words='english')
x_trainVideoconferencia = df["Text"]
y_trainVideoconferencia = df["Class"]
count = 0
for x in x_trainVideoconferencia:
    x_trainVideoconferencia[count] = capar(x)
    count += 1
# 4- fit transform
x_traincvVideoconferencia = cvVideoconferencia.fit_transform(x_trainVideoconferencia)
# 6- Clasificador
mnbVideoconferencia = GridSearchCV(SVC(), [{'kernel': ['rbf'], 'gamma': [1e-3], 'C': [1000]}])
mnbVideoconferencia.fit(x_traincvVideoconferencia, y_trainVideoconferencia)  # lo entreno con el train

#3- ENTRENAMIENTO CUESTIONARIOS
df = pd.read_csv(pathAbs+'/archivos/csv/CuestionariosXTemas_ES.csv', sep=',')  # no se porq solo va con ruta absoluta
# 3- TfIdf Vectorized. Y separamos para train y test
cvCuestionarios = TfidfVectorizer(min_df=1, stop_words='english')
x_trainCuestionarios = df["Text"]
y_trainCuestionarios = df["Class"]
count = 0
for x in x_trainCuestionarios:
    x_trainCuestionarios[count] = capar(x)
    count += 1
# 4- fit transform
x_traincvCuestionarios = cvCuestionarios.fit_transform(x_trainCuestionarios)
# 6- Clasificador
mnbCuestionarios = GridSearchCV(SVC(), [{'kernel': ['rbf'], 'gamma': [1e-3], 'C': [1000]}])
mnbCuestionarios.fit(x_traincvCuestionarios, y_trainCuestionarios)  # lo entreno con el train

def clasificador(text):
    x_test= [capar(text)]
    x_testcv = cvClase.transform(x_test)
    # Sacamos la predicción
    predicion = mnbClase.predict(x_testcv)
    # nos dará el threshold de cada clase
    threshold = mnbClase.predict_proba(x_testcv)
    return predicion, threshold


def buscarManual(clase,text):
    if (clase == 'videoconferencia'):
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
    #threshold = mnb.predict_proba(x_testcv)  #el de grid no tiene el predict proba q da el threshold. habria q buscar otro modo
    threshold=0
    return predicion, threshold

def conseguirPaginaCorrespondiente(clase, tema):
    # Cargamos el csv que nos dice que pagina del manual corresponde con el tema
    df = pd.read_csv(pathAbs+'/archivos/csv/paginaCorrespondiente.csv', sep=',')
    clasee = df.loc[:, 'Class'] == clase
    df_1 = df.loc[clasee]
    subclase = df_1.loc[:, 'SubClass'] == int(tema)
    df_2 = df_1.loc[subclase]
    pagina = df_2.values[0][2]
    print("PAGINA  ", pagina)
    #Una vez tenemos la pagina, tenemos que crear la url para la pagina correspondiente. añadiendo a la url #page=8 - se abre en la apg que queramos
    url='null'
    if(clase=='videoconferencia'):
        url = "https://www.ehu.eus/documents/1852718/14189177/Bilera+birtualak++Collaborate-rekin+Irakasleentzako+eskuliburua.pdf/a393e0d9-29ee-5e8c-9122-d1983a4939d5" + "#page=" + str(pagina)
    elif(clase=='cuestionarios'):
        url = "https://www.ehu.eus/documents/1852718/14449606/Cuestionarios-eGela.pdf/541e8924-4d67-bcc6-a208-eff3bd438b6e?t=1588762441000" + "#page=" + str(pagina)
    return url

def clasificarPregunta(pregunta):
    clasePredecida, threshold = clasificador(pregunta)

    print(clasePredecida[0])
    print(threshold[0])
    if max(threshold[0]) > 0.50:  # si el threshold es >50% clasificaremos como que será correcta la clase que predice
        clase=clasePredecida[0]
        # mirar el foro de videoconferencia
        # si no se en cuentra mirar en el manual de videoconferencia
        temaPredecido, thresholdTema = buscarManual(clase, pregunta)

        tema = str(temaPredecido[0])

        # Habria que mirar el threshold. pero de momento que lo haga siempre
        # Conseguimos la url que le corresponde al tema y clase
        url = conseguirPaginaCorrespondiente(clase, tema)

    return clase,tema,url
