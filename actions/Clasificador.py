import os
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
import pickle
from actions.Capar import capar
from actions.Traductor import traducirCastellano
from datetime import datetime


print(os.getcwd())
pathAbs=os.getcwd()



mnbClase=''
cvClase=''
mnbVideoconferencia=''
cvVideoconferencia=''
mnbCuestionarios=''
cvCuestionarios=''


def inicializarModelo(nuevo):  #Si nuevo=true, queremos crear un modelo, si nuevo=false cargamos un modelo
    if(nuevo):
        crearModelos()
    else:
        cargarModelos()

def cargarModelos():
    global mnbClase
    global cvClase
    global mnbVideoconferencia
    global cvVideoconferencia
    global mnbCuestionarios
    global cvCuestionarios

    print("Cargamos los modelos")

    mnbClase, xtrainClase = pickle.load(open(pathAbs + '/modelosClasificadorManuales/modelo_clase_2021-04-11_11:41:24.pkl', 'rb'))
    cvClase = TfidfVectorizer(min_df=1, stop_words='english')
    cvClase.fit_transform(xtrainClase)

    mnbVideoconferencia, xtrainVideoconferencia = pickle.load(open(pathAbs + '/modelosClasificadorManuales/modelo_videoconferencia_2021-04-11_11:41:24.pkl', 'rb'))
    cvVideoconferencia = TfidfVectorizer(min_df=1, stop_words='english')
    cvVideoconferencia.fit_transform(xtrainVideoconferencia)

    mnbCuestionarios, xtrainCuestionarios = pickle.load(open(pathAbs + '/modelosClasificadorManuales/modelo_cuestionarios_2021-04-11_11:41:24.pkl', 'rb'))
    cvCuestionarios = TfidfVectorizer(min_df=1, stop_words='english')
    cvCuestionarios.fit_transform(xtrainCuestionarios)


def crearModelos():
    formatoFecha = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    global mnbClase
    global cvClase
    global mnbVideoconferencia
    global cvVideoconferencia
    global mnbCuestionarios
    global cvCuestionarios

    print("Creamos los modelos")

    # 1- ENTRENAMIENTO CLASES
    # 2- Cargamos el dataset
    df = pd.read_csv(pathAbs + '/archivos/csv/concatenateCapado.csv', sep=',')  # no se porq solo va con ruta absoluta
    # 3- imprimimos las longitudes
    print("Len total: ", len(df))
    # 3- TfIdf Vectorized. Y separamos para train y test
    cvClase = TfidfVectorizer(min_df=1, stop_words='english')
    x_trainClase = df["Text"].values.astype('U')
    y_trainClase = df["Class"]
    # 4- fit transform
    x_traincvClase = cvClase.fit_transform(x_trainClase)
    # 6- Clasificador
    svcClase = LinearSVC(random_state=0, tol=1e-5)
    mnbClase = CalibratedClassifierCV(svcClase)
    mnbClase.fit(x_traincvClase, y_trainClase)  # lo entreno con el train
    filename = pathAbs + '/modelosClasificadorManuales/modelo_clase_'+formatoFecha+'.pkl'
    objectsClase = (mnbClase, x_trainClase)
    pickle.dump(objectsClase, open(filename, 'wb'))

    # 2- ENTRENAMIENTO videoconferencia
    df = pd.read_csv(pathAbs + '/archivos/csv/VideoconferenciaXTemas_ESCapado.csv',
                     sep=',')  # no se porq solo va con ruta absoluta
    print("Len videoconferencia: ", len(df))
    # 3- TfIdf Vectorized. Y separamos para train y test
    cvVideoconferencia = TfidfVectorizer(min_df=1, stop_words='english')
    x_trainVideoconferencia = df["Text"].values.astype('U')
    y_trainVideoconferencia = df["Class"]
    # 4- fit transform
    x_traincvVideoconferencia = cvVideoconferencia.fit_transform(x_trainVideoconferencia)
    # 6- Clasificador
    svcVideoconferencia = LinearSVC(random_state=0, tol=1e-5)
    mnbVideoconferencia = CalibratedClassifierCV(svcVideoconferencia)
    mnbVideoconferencia.fit(x_traincvVideoconferencia, y_trainVideoconferencia)  # lo entreno con el train
    filename = pathAbs + '/modelosClasificadorManuales/modelo_videoconferencia_' + formatoFecha + '.pkl'
    objectsVideoconferencia = (mnbVideoconferencia, x_trainVideoconferencia)
    pickle.dump(objectsVideoconferencia, open(filename, 'wb'))

    # 3- ENTRENAMIENTO CUESTIONARIOS
    df = pd.read_csv(pathAbs + '/archivos/csv/CuestionariosXTemas_ESCapado.csv',
                     sep=',')  # no se porq solo va con ruta absoluta
    print("Len cuestionarios: ", len(df))
    # 3- TfIdf Vectorized. Y separamos para train y test
    cvCuestionarios = TfidfVectorizer(min_df=1, stop_words='english')
    x_trainCuestionarios = df["Text"].values.astype('U')
    y_trainCuestionarios = df["Class"]
    # 4- fit transform
    x_traincvCuestionarios = cvCuestionarios.fit_transform(x_trainCuestionarios)
    # 6- Clasificador
    svcCuestionarios = LinearSVC(random_state=0, tol=1e-5)
    mnbCuestionarios = CalibratedClassifierCV(svcCuestionarios)
    mnbCuestionarios.fit(x_traincvCuestionarios, y_trainCuestionarios)  # lo entreno con el train
    filename = pathAbs + '/modelosClasificadorManuales/modelo_cuestionarios_' + formatoFecha + '.pkl'
    objectsCuestionarios = (mnbCuestionarios, x_trainCuestionarios)
    pickle.dump(objectsCuestionarios, open(filename, 'wb'))


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
    print(text)
    x_testcv = cv.transform(text)
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
    url = df_2.values[0][3]
    return url

def clasificarPregunta(pregunta):
    print("Pregunta: ",pregunta)
    preguntaCastellano, idioma=traducirCastellano(pregunta)
    preguntaCastellanoCapada=[capar(preguntaCastellano)]
    clasePredecida, threshold = clasificador('clase',preguntaCastellanoCapada)
    if threshold > 0.50:  # si el threshold es >50% clasificaremos como que será correcta la clase que predice
        # mirar el foro de la clase
        # si no se en cuentra mirar en el manual de la clase
        temaPredecido, thresholdTema = clasificador(clasePredecida, preguntaCastellanoCapada)

        # Habria que mirar el threshold. pero de momento que lo haga siempre
        # Conseguimos la url que le corresponde al tema y clase
        url = conseguirPaginaCorrespondiente(clasePredecida, temaPredecido)
    print("Clase predecida: ", clasePredecida, "  Threshold clase: ", threshold, "\nTema predecido: ", temaPredecido, "  Threshold tema: ",thresholdTema,"\n")
    return clasePredecida,temaPredecido,url,idioma
