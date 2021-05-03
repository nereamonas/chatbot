import pandas as pd
import os
print(os.getcwd())

pathAbs=os.getcwd()

def añadirATodasLasPreguntasRealizadas(preguntaUsuario, clase, tema):
    #Guardaremos todas las preguntas que se realizen

    #Rellenamos tambn el csv de feedback completo
    df = pd.read_csv(pathAbs+'/archivos/Feedback/FeedbackCompleto.csv', sep=',')
    size = len(df)
    df.loc[size, 'Pregunta'] = preguntaUsuario
    df.loc[size, 'Clase'] = clase
    df.loc[size, 'Tema'] = tema
    df.to_csv(pathAbs+'/archivos/Feedback/FeedbackCompleto.csv', index=False)
    print("se ha añadido en el fichero FeedbackCompleto.csv el elemento: [" + preguntaUsuario +", "+clase+", "+str(tema)+"] \n")

def añadirFeedBackManuales(estado, preguntaUsuario):
    # Rellenamos tambn el csv de feedback completo
    df = pd.read_csv(pathAbs + '/archivos/Feedback/FeedbackCompleto.csv', sep=',')
    i = len(df) - 1
    continuar = True
    while (continuar):
        if (preguntaUsuario in df._get_value(i, 'Pregunta')):  # miramos si la pregunta esta dentro
            continuar = False
            df.loc[i, "Manuales"] = estado
        else:
            i -= 1
    df.to_csv(pathAbs + '/archivos/Feedback/FeedbackCompleto.csv', index=False)
    print("se ha actualizado en el fichero FeedbackCompleto.csv el elemento: " + preguntaUsuario + " con el feedback Manuales a " + estado+"\n")



def añadirFeedBackBotonesClase(clase, preguntaUsuario):
    #Despues de intentar redirigir la pregunta del usuario mediante botones, guardaremos en bien o mal clasificados el feedback del usuario
    # Rellenamos tambn el csv de feedback completo
    df = pd.read_csv(pathAbs + '/archivos/Feedback/FeedbackCompleto.csv', sep=',')
    i = len(df) - 1
    continuar = True
    while (continuar):
        if (preguntaUsuario in df._get_value(i, 'Pregunta')):  # miramos si la pregunta esta dentro
            continuar = False
            df.loc[i, "ClaseBotones"] = clase
        else:
            i -= 1
    df.to_csv(pathAbs + '/archivos/Feedback/FeedbackCompleto.csv', index=False)
    print("se ha actualizado en el fichero FeedbackCompleto.csv el elemento: " + preguntaUsuario + " con el feedback Botones Clase a " + clase+"\n")


def añadirFeedBackBotonesTema(tema, preguntaUsuario):
    #Despues de intentar redirigir la pregunta del usuario mediante botones, guardaremos en bien o mal clasificados el feedback del usuario
    # Rellenamos tambn el csv de feedback completo
    df = pd.read_csv(pathAbs + '/archivos/Feedback/FeedbackCompleto.csv', sep=',')
    i = len(df) - 1
    continuar = True
    while (continuar):
        if (preguntaUsuario in df._get_value(i, 'Pregunta')):  # miramos si la pregunta esta dentro
            continuar = False
            df.loc[i, "TemaBotones"] = tema
        else:
            i -= 1
    df.to_csv(pathAbs + '/archivos/Feedback/FeedbackCompleto.csv', index=False)
    print("se ha actualizado en el fichero FeedbackCompleto.csv el elemento: " + preguntaUsuario + " con el feedback Botones tema a " + str(tema) +"\n")


def añadirFeedBackBotones(estado, preguntaUsuario):
    #Despues de intentar redirigir la pregunta del usuario mediante botones, guardaremos en bien o mal clasificados el feedback del usuario
    # Rellenamos tambn el csv de feedback completo
    df = pd.read_csv(pathAbs + '/archivos/Feedback/FeedbackCompleto.csv', sep=',')
    i = len(df) - 1
    continuar = True
    while (continuar):
        if (preguntaUsuario in df._get_value(i, 'Pregunta')):  # miramos si la pregunta esta dentro
            continuar = False
            df.loc[i, "Botones"] = estado
        else:
            i -= 1
    df.to_csv(pathAbs + '/archivos/Feedback/FeedbackCompleto.csv', index=False)
    print("se ha actualizado en el fichero FeedbackCompleto.csv el elemento: " + preguntaUsuario + " con el feedback Botones a " + estado+"\n")
