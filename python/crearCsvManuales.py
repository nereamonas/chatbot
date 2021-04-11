import docx2txt
import pandas as pd
from actions.Capar import capar

#Esta por pasos. Es mejor ir haciendolo paso a paso comprobando que lo haga correctamente y modificando los errores o las cosas que no encuentre correctamente.

#1- PRIMERO CONVERTIMOS LOS DOCX EN TXT PARA TRATARLOS MEJOR
#He convertido todos, aunque solo se usarán Videoconferencias y cuestionarios ES
def pasoUno():
    print("Paso 1 - Inicio - Convertimos los docx en txt para tratarlos mejor")
    MY_TEXT = docx2txt.process("../archivos/docx/1-Estructura_ES.docx")
    with open("../archivos/txt/1-Estructura_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/1-Estructura_EUS.docx")
    with open("../archivos/txt/1-Estructura_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/2-GestionUsuarios_ES.docx")
    with open("../archivos/txt/2-GestionUsuarios_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/2-GestionUsuarios_EUS.docx")
    with open("../archivos/txt/2-GestionUsuarios_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/3-Registro_ES.docx")
    with open("../archivos/txt/3-Registro_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/4-Registro_EUS.docx")
    with open("../archivos/txt/4-Registro_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/5-HerramientasComunicacion_ES.docx")
    with open("../archivos/txt/5-HerramientasComunicacion_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/5-HerramientasComunicacion_EUS.docx")
    with open("../archivos/txt/5-HerramientasComunicacion_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/6-Videoconferencia_ES.docx")
    with open("../archivos/txt/6-Videoconferencia_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/6-Videoconferencia_EUS.docx")
    with open("../archivos/txt/6-Videoconferencia_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/7-GestionRecursos_ES.docx")
    with open("../archivos/txt/7-GestionRecursos_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/7-GestionRecursos_EUS.docx")
    with open("../archivos/txt/7-GestionRecursos_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/8-Cuestionarios_ES.docx")
    with open("../archivos/txt/8-Cuestionarios_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/8-Cuestionarios_EUS.docx")
    with open("../archivos/txt/8-Cuestionarios_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/9-H5P_ES.docx")
    with open("../archivos/txt/9-H5P_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/9-H5P_EUS.docx")
    with open("../archivos/txt/9-H5P_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/10-Seguimiento_ES.docx")
    with open("../archivos/txt/10-Seguimiento_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/10-Seguimiento_EUS.docx")
    with open("../archivos/txt/10-Seguimiento_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/11-Colaborativas_ES.docx")
    with open("../archivos/txt/11-Colaborativas_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/11-Colaborativas_EUS.docx")
    with open("../archivos/txt/11-Colaborativas_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/12-Taller_ES.docx")
    with open("../archivos/txt/12-Taller_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/12-Taller_EUS.docx")
    with open("../archivos/txt/12-Taller_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/13-Calificaciones_ES.docx")
    with open("../archivos/txt/13-Calificaciones_ES.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)

    MY_TEXT = docx2txt.process("../archivos/docx/13-Calificaciones_EUS.docx")
    with open("../archivos/txt/13-Calificaciones_EUS.txt", "w") as text_file:
        print(MY_TEXT, file=text_file)
    print("Paso 1 - Fin")

#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------

#2- CONVERTIMOS LOS TXT EN CSV
def pasoDos():
    print("Paso 2 - Inicio - Convertimos los txt en csv. Solo vamos a tratar los temas: videoconferencias y cuestionarios")
    #2.1- VIDEOCONFERENCIAS. Convertimos de txt a csv
    df = pd.read_csv("../archivos/txt/6-Videoconferencia_ES.txt", delimiter = '\n', error_bad_lines=False)
    df.to_csv('../archivos/csvIntermedios/Videoconferencia_ES.csv')

    #2.2- CUESTIONARIOS
    df = pd.read_csv("../archivos/txt/8-Cuestionarios_ES.txt", delimiter = '\n', error_bad_lines=False)
    df.to_csv('../archivos/csvIntermedios/Cuestionarios_ES.csv')

    print("Paso 2 - Fin")


#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------

#3- Añadiremos la columna class al csv, para indicar a que clase pertenece el texto
#Antes de realizar este paso hay q editar manualmente un poco el fichero. el problema es que el indice, lo crea en diferentes columnas, por la separación y asi, entonces hay q unirlos en una misma celda y luego ya seguir
#Solo debe estar la primera columna con el numero y la segunda con el texto, lo demas vacio
def pasoTres():
    print("Paso 3 - Inicio - Añadiremos la columna class al csv")
    #3.1- VIDEOCONFERENCIAS. borramos la priera columna y agregamos una nueva columna llamada class
    df2 = pd.read_csv('../archivos/csvIntermedios/Videoconferencia_ES.csv',names=["Number", "Text", "Class"])
    df2.drop('Number', axis = 1, inplace = True)
    df2['Class']='videoconferencia'
    print(df2)
    df2.to_csv('../archivos/csvIntermedios/VideoconferenciaClass_ES.csv', index = False)

    #3.2- CUESTIONARIOS
    df2 = pd.read_csv('../archivos/csvIntermedios/Cuestionarios_ES.csv',names=["Number","Text", "Class"])
    df2.drop('Number', axis = 1, inplace = True)
    df2['Class']='cuestionarios'
    print(df2)
    df2.to_csv('../archivos/csvIntermedios/CuestionariosClass_ES.csv', index = False)
    print("Paso 3 - Fin")

#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------

#3 - CONCATENAR LOS DOS FICHEROS
def pasoCuatro():
    print("Paso 4 - Inicio - Unimos los dos ficheros (videoconferencias y cuestionarios) en un mismo csv")
    #unir todos en un solo csv para meter al clasificador: Sin poner la primera linea
    df1 = pd.read_csv('../archivos/csvIntermedios/VideoconferenciaClass_ES.csv')
    df2 = pd.read_csv('../archivos/csvIntermedios/CuestionariosClass_ES.csv')
    out = df1.append(df2)
    out.to_csv('../archivos/csvIntermedios/concatenate.csv', index = False)
    print("Paso 4 - Fin")

#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------

#Crearemos una copia del de cuestionarios y videoconferencias y tendremos que escribir a mano cada frase a q numero de tema pertenece. pero es bastante rapido
def pasoCinco():
    print("Paso 5 - Inicio - Manualmente hay q decir cada frase a que tema pertenece")

    df2 = pd.read_csv('../archivos/csvIntermedios/Videoconferencia_ES.csv', names=["Number", "Text", "Class"])
    df2.drop('Number', axis=1, inplace=True)
    df2['Class'] = ''
    df2.to_csv('../archivos/csvIntermedios/VideoconferenciaXTemas_ES.csv', index=False)

    df2 = pd.read_csv('../archivos/csvIntermedios/Cuestionarios_ES.csv', names=["Number", "Text", "Class"])
    df2.drop('Number', axis=1, inplace=True)
    df2['Class'] = ''
    df2.to_csv('../archivos/csvIntermedios/CuestionariosXTemas_ES.csv', index=False)

    print("Paso 5 - Fin")
#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------


#6- CAPAMOS LOS TEXTOS DE LOS 3 FICHEROS
def pasoSeis():
    print("Paso 6 - Inicio - Crearemos los ficheros finales - Capamos los textos de los 3 ficheros para conseguir un mejor resultado de las predicciones")

    df3 = pd.read_csv('../archivos/csvIntermedios/concatenate.csv')
    clase = df3["text"]
    count = 0
    for x in clase:
        clase[count] = capar(x)
        count += 1
    df3["text"] = clase
    df3.to_csv('../archivos/csv/concatenateCapado.csv', index=False)


    df4 = pd.read_csv('../archivos/csvIntermedios/VideoconferenciaXTemas_ES.csv')
    clase = df4["text"]
    count = 0
    for x in clase:
        clase[count] = capar(x)
        count += 1
    df4["text"] = clase
    df4.to_csv('../archivos/csv/VideoconferenciaXTemas_ESCapado.csv', index=False)


    df5 = pd.read_csv('../archivos/csvIntermedios/CuestionariosXTemas_ES.csv')
    clase=df5["text"]
    count=0
    for x in clase:
        clase[count]=capar(x)
        count+=1
    df5["text"]=clase
    df5.to_csv('../archivos/csv/CuestionariosXTemas_ESCapado.csv', index = False)
    print("Paso 5 - Fin")



#Esta por pasos. Es mejor ir haciendolo paso a paso comprobando que lo haga correctamente y modificando los errores o las cosas que no encuentre correctamente.

#PROCESO A EJECUTAR:
#pasoUno()
#pasoDos()
#Pause - Hay q hacer cambios a mano antes de seguir
#pasoTres()
#pasoCuatro()
#pasoCinco()
#Pause - Hay que hacer cambio a mano antes de seguir
#pasoSeis()
#Ya tendriamos los csv para la clasificacion. Si se quiere añadir algun dato suelo no hay q seguir to do el proceso. Se capa la frase y se añade manualmente al final de los csv indicando la clase y el tema correcponiente












