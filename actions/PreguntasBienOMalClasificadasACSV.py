from re import split
import os
print(os.getcwd())
pathAbs=os.getcwd()

def añadir(estado, preguntaUsuario, respuestaBot):
    tema = [int(temp)for temp in respuestaBot.split() if temp.isdigit()][0]
    clase = ''
    if 'videoconferencia' in respuestaBot:
        clase = 'videoconferencia'
    elif 'cuestionario':
        clase = 'cuestionarios'

    if estado=='bien':
        archivo=pathAbs+'/archivos/examinar/bienClasificados.csv'
    else:
        archivo=pathAbs+'/archivos/examinar/malClasificados.csv'
    if clase != '':
        # Guardamos en el csv
        import csv
        with open(archivo, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([preguntaUsuario, clase, str(tema)])
            print("se ha registrado en el fichero "+estado+"Clasificados.csv el elemento: ["+preguntaUsuario+ ", "+clase+", "+str(tema)+"]")
        file.close()
