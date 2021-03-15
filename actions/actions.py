# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from actions.Clasificador import clasificarPregunta
from actions.PreguntasBienOMalClasificadasACSV import añadir
import os
import pandas as pd
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []


class ActionDefaultFallback(Action):
    """Executes the fallback action and goes back to the previous state
    of the dialogue"""

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        pregunta = tracker.latest_message.get('text')
        clase,tema,url,idioma=clasificarPregunta(pregunta)
        textRespuesta='Clase clasificada: '+clase+ "\nTema predecido: "+str(tema)+"\nUrl: "+url
        textoHaServidoDeAyuda="¿Te ha servido de ayuda?"
        textoRespuestaUsuario = 'Tu pregunta puede pertenece al tema ' + clase + ", sección " + str(tema) + "\n[Quizá encuentres tu respuesta aquí](" + url + ')'
        buttons = [{"payload": "/affirm{\"idioma\": \"" + idioma + "\"}", "title": "Si"}, {"payload": "/deny{\"idioma\": \"" + idioma + "\"}", "title": "No"}]
        if idioma=='eu':
            if clase=='videoconferencia':
                clase='Bideokonferentzia'
            elif clase=='cuestionarios':
                clase='Galdetegiak'
            url=url.replace('/es/','/eu/')
            textRespuesta = 'Sailkatutako klasea: ' + clase + "\nSailkatutako gaia: " + str(tema) + "\nUrl: " + url
            textoHaServidoDeAyuda = "Lagungarria izan da?"
            textoRespuestaUsuario = 'Zure galdera ' + clase + " gaiko, " + str(tema) + " atalekoa izan daiteke \n[Seguruenik erantzuna hemen topatuko duzu](" + url + ')'
            buttons = [{"payload": "/affirm{\"idioma\": \"" + idioma + "\"}", "title": "Bai"}, {"payload": "/deny{\"idioma\": \"" + idioma + "\"}", "title": "Ez"}]  #al deny le pasamos el idioma del usuario

        #dispatcher.utter_message(text=textRespuesta)
        #dispatcher.utter_message(attachment=url)
        dispatcher.utter_message(text=textoRespuestaUsuario)
        dispatcher.utter_message(text=textoHaServidoDeAyuda,buttons=buttons)

        #Me he dado cuenta q los manuales en pdf son mas cortros que los del doc. entonces las paginas no cuadran.


# dispatcher.utter_message('<a href="">Click para abrir el manual</a>')
# dispatcher.utter_message('[Click para abrir el manual](https://www.ehu.eus/documents/1852718/14189177/Bilera+birtualak++Collaborate-rekin+Irakasleentzako+eskuliburua.pdf/a393e0d9-29ee-5e8c-9122-d1983a4939d5)')


class ActionNoHaSidoDeAyuda(Action):

     def name(self) -> Text:
        return "action_noHaSidoDeAyuda"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

         idioma=tracker.latest_message.get('entities')[0].get('value')  # asi cojo el valor del entity idioma
         for event in (list(reversed(tracker.events)))[:5]:
             if event.get("event") == "user":
                 preguntaUsuario = event.get("text")
             if event.get("event") == "bot":
                 respuestaBot = event.get("text")
         preguntaUsuario=tracker.events[len(tracker.events)-8].get('text')
         respuestaBot=tracker.events[len(tracker.events) - 5].get('text')
         añadir('mal', preguntaUsuario, respuestaBot)

         pathAbs = os.getcwd()
         if(idioma=='eu'):
             df = pd.read_csv(pathAbs + '/archivos/csv/paginaCorrespondienteEU.csv', sep=',')
             textoRespuesta1="Sentitzen dut lagundu ez izatea"
             textoRespuesta2="Saia gaitezen beste modu batera. Ze gairi buruz doa zure galdera?"
         else:
             df = pd.read_csv(pathAbs + '/archivos/csv/paginaCorrespondiente.csv', sep=',')
             textoRespuesta1 ="Sentimos no haberte podido ayudar."
             textoRespuesta2 ="Vamos a probar de otra manera. ¿Sobre cual de estos temas trata tu pregunta?"
         uniqueClases=df['Class'].unique()
         buttons = []
         for i in uniqueClases:
             payload = "/claseBotones{\"claseTitulo\": \"" + i+ "\", \"idioma\": \""+idioma+"\"}"
             buttons.append({"title": i, "payload": payload})

         dispatcher.utter_message(text=textoRespuesta1)

         dispatcher.utter_message(text=textoRespuesta2, buttons=buttons)

         return []


class ActionHaSidoDeAyuda(Action):

    def name(self) -> Text:
        return "action_haSidoDeAyuda"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        idioma = tracker.latest_message.get('entities')[0].get('value')
        if (idioma=='eu'):
            textoRespuesta="Eskerrik asko"
        else:
            textoRespuesta="Muchas gracias"

        for event in (list(reversed(tracker.events)))[:5]:
            if event.get("event")=="user":
                preguntaUsuario = event.get("text")
            if event.get("event") == "bot":
                respuestaBot=event.get("text")

        preguntaUsuario=tracker.events[len(tracker.events)-8].get('text')
        respuestaBot=tracker.events[len(tracker.events) - 5].get('text')
        añadir('bien',preguntaUsuario,respuestaBot)
        dispatcher.utter_message(text=textoRespuesta)

        return []



class ActionBotonesClase(Action):

    def name(self) -> Text:
        return "action_botonesClase"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        idioma=tracker.latest_message.get('entities')[1].get('value')  #Cogemos el entity que hemos guardado del idioma
        clase=tracker.latest_message.get('entities')[0].get('value')  # cogemos el entity que hemos guardado de titulo

        # Tenemos q mandarle botones
        pathAbs = os.getcwd()
        if (idioma == 'eu'):
            df = pd.read_csv(pathAbs + '/archivos/csv/paginaCorrespondienteEU.csv', sep=',')
            textoRespuesta=clase+" gaiaren barnean, hauetako ze aukerei dagokio zure galdera?"
        else:
            df = pd.read_csv(pathAbs + '/archivos/csv/paginaCorrespondiente.csv', sep=',')
            textoRespuesta ="Dentro del tema "+clase+" cual de estas opciones concuerda más con tu pregunta? "
        clasee = df.loc[:, 'Class'] == clase
        df_1 = df.loc[clasee]
        buttons = []
        for i in range(len(df_1)):
            payload = "/buscarPorBotones{\"temaTitulo\": \"" + df_1.values[i][2] + "\", \"idioma\": \""+idioma+"\"}"
            buttons.append({"title": df_1.values[i][2], "payload": payload})

        dispatcher.utter_message(text=textoRespuesta, buttons=buttons)

        return []


class ActionPagBoton(Action):

    def name(self) -> Text:
        return "action_pagBoton"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        titulo = tracker.latest_message.get('entities')[0].get('value')  # asi cojo el valor del entity idioma
        idioma = tracker.latest_message.get('entities')[1].get('value')  # asi cojo el valor del entity idioma

        pathAbs = os.getcwd()
        if (idioma == 'eu'):
            df = pd.read_csv(pathAbs + '/archivos/csv/paginaCorrespondienteEU.csv', sep=',')
            textoRespuesta="[Seguruenik erantzuna hemen topatuko duzu]("
        else:
            df = pd.read_csv(pathAbs + '/archivos/csv/paginaCorrespondiente.csv', sep=',')
            textoRespuesta="[Quizá encuentres tu respuesta aquí]("
        titulo = df.loc[:, 'titulo'] == titulo
        df_1 = df.loc[titulo]
        url = df_1.values[0][3]

        dispatcher.utter_message(text=textoRespuesta + url + ')')

        return []

class ActionSesionStart(Action):

    def name(self) -> Text:
        return "action_session_start"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="HOLA")

        return []
