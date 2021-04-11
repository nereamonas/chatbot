# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from actions.Clasificador import clasificarPregunta, inicializarModelo
from actions.FeedbackACSV import añadirFeedBackManuales, añadirFeedBackBotones, añadirTodasLasPreguntasRealizadas
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

inicializarModelo(False) #Inicializamos el modelo. True si queremos que cree un nuevo modelo. False si queremos que cargue un modelo en concreto

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
        textoRespuestaUsuario = 'Tu pregunta puede pertenecer al tema ' + clase + ", sección " + str(tema) + "\n[Quizá encuentres tu respuesta aquí](" + url + ')'
        buttons = [{"payload": "/affirm", "title": "Si"}, {"payload": "/deny", "title": "No"}]
        if idioma=='eu':
            if clase=='videoconferencia':
                clase='Bideokonferentzia'
            elif clase=='cuestionarios':
                clase='Galdetegiak'
            url=url.replace('/es/','/eu/')
            textRespuesta = 'Sailkatutako klasea: ' + clase + "\nSailkatutako gaia: " + str(tema) + "\nUrl: " + url
            textoHaServidoDeAyuda = "Lagungarria izan da?"
            textoRespuestaUsuario = 'Zure galdera ' + clase + " gaiko, " + str(tema) + " atalekoa izan daiteke \n[Seguruenik erantzuna hemen topatuko duzu](" + url + ')'
            buttons = [{"payload": "/affirm", "title": "Bai"}, {"payload": "/deny", "title": "Ez"}]  #al deny le pasamos el idioma del usuario
        #dispatcher.utter_message(text=textRespuesta)
        #dispatcher.utter_message(attachment=url)
        dispatcher.utter_message(text=textoRespuestaUsuario)
        dispatcher.utter_message(text=textoHaServidoDeAyuda,buttons=buttons)
        añadirTodasLasPreguntasRealizadas(pregunta,textoRespuestaUsuario)
        # Almacenamos el idioma en un slot, un slot es como la memoria del bot. asi en to do momento podremos saber el idioma en el que ha hablado el usuairo
        return [SlotSet("idioma",idioma)]

class ActionNoHaSidoDeAyuda(Action):

     def name(self) -> Text:
        return "action_noHaSidoDeAyuda"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

         idioma=tracker.get_slot('idioma') # asi cojo el valor del slot idioma
         preguntaUsuario=tracker.events[len(tracker.events)-9].get('text')
         añadirFeedBackManuales('mal', preguntaUsuario)

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
             payload = "/claseBotones{\"claseTitulo\": \"" + i+ "\"}"
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

        idioma = tracker.get_slot('idioma')  # asi cojo el valor del slot idioma
        if (idioma=='eu'):
            textoRespuesta="Eskerrik asko"
        else:
            textoRespuesta="Muchas gracias"

        preguntaUsuario=tracker.events[len(tracker.events)-9].get('text')
        añadirFeedBackManuales('bien',preguntaUsuario)
        dispatcher.utter_message(text=textoRespuesta)

        return []



class ActionBotonesClase(Action):

    def name(self) -> Text:
        return "action_botonesClase"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        idioma = tracker.get_slot('idioma')  # asi cojo el valor del slot idioma
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
            payload = "/buscarPorBotones{\"temaTitulo\": \"" + df_1.values[i][2] + "\"}"
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
        idioma = tracker.get_slot('idioma')  # asi cojo el valor del slot idioma

        pathAbs = os.getcwd()
        if (idioma == 'eu'):
            df = pd.read_csv(pathAbs + '/archivos/csv/paginaCorrespondienteEU.csv', sep=',')
            textoRespuesta="[Seguruenik erantzuna hemen topatuko duzu]("
            textoHaServidoDeAyuda = "Lagungarria izan da?"
            buttons = [{"payload": "/affirm", "title": "Bai"},
                       {"payload": "/deny","title": "Ez"}]  # al deny le pasamos el idioma del usuario
        else:
            df = pd.read_csv(pathAbs + '/archivos/csv/paginaCorrespondiente.csv', sep=',')
            textoRespuesta="[Quizá encuentres tu respuesta aquí]("
            textoHaServidoDeAyuda = "¿Te ha servido de ayuda?"
            buttons = [{"payload": "/affirm", "title": "Si"},
                       {"payload": "/deny", "title": "No"}]
        titulo = df.loc[:, 'titulo'] == titulo
        df_1 = df.loc[titulo]
        url = df_1.values[0][3]

        dispatcher.utter_message(text=textoRespuesta + url + ')')
        dispatcher.utter_message(text=textoHaServidoDeAyuda,buttons=buttons)

        return []


class ActionNoHaSidoDeAyudaBotones(Action):

     def name(self) -> Text:
        return "action_noHaSidoDeAyudaBotones"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:


         idioma=tracker.get_slot('idioma') # asi cojo el valor del slot idioma
         pregunta = tracker.events[len(tracker.events) - 26].get('text')  #Cojo del x mensaje del usuario la pregunta

         if(idioma=='eu'):
             textoRespuesta1="Sentitzen dut lagundu ez izatea"
         else:
             textoRespuesta1 ="Sentimos no haberte podido ayudar."

         añadirFeedBackBotones('mal', pregunta)

         dispatcher.utter_message(text=textoRespuesta1)

         return []


class ActionHaSidoDeAyudaBotones(Action):

    def name(self) -> Text:
        return "action_haSidoDeAyudaBotones"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        idioma = tracker.get_slot('idioma')  # asi cojo el valor del slot idioma
        pregunta = tracker.events[len(tracker.events) - 26].get('text')  # Cojo del x mensaje del usuario la pregunta

        if (idioma=='eu'):
            textoRespuesta="Eskerrik asko"
        else:
            textoRespuesta="Muchas gracias"

        añadirFeedBackBotones('bien', pregunta)
        dispatcher.utter_message(text=textoRespuesta)

        return []

class ActionSesionStart(Action):

    def name(self) -> Text:
        return "action_session_start"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hola! Soy un chatbot, que puede responder preguntas sobre dos temas: Videoconferencias y Cuestionarios. \n Házme una pregunta y te daré la respuesta.")

        return []


"""
Pruebas tracker
cont=0
        for event in (list(reversed(tracker.events)))[:35]:
            if event.get("event") == "user":
                print(cont)
                print(" USER  "+event.get("text"))
            if event.get("event") == "bot":
                print(cont)
                print(" BOT  "+event.get("text"))
            cont+=1

"""