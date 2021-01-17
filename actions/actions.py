# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#
from rasa.shared.core.events import SessionStarted, ActionExecuted
from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType
from rasa_sdk.executor import CollectingDispatcher
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

from Clasificador import Clasificador
clasificador = Clasificador()  #clasificador
hola="PEPE"
training_data = {
    "videoconferenciaES":['txt/6-Videoconferencia_ES.txt'],
    "videoconferenciaEU": ['txt/6-Videoconferencia_EUS.txt'],
    "cuestionarioES": ['txt/8-Cuestionarios_ES.txt'],
    "cuestionarioEU": ['txt/8-Cuestionarios_EUS.txt'],
}
#ENTRENAR
for categoria, ficheros in training_data.items(): #recorremos todos los elementos del training data
    for fichero in ficheros:
        print("Clasificando "+ fichero)
        f = open(fichero, "r")  #cogemos el fichero
        contents = f.read()  #guardamos to do su contenido en un atr
        clasificador.train(contents,categoria)  #lo entrenamos


class ActionDefaultFallback(Action):
    """Executes the fallback action and goes back to the previous state
    of the dialogue"""

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        pregunta = tracker.latest_message.get('text')
        resultado = clasificador.clasificaPregunta(pregunta)
        respuesta="Clase clasificada: "+ resultado

        dispatcher.utter_message(text=respuesta)

        #Ahora con la clase mirar en el foro

        # si no se en cuentra mirar en el manual

        return []
