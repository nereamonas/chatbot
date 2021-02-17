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
            buttons = [{"payload": "/affirm", "title": "Bai"}, {"payload": "/deny", "title": "Ez"}]

        #dispatcher.utter_message(text=textRespuesta)
        #dispatcher.utter_message(attachment=url)
        dispatcher.utter_message(text=textoRespuestaUsuario)
        dispatcher.utter_message(text=textoHaServidoDeAyuda,buttons=buttons)

        #Me he dado cuenta q los manuales en pdf son mas cortros que los del doc. entonces las paginas no cuadran.


# dispatcher.utter_message('<a href="">Click para abrir el manual</a>')
# dispatcher.utter_message('[Click para abrir el manual](https://www.ehu.eus/documents/1852718/14189177/Bilera+birtualak++Collaborate-rekin+Irakasleentzako+eskuliburua.pdf/a393e0d9-29ee-5e8c-9122-d1983a4939d5)')