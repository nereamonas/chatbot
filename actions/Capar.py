import re

import nltk
import spacy
from unicodedata import normalize
from stop_words import get_stop_words

stop_words = get_stop_words('spanish')
stop_words.remove("no")
stop_words.remove("sin")

spanish_stemmer = nltk.stem.SnowballStemmer('spanish')

nlp = spacy.load('es_core_news_sm')

def capar(text):
    text=" "+text+" "
    # 1- Poner to do en minusculas
    text = text.lower()
    # 2- quitar las url
    text = re.sub("https?://[^\s]+", "", text)  # quitar las url.
    # 3- quitar simbolos
    symbols = "'!\"#$%&()*+-./:;<=,>?@[\]^_`{|}~“”¿¡ºª·"  # \n /// no puedo quitarle los " jaajaj text = text.replace('"', "")
    for i in symbols:
        text = text.replace(i, "")
    text=text.replace('"', '')
    # 4- quitar numeros
    text = re.sub("\d+", "", text)
    # 5- quitar get_stop_words('spanish')
    for word in stop_words:
        text = text.replace(" " + word + " ", " ")  # a, al, algo, algunas...
    # 6- Stemming en español- SnowballStemmer('spanish')
    text_words = text.split(" ")
    for palabra in text_words:
        text = text.replace(palabra, spanish_stemmer.stem(palabra))
    # 7- corrige un poco las palabras. pero ns si hace falta ahora...
    doc= nlp(text)
    for token in doc:
        if (token.text != token.lemma_):
            text.replace(token.text,token.lemma_)
    # 9- eliminar tildes y simbolos raros. pero sin eliminar la ñ
    # -> NFD y eliminar diacríticos
    text = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1",normalize("NFD", text), 0, re.I)
    # -> NFC
    text = normalize('NFC', text)
    #Eliminar verbos comunes que no aportan información. como: sabes, hacer, podrías. Para evitar que clasifique preguntas como Sabes como hacerlo?
    lista=["sab","hac","podri", "ayud"]  #solucion
    for word in lista:
        text = text.replace(" " + word + " ", " ")
    return text

