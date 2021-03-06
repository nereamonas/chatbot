import pandas as pd
from stop_words import get_stop_words
import re
import spacy
import nltk
from autocorrect import Speller
from unicodedata import normalize
import csv
import sys


def capar(text):
    # 1- Poner to do en minusculas
    text = text.lower()
    # 2- quitar las url
    text = re.sub("https?://[^\s]+", "", text)  # quitar las url.
    # 3- quitar get_stop_words('spanish')
    stop_words = get_stop_words('spanish')
    for word in stop_words:
        if (word!='no'):
            text = text.replace(" " + word + " ", " ")  # a, al, algo, algunas...
    ###########################################################################################
    #   QUITAR <p> y </p>
    text = re.sub("<*>", "", text)
    #text = text.replace("<p>","")
    #text = text.replace("</p>", "")
    text = text.replace('\n', ' ')
    ###########################################################################################
    # 4- quitar simbolos
    symbols = "'!\"#$%&()*+-./:;<=,>?@[\]^_`{|}~“”"  # \n /// no puedo quitarle los " jaajaj text = text.replace('"', "")
    for i in symbols:
        text = text.replace(i, "")
    text=text.replace('"', '')
    # 5- quitar numeros
    text = re.sub("\d+", "", text)
    # 7- Stemming en español- SnowballStemmer('spanish')
    spanish_stemmer = nltk.stem.SnowballStemmer('spanish')
    text_words = text.split(" ")
    for palabra in text_words:
        text = text.replace(palabra, spanish_stemmer.stem(palabra))
    # 8- autocorreccion de palabras
    spell = Speller('es')
    text=spell(text)
    # 9- corrige un poco las palabras. pero ns si hace falta ahora...
    nlp = spacy.load('es_core_news_sm')
    doc= nlp(text)
    for token in doc:
        if (token.text != token.lemma_):
            #print(token.text, "|", token.lemma_, '|', token.pos_)
            text.replace(token.text,token.lemma_)
    # 10- eliminar tildes y simbolos raros. pero sin eliminar la ñ
    # -> NFD y eliminar diacríticos
    text = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1",normalize("NFD", text), 0, re.I)
    # -> NFC
    text = normalize('NFC', text)
    return text

#7.Estandarizar palabras
# frases como "estoy muuuuuy feliz" las convertimos en "estoy muy feliz"
# texto = ''.join(''.join(s)[:2] for _, s in itertools.groupby(texto))

csv.field_size_limit(sys.maxsize)
with open("galderak.csv", 'r') as g:
    galderak = csv.reader(g)
    with open("preguntasCapadas.csv", 'w') as cap:
        nuevo = csv.writer(cap)
        for row in galderak:
            row[3]=capar(row[3])
            nuevo.writerow(row)

