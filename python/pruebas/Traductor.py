import re
from locale import normalize

import googletrans
import nltk
import spacy
from autocorrect import Speller
from stop_words import get_stop_words

texto="Nire izena Nerea da"
texto2="Nola grabatu dezaket bbc klase bat?"

from googletrans import Translator

print(googletrans.LANGUAGES)

translator=Translator()

translation = translator.translate(texto,dest='es')
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")

translation = translator.translate(texto2,dest='es')
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")

spanish_stemmer = nltk.stem.SnowballStemmer('spanish')
print(spanish_stemmer.stem("llamar"))
print(spanish_stemmer.stem("llamo"))
print(spanish_stemmer.stem("llamaban"))
stop_words = get_stop_words('spanish')
print(stop_words)


text="sabes hacerlo podrias solucionarlo ayudarme tener"
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

print(text)
