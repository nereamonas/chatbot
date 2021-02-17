import googletrans

texto="Nire izena Nerea da"
texto2="Nola grabatu dezaket bbc klase bat?"

from googletrans import Translator

print(googletrans.LANGUAGES)

translator=Translator()

translation = translator.translate(texto,dest='es')
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")

translation = translator.translate(texto2,dest='es')
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
