from googletrans import Translator


def traducirCastellano(text):
    traductor = Translator()
    traduccion = traductor.translate(text, dest='es')
    print(f"{traduccion.origin} ({traduccion.src}) --> {traduccion.text} ({traduccion.dest})")
    return traduccion.text, traduccion.src

def traducirEuskera(text):
    traductor = Translator()
    traduccion = traductor.translate(text, dest='eu')
    print(f"{traduccion.origin} ({traduccion.src}) --> {traduccion.text} ({traduccion.dest})")
    return traduccion.text