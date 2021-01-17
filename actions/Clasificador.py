import math, re
from collections import defaultdict
from stop_words import get_stop_words
import nltk.stem
from langdetect import detect

class Clasificador():
	def __init__(self):
		self.clases = defaultdict(lambda : defaultdict(int))

	def dividirPorPalabras(self,text):  # Obtenemos todo el texto y lo dividimos por palabras
		#Detectar idioma
		idioma=detect(text)  # el de euskera no detecta. entonces sera o es o otros q equivale a eu

		# 1- Poner to do en minusculas
		text = text.lower()

		# 2- quitado las url
		text = re.sub("https?://[^\s]+", "", text)  # quitar las url.

		# 3- quitar get_stop_words('spanish')
		if ("es"==idioma):
			stop_words = get_stop_words('spanish')
		else:
			stop_words=["al","anitz","arabera","asko","baina","bat","batean","batek","bati","batzuei","batzuek","batzuetan","batzuk","bera","beraiek","berau","berauek","bere","berori","beroriek","beste","bezala","da","dago","dira","ditu","du","dute","edo","egin","ere","eta","eurak","ez","gainera","gu","gutxi","guzti","haiei","haiek","haietan","hainbeste","hala","han","handik","hango","hara","hari","hark","hartan","hau","hauei","hauek","hauetan","hemen","hemendik","hemengo","hi","hona","honek","honela","honetan","honi","hor","hori","horiei","horiek","horietan","horko","horra","horrek","horrela","horretan","horri","hortik","hura","izan","ni","noiz","nola","non","nondik","nongo","nor","nora","ze","zein","zen","zenbait","zenbat","zer","zergatik","ziren","zituen","zu","zuek","zuen","zuten"] #https://github.com/stopwords-iso/stopwords-eu

		for word in stop_words:
			text=text.replace(" "+word+" "," ") # a, al, algo, algunas...

		# 4- quitar simbolos
		symbols = "'!\"#$%&()*+-./:;<=>?@[\]^_`{|}~"  #\n /// no puedo quitarle los " jaajaj text = text.replace('"', "")
		for i in symbols:
			text=text.replace(i,"")

		# 5- quitar numeros
		text = re.sub("\d+", "", text)

		# 6- mas de dos letras y letras entre a-z
		# Dividimos en palabras. ya no es texto
		palabras = re.findall("[a-z]{2,}", text, re.I)  # [re.I]	It ignores case

		# 7- SnowballStemmer('spanish')
		if ("es"==idioma):
			spanish_stemmer = nltk.stem.SnowballStemmer('spanish')
		else:
			spanish_stemmer = nltk.stem.SnowballStemmer('spanish')  # creo q no hay esta biblioteca en euskera ns o si jj
		palabras = [spanish_stemmer.stem(palabra) for palabra in palabras]

		return palabras

	def train(self, texto, clase): # Metodo para entrenar al clasificador
		palabras = self.dividirPorPalabras(texto)  # conseguimos las palabras
		# nos guardara en la posicion de cada palabra la cantidad de veces que se repite.  ejemplo: {'videoconferencias': 1, 'con': 35, 'collaborate': 7, 'ultra': 5, 'manual': 6, 'para': 52, 'el': 128}
		for word in palabras:
			self.clases[clase][word] += 1


	def sacarProbabilidad(self,palabras,clase): # Parte logica, la cual etiquetara al texto
		#AQUI HAY MAS INFO y implementacion mas compleja: https://www.instintoprogramador.com.mx/2019/07/python-para-nlp-crear-un-modelo-tf-idf.html
		#https://github.com/LuisAlejandroSalcedo/StringTagger-Clasificador-de-Texto
		palabrasTotales = sum(self.clases[clase].values())  #cuantas palabras hay en cada clase
		#print (numWords)
		logSum = 0
		for palabra in palabras:  # analizaremos cada palabra de la frase
			#print (self.clases[clase][palabra])
			freq = self.clases[clase][palabra] + 1  #suma una repeticion
			#print (freq)
			prob = freq/palabrasTotales #freq actual de la palabra / palabras totales  TF = (Frequency of the word in the sentence) / (Total number of words in the sentence)
			#print (prob)
			logSum += math.log(prob)   #calcula el log. y vamos sumando todos los resultados. IDF: log((Total number of sentences (documents))/(Number of sentences (documents) containing the word))
			#print(logSum)

		return logSum
	
	def clasificaPregunta(self,texto): # Rrecibe la pregunt ay la clasifica
		probs = {}
		palabras = self.dividirPorPalabras(texto)  #dividimos la pregunta por palabras
		for clase in self.clases:  # tendrá que mirar por todas las clases cual es su probabilidad para así quedarse con la más alta
			probs[clase] = self.sacarProbabilidad(palabras,clase)
		
		elegida = sorted(probs,key=probs.get)[-1]  #ns quedamos con el de mayor probabilidad

		return elegida



