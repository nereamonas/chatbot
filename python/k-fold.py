
# compare the number of repeats for repeated k-fold cross-validation
from scipy.stats import sem
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
import os
import re
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from stop_words import get_stop_words
print(os.getcwd())
pathAbs=os.getcwd()


def capar(text):
    # 1- Poner to do en minusculas
    text = text.lower()
    # 2- quitado las url
    text = re.sub("https?://[^\s]+", "", text)  # quitar las url.
    # 3- quitar get_stop_words('spanish')
    stop_words = get_stop_words('spanish')
    for word in stop_words:
        text = text.replace(" " + word + " ", " ")  # a, al, algo, algunas...
    # 4- quitar simbolos
    symbols = "'!\"#$%&()*+-./:;<=,>?@[\]^_`{|}~“”"  # \n /// no puedo quitarle los " jaajaj text = text.replace('"', "")
    for i in symbols:
        text = text.replace(i, "")
    text=text.replace('"', '')
    # 5- quitar numeros
    text = re.sub("\d+", "", text)
    return text


#2- Cargamos el dataset
df=pd.read_csv('../archivos/csv/concatenate.csv',sep=',')  # no se porq solo va con ruta absoluta

#3- imprimimos las longitudes
print("len total: ",len(df),"\nLen videoconferencia: ",len(df[df.Class=='videoconferencia']),"\nLen cuestionarios: ",len(df[df.Class=='cuestionarios']))

#3- TfIdf Vectorized. Y separamos para train y test
cv = TfidfVectorizer(min_df=1,stop_words='english')
x=df["Text"]

y=df["Class"]
count=0
for t in x:
    x[count]=capar(t)
    count+=1

#4- fit transform
x=cv.fit_transform(x)


# evaluate a model with a given number of repeats
def evaluate_model(x, y, repeats):
    # prepare the cross-validation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model
    model = MultinomialNB(fit_prior=False)
    # evaluate model
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# create dataset
# configurations to test
repeats = range(1, 16)
results = list()
for r in repeats:
    # evaluate using a given number of repeats
    scores = evaluate_model(x, y, r)
    # summarize
    print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
    # store
    results.append(scores)
# plot the results
pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
pyplot.show()