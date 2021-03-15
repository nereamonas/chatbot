#1- imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

#2- Cargamos el dataset
df=pd.read_csv('../../archivos/csvIntermedios/concatenate.csv', sep=',')

#3- Cambiamos la clase videoconferencia a 0 y cuestionario a 1 y imprimimos las longitudes
print("len total: ",len(df),"\nLen videoconferencia: ",len(df[df.Class=='videoconferencia']),"\nLen cuestionarios: ",len(df[df.Class=='cuestionarios']))

#3- TfIdf Vectorized. Y separamos para train y test
cv = TfidfVectorizer(min_df=1,stop_words='english',use_idf=True,sublinear_tf = True,norm='l2')  #tokenizer=tokenizer,
#cv = TfidfVectorizer(max_features=20000, strip_accents='unicode',stop_words='english',analyzer='word', use_idf=True,  ngram_range=(1,2),sublinear_tf= True , norm='l2')

# tfidf = vect.fit_transform(x_train)
# sum norm l2 documents
#vect_sum = tfidf.sum(axis=1)
cTest=df["Text"]
for x in cTest:
    print(x)

x_train, x_test, y_train, y_test = train_test_split(df["Text"], df["Class"], test_size=0.2, random_state=4)

#4- fit transform
x_traincv=cv.fit_transform(x_train)
a=x_traincv.toarray()

#5- inverse transform
cv.inverse_transform(a[0])

x_testcv=cv.transform(x_test)
x_testcv.toarray()

#6- Clasificador
mnb = MultinomialNB()  #0.930693
mnb.fit(x_traincv,y_train)

#Sacamos la predicción
predicion=mnb.predict(x_testcv)
real=np.array(y_test)

#nos dará el threshold de cada clase
threshold=mnb.predict_proba(x_testcv)
#print(threshold)


print(metrics.classification_report(y_test, predicion))

confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


count=0
lenthreshold=0  # los q clasifica con threshold>0.7
print("\n")
for i in range (len(predicion)):
    if max(threshold[i])>0.50:
        lenthreshold=lenthreshold+1
        if predicion[i]==real[i]:
            count=count+1
        else:
            print("Text: ",x_test.iloc[i], "predicion:  ", predicion[i],"  real:   ",real[i], " y ",max(threshold[i]))
print("Bien clasificados: ",count, "\nThreshold > 0.7 ",lenthreshold, "\nPorcenaje correcto: ",count/lenthreshold)

print("Cuantos son sin diferenciar los threshold ",len(predicion))  #no trabajaria 30.


#--------------------------------------------------------------------------------



mnb=MultinomialNB(fit_prior=False)
print("\n\nMULTINOMIALNB fit_prior=False")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)






