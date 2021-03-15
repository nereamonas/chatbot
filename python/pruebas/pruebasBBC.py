
import numpy as np
import pandas as pd
from networkx.drawing.tests.test_pylab import plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
#sklearn.metrics.accuracy_score
from sklearn import svm, datasets, metrics
from sklearn.svm import LinearSVC, SVC
import warnings

from actions.Capar import capar

warnings.filterwarnings('ignore')


print("\n\nVIDEOCONFERENCIA")
df=pd.read_csv('../../archivos/csv/VideoconferenciaXTemas_ESCapado.csv', sep=',')
dfPruebas=pd.read_csv('../pruebas/pruebasBBC.csv',sep=',')

print("len total: ",len(dfPruebas))

x_train=df["Text"].values.astype('U')
y_train=df["Class"]
x_test=dfPruebas["Text"].values.astype('U')
cont=0
for x in x_test:
    xx=capar(x)
    x_test[cont]=xx
    cont=cont+1
y_test=dfPruebas["Class"]

cv1 = TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=cv1.fit_transform(x_train)

x_testcv=cv1.transform(x_test)
x_testcv.toarray()

#6- Clasificador
mnb=LinearSVC(random_state=0, tol=1e-5)
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)

print("  accuracy ",metrics.accuracy_score(y_test, predicion))


print(metrics.classification_report(y_test, predicion))

confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


#Sacamos la predicción
predicion=mnb.predict(x_testcv)
real=np.array(y_test)

#nos dará el threshold de cada clase

#print(threshold)

count=0
lenthreshold=0  # los q clasifica con threshold>0.7
print("\n")
c=0
for i in range (len(predicion)):
        lenthreshold=lenthreshold+1
        if predicion[i]==real[i]:
            count=count+1
            print(x_test[c]," predicion:  ", predicion[i], "  real:   ", real[i], " y ")
        else:
            print("XXXXXXXXX ",x_test[c],"predicion:  ", predicion[i],"  real:   ",real[i], " y ")
        c+=1
print("Bien clasificados: ",count, "\nThreshold > 0.7 ",lenthreshold, "\nPorcenaje correcto: ",count/lenthreshold)



