
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
warnings.filterwarnings('ignore')
print("CUESTIONARIOS")
df=pd.read_csv('../../archivos/csvIntermedios/CuestionariosXTemas_ESCapado.csv', sep=',')

print("len total: ",len(df))
lista = range(100)
df_x=df["Text"].values.astype('U')
df_y=df["Class"]

cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2 ) #random_state=4
#print("Entrenamiento x :  ",x_train)  #coge 404. x-text. y-class
#print("Entrenamiento y :  ",y_train)
#print("Test x :  ",x_test)  #coge 101
#print("Test y :  ",y_test)


cv1 = TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
#print(a[0])

cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()

#6- Clasificador
print("----------------------------------------------------------------------------------")
print('\033[1m', "LINEAR SVC 2", '\033[0m')
accuracy = 0.0
p_final=[]
y_final=[]
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    svc=LinearSVC(random_state=0, tol=1e-5)
    mnb = CalibratedClassifierCV(svc)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    p=mnb.predict_proba(x_testcv)
    #print(p)
    cont = 0
    for i in y_test:
        p_final.append(predicion[cont])
        y_final.append(i)
        cont += 1
    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

m=[[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]
cont=0
for i in y_final:
    m[p_final[cont]][y_final[cont]]= m[p_final[cont]][y_final[cont]]+1
    cont+=1

print("  \t","  \t","0","  \t","1","  \t","2","  \t","3","  \t","4","  \t","5","  \t","6","  \t","7","  \t","8","  \t","9","  \t","10")
print("__________________________________________________________________________________________________")
for i in range(0,11):
    print(i,"|  \t",end="")
    for j in range(0,11):
        print(m[i][j],"  \t",end="")
    print()
print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')
#----------------------


#-----------------------
print("\n\n-----------------------------------------------------------------------")

print("\n\nVIDEOCONFERENCIA")
df=pd.read_csv('../../archivos/csv/VideoconferenciaXTemas_ESCapado.csv', sep=',')

print("len total: ",len(df))

df_x=df["Text"].values.astype('U')
df_y=df["Class"]

cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2) #random_state=4

cv1 = TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
#print(a[0])

cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()

#6- Clasificador
print("----------------------------------------------------------------------------------")
print('\033[1m', "LINEAR SVC", '\033[0m')
accuracy = 0.0
p_final=[]
y_final=[]
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=LinearSVC(random_state=0, tol=1e-5)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)

    cont=0
    for i in y_test:
        p_final.append(predicion[cont])
        y_final.append(i)
        cont+=1
    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)
print("len: p ",len(p_final), "   ",p_final, "  \nlen: y ",len(y_final),"   ",y_final)

m=[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
cont=0
for i in y_final:
    m[p_final[cont]][y_final[cont]]= m[p_final[cont]][y_final[cont]]+1
    cont+=1
print(" \t","  \t","0","  \t","1","  \t","2","  \t","3","  \t","4","  \t","5","  \t","6","  \t","7","  \t","8","  \t","9","  \t","10","  \t","10")
print("__________________________________________________________________________")
for i in range(0,10):
    print(i,"|  \t",end="")
    for j in range(0,10):
        print(m[i][j],"  \t",end="")
    print()

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')



