import numpy as np
import pandas as pd
from networkx.drawing.tests.test_pylab import plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, svm
from sklearn.svm import LinearSVC, SVC

#2- Cargamos el dataset
#df=pd.read_csv('../archivos/csvIntermedios/concatenate.csv',sep=',')
df=pd.read_csv('../../archivos/csvIntermedios/concatenateCapado.csv', sep=',')

#3- imprimimos las longitudes
print("len total: ",len(df),"\nLen videoconferencia: ",len(df[df.Class=='videoconferencia']),"\nLen cuestionarios: ",len(df[df.Class=='cuestionarios']))

#4- TfIdf Vectorized. Y separamos los elementos para train y test (para hacer pruebas y ver q tan bueno seria, aunq realmente sea todo train)
cv = TfidfVectorizer(min_df=1,stop_words='english')

xtrain, x_test, ytrain, y_test = train_test_split(df["Text"].values.astype('U'), df["Class"], test_size=0.2) #, random_state=1

print("tamaños: train: 301 dev: 101  test: 101 ")
#5- fit transform


#6- Clasificador

#-------------------------------------------------
#----------------------------------------------------
lista = range(100)

print("----------------------------------------------------------------------------------")
print('\033[1m',"MULTINOMIALNB",'\033[0m')
accuracy=0.0
for x in list(lista):
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2
    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb = MultinomialNB()  #0.930693

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:",x, "  accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)

print('\033[1m',"ACCURACY MEDIO: ",accuracy/100,'\033[0m')

accuracy=0.0

print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"MULTINOMIALNB fit_prior=False",'\033[0m')

for x in list(lista):
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=MultinomialNB(fit_prior=False)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:",x, "  accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)

print('\033[1m',"ACCURACY MEDIO: ",accuracy/100,'\033[0m')
accuracy=0.0

print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"RANDOM FOREST",'\033[0m')

for x in list(lista):
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:",x, "  accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)

print('\033[1m',"ACCURACY MEDIO: ",accuracy/100,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"LINEAR SVC",'\033[0m')

for x in list(lista):
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=LinearSVC()
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:",x, "  accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)

print('\033[1m',"ACCURACY MEDIO: ",accuracy/100,'\033[0m')
accuracy=0.0

print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"LINEAR SVC 2",'\033[0m')

for x in list(lista):
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=LinearSVC(random_state=0, tol=1e-5)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)


    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:",x, "  accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)


print('\033[1m',"ACCURACY MEDIO: ",accuracy/100,'\033[0m')
accuracy=0.0



print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"Logistic Regression",'\033[0m')

for x in list(lista):
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=LogisticRegression(random_state=0)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)

    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:",x, "  accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)

print('\033[1m',"ACCURACY MEDIO: ",accuracy/100,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"SGD CLASSIFIER",'\033[0m')

for x in list(lista):
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)

    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:",x, "  accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)

print('\033[1m',"ACCURACY MEDIO: ",accuracy/100,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"GRID SEARCH CV",'\033[0m')

for x in list(lista):
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=GridSearchCV(estimator=SVC(),param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')},cv=5, n_jobs=-1)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)

    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:",x, "  accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)

print('\033[1m',"ACCURACY MEDIO: ",accuracy/100,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"GridSearchCV con mas variables",'\033[0m')

for x in list(lista):
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), tuned_parameters)
    clf.fit(x_traincv, y_train)
    y_pred = clf.predict(x_devcv)

    print("iteracion:",x, "  accuracy ",metrics.accuracy_score(y_dev, y_pred))
    accuracy+=metrics.accuracy_score(y_dev, y_pred)

print('\033[1m',"ACCURACY MEDIO: ",accuracy/100,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"Ahora aplicamos sobre el dev",'\033[0m')
print("Sacaremos el threshold y la clase")

# 5- fit transform

xtraincv = cv.fit_transform(xtrain)
x_testcv = cv.transform(x_test)
svc=LinearSVC(random_state=0, tol=1e-5)
mnb = CalibratedClassifierCV(svc)
mnb.fit(xtraincv,ytrain)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
p=mnb.predict_proba(x_testcv)
for x in range(101):
    print("Threshold: ",p[x]," Clase: ",predicion[x]);
#print(p)

#print(mnb.predict_proba(x_devcv))
#print(predicion)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_testcv, y_test)))
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

print('\033[1m',"ACCURACY: ",metrics.accuracy_score(y_test, predicion),'\033[0m')




#https://www.aprendemachinelearning.com/que-es-overfitting-y-underfitting-y-como-solucionarlo/
#https://relopezbriega.github.io/blog/2016/05/29/machine-learning-con-python-sobreajuste/
#https://stackoverflow.com/questions/56207277/am-i-having-an-overfitting-problem-with-my-text-classification
#https://stackoverflow.com/questions/62186526/should-my-model-always-give-100-accuracy-on-training-dataset