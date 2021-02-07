
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
df=pd.read_csv('../archivos/csv/CuestionariosXTemas_ESCapado.csv',sep=',')

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

mnb=MultinomialNB(fit_prior=False)
mnb.fit(x_traincv,y_train)

#Sacamos la predicción
predicion=mnb.predict(x_testcv)
real=np.array(y_test)

#nos dará el threshold de cada clase
threshold=mnb.predict_proba(x_testcv)
#print(threshold)


predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))

confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)



#6- Clasificador
print("----------------------------------------------------------------------------------")
print('\033[1m', "MULTINOMIALNB", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb = MultinomialNB()  #0.930693
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

print("----------------------------------------------------------------------------------")
print('\033[1m', "MULTINOMIALNB fit_prior=False", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=MultinomialNB(fit_prior=False)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

print("----------------------------------------------------------------------------------")
print('\033[1m', "RANDOM FOREST", '\033[0m')
accuracy = 0.0
for x in list(lista):

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')


print("----------------------------------------------------------------------------------")
print('\033[1m', "LINEAR SVC", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=LinearSVC()
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

print("----------------------------------------------------------------------------------")
print('\033[1m', "LINEAR SVC 2", '\033[0m')
accuracy = 0.0
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
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')
#----------------------

print("----------------------------------------------------------------------------------")
print('\033[1m', "Logistic Regression", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=LogisticRegression(random_state=0)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

print("----------------------------------------------------------------------------------")
print('\033[1m', "MULTINOMIALNB", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

print("----------------------------------------------------------------------------------")
print('\033[1m', "GRID SEARCH CV", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=GridSearchCV(estimator=SVC(),param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')},cv=5, n_jobs=-1)

    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')


print("----------------------------------------------------------------------------------")
print('\033[1m', "GridSearchCV con mas variables", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)

    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), tuned_parameters)
    clf.fit(x_traincv, y_train)
    y_pred = clf.predict(x_testcv)

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, y_pred))
    accuracy += metrics.accuracy_score(y_test, y_pred)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')




#-----------------------
print("\n\n-----------------------------------------------------------------------")

print("\n\nVIDEOCONFERENCIA")
df=pd.read_csv('../archivos/csv/VideoconferenciaXTemas_ESCapado.csv',sep=',')

print("len total: ",len(df))

df_x=df["Text"].values.astype('U')
df_y=df["Class"]

cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2) #random_state=4
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

mnb=MultinomialNB(fit_prior=False)
mnb.fit(x_traincv,y_train)

#Sacamos la predicción
predicion=mnb.predict(x_testcv)
real=np.array(y_test)

#nos dará el threshold de cada clase
threshold=mnb.predict_proba(x_testcv)
#print(threshold)


predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))

confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


#6- Clasificador
print("----------------------------------------------------------------------------------")
print('\033[1m', "MULTINOMIALNB", '\033[0m')
accuracy = 0.0
for x in list(lista):

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb = MultinomialNB()
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

print("----------------------------------------------------------------------------------")
print('\033[1m', "MULTINOMIALNB fit_prior=False", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=MultinomialNB(fit_prior=False)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

print("----------------------------------------------------------------------------------")
print('\033[1m', "RANDOM FOREST", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

print("----------------------------------------------------------------------------------")
print('\033[1m', "LINEAR SVC", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=LinearSVC()
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

print("----------------------------------------------------------------------------------")
print('\033[1m', "LINEAR SVC", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=LinearSVC(random_state=0, tol=1e-5)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')






#----------------------
print("----------------------------------------------------------------------------------")
print('\033[1m', "Logistic Regression", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=LogisticRegression(random_state=0)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

print("----------------------------------------------------------------------------------")
print('\033[1m', "SGD CLASSIFIER", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')

"""
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}
GridSearchCV(text_clf, parameters, n_jobs=-1)
GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf.best_score_
gs_clf.best_params_
"""
print("----------------------------------------------------------------------------------")
print('\033[1m', "GRID SEARCH CV", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    mnb=GridSearchCV(estimator=SVC(),param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')},cv=5, n_jobs=-1)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_testcv)
    confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, predicion))
    accuracy += metrics.accuracy_score(y_test, predicion)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')


print("----------------------------------------------------------------------------------")
print('\033[1m', "GridSearchCV con mas variables", '\033[0m')
accuracy = 0.0
for x in list(lista):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)  # random_state=4
    x_traincv = cv1.fit_transform(x_train)
    x_testcv = cv1.transform(x_test)
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), tuned_parameters)
    clf.fit(x_traincv, y_train)

    y_pred = clf.predict(x_testcv)

    print("iteracion:", x,"  accuracy ", metrics.accuracy_score(y_test, y_pred))
    accuracy += metrics.accuracy_score(y_test, y_pred)

print('\033[1m', "ACCURACY MEDIO: ", accuracy / 100, '\033[0m')


