
import numpy as np
import pandas as pd
from networkx.drawing.tests.test_pylab import plt
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
df=pd.read_csv('../archivos/csv/CuestionariosXTemas_ES.csv',sep=',')

print("len total: ",len(df))

df_x=df["Text"]
df_y=df["Class"]

cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
#print("Entrenamiento x :  ",x_train)  #coge 404. x-text. y-class
#print("Entrenamiento y :  ",y_train)
#print("Test x :  ",x_test)  #coge 101
#print("Test y :  ",y_test)


cv1 = TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
#print(a[0])

cv1.inverse_transform(a[0])

x_train.iloc[0]
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


count=0
lenthreshold=0  # los q clasifica con threshold>0.7
print("\n")
for i in range (len(predicion)):
    if max(threshold[i])>0.10:
        lenthreshold=lenthreshold+1
        if predicion[i]==real[i]:
            count=count+1
        else:
            print("Text: ",x_test.iloc[i], "predicion:  ", predicion[i],"  real:   ",real[i], " y ",max(threshold[i]))
print("Bien clasificados: ",count, "\nThreshold > 0.1 ",lenthreshold )

print("Cuantos son sin diferenciar los threshold ",len(predicion))  #no trabajaria 30.


#6- Clasificador
mnb = MultinomialNB()  #0.930693
print("MULTINOMIALNB")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


mnb=MultinomialNB(fit_prior=False)
print("\n\nMULTINOMIALNB fit_prior=False")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

MultinomialNB(alpha = 1.0, class_prior = None, fit_prior = False)
print("\n\nMULTINOMIALNB fit_prior=False ...")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


mnb=RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0)
print("\n\nRANDOM FOREST")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)



mnb=LinearSVC()
print("\n\nLINEAR SVC")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

mnb=LinearSVC(random_state=0, tol=1e-5)
print("\n\nLINEAR SVC")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

print("\n\nLINEAR SVC 2" )
clf_sets = [(LinearSVC(penalty='l1', loss='squared_hinge', dual=False,
                       tol=1e-3),
             np.logspace(-2.3, -1.3, 10)),
            (LinearSVC(penalty='l2', loss='squared_hinge', dual=True),np.logspace(-4.5, -2, 10))]
colors = ['navy', 'cyan', 'darkorange']
lw = 2
for clf, cs, in clf_sets:
    # set up the plot for each regressor
    for k, train_size in enumerate(np.linspace(0.3, 0.7, 3)[::-1]):
        fig, axes = plt.subplots(nrows=2, sharey=True, figsize=(9, 10))
        param_grid = dict(C=cs)
        # To get nice curve, we need a large number of iterations to
        # reduce the variance
        grid = GridSearchCV(clf, refit=False, param_grid=param_grid,
                            cv=ShuffleSplit(train_size=train_size,
                                            test_size=.3,
                                            n_splits=250, random_state=1))
        grid.fit(x_traincv,y_train)
        scores = grid.cv_results_['mean_test_score']
        print(scores)


#----------------------

mnb=LogisticRegression(random_state=0)
print("\n\nLogistic Regression")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


mnb=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
print("\n\nSGD CLASSIFIER")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

"""
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}
GridSearchCV(text_clf, parameters, n_jobs=-1)
GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf.best_score_
gs_clf.best_params_
"""
mnb=GridSearchCV(estimator=SVC(),param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')},cv=5, n_jobs=-1)
print("\n\nGRID SEARCH CV")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)



print("\n\nGridSearchCV con mas variables")
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(SVC(), tuned_parameters)
clf.fit(x_traincv, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
print("Detailed classification report:")
y_pred = clf.predict(x_testcv)
print(metrics.classification_report(y_test, y_pred))
print()





#-----------------------
print("\n\n-----------------------------------------------------------------------")

print("\n\nVIDEOCONFERENCIA")
df=pd.read_csv('../archivos/csv/VideoconferenciaXTemas_ES.csv',sep=',')

print("len total: ",len(df))

df_x=df["Text"]
df_y=df["Class"]

cv = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
#print("Entrenamiento x :  ",x_train)  #coge 404. x-text. y-class
#print("Entrenamiento y :  ",y_train)
#print("Test x :  ",x_test)  #coge 101
#print("Test y :  ",y_test)


cv1 = TfidfVectorizer(min_df=1,stop_words='english')
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
#print(a[0])

cv1.inverse_transform(a[0])

x_train.iloc[0]
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


count=0
lenthreshold=0  # los q clasifica con threshold>0.7
print("\n")
for i in range (len(predicion)):
    if max(threshold[i])>0.10:
        lenthreshold=lenthreshold+1
        if predicion[i]==real[i]:
            count=count+1
        else:
            print("Text: ",x_test.iloc[i], "predicion:  ", predicion[i],"  real:   ",real[i], " y ",max(threshold[i]))
print("Bien clasificados: ",count, "\nThreshold > 0.1 ",lenthreshold )

print("Cuantos son sin diferenciar los threshold ",len(predicion))  #no trabajaria 30.


#6- Clasificador
mnb = MultinomialNB()  #0.930693
print("MULTINOMIALNB")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


mnb=MultinomialNB(fit_prior=False)
print("\n\nMULTINOMIALNB fit_prior=False")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

MultinomialNB(alpha = 1.0, class_prior = None, fit_prior = False)
print("\n\nMULTINOMIALNB fit_prior=False ...")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


mnb=RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0)
print("\n\nRANDOM FOREST")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)



mnb=LinearSVC()
print("\n\nLINEAR SVC")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

mnb=LinearSVC(random_state=0, tol=1e-5)
print("\n\nLINEAR SVC")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

print("\n\nLINEAR SVC 2" )
clf_sets = [(LinearSVC(penalty='l1', loss='squared_hinge', dual=False,
                       tol=1e-3),
             np.logspace(-2.3, -1.3, 10)),
            (LinearSVC(penalty='l2', loss='squared_hinge', dual=True),np.logspace(-4.5, -2, 10))]
colors = ['navy', 'cyan', 'darkorange']
lw = 2
for clf, cs, in clf_sets:
    # set up the plot for each regressor
    for k, train_size in enumerate(np.linspace(0.3, 0.7, 3)[::-1]):
        fig, axes = plt.subplots(nrows=2, sharey=True, figsize=(9, 10))
        param_grid = dict(C=cs)
        # To get nice curve, we need a large number of iterations to
        # reduce the variance
        grid = GridSearchCV(clf, refit=False, param_grid=param_grid,
                            cv=ShuffleSplit(train_size=train_size,
                                            test_size=.3,
                                            n_splits=250, random_state=1))
        grid.fit(x_traincv,y_train)
        scores = grid.cv_results_['mean_test_score']
        print(scores)











#----------------------

mnb=LogisticRegression(random_state=0)
print("\n\nLogistic Regression")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


mnb=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
print("\n\nSGD CLASSIFIER")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

"""
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}
GridSearchCV(text_clf, parameters, n_jobs=-1)
GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf.best_score_
gs_clf.best_params_
"""
mnb=GridSearchCV(estimator=SVC(),param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')},cv=5, n_jobs=-1)
print("\n\nGRID SEARCH CV")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)



print("\n\nGridSearchCV con mas variables")
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(SVC(), tuned_parameters)
clf.fit(x_traincv, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
print("Detailed classification report:")
y_pred = clf.predict(x_testcv)
print(metrics.classification_report(y_test, y_pred))
print()



