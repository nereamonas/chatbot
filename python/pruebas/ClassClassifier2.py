#1- imports
import numpy as np
import pandas as pd
from networkx.drawing.tests.test_pylab import plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, svm
from sklearn.svm import LinearSVC, SVC

#2- Cargamos el dataset
df=pd.read_csv('../../archivos/csvIntermedios/concatenate.csv', sep=',')

#3- imprimimos las longitudes
print("len total: ",len(df),"\nLen videoconferencia: ",len(df[df.Class=='videoconferencia']),"\nLen cuestionarios: ",len(df[df.Class=='cuestionarios']))

#4- TfIdf Vectorized. Y separamos los elementos para train y test (para hacer pruebas y ver q tan bueno seria, aunq realmente sea todo train)
cv = TfidfVectorizer(min_df=1,stop_words='english')
cTest=df["Text"]
x_train, x_test, y_train, y_test = train_test_split(df["Text"], df["Class"], test_size=0.2, random_state=4)

x_train, x_test2, y_train, y_test2 = train_test_split(x_train, y_train, test_size=0.2, random_state=4)


#5- fit transform
x_traincv=cv.fit_transform(x_train)
x_testcv=cv.transform(x_test)
x_test2cv=cv.transform(x_test2)
#6- Clasificador
mnb = MultinomialNB()  #0.930693
print("MULTINOMIALNB")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
print("Resultados de test: ",metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

mnb=MultinomialNB(fit_prior=False)
print("\n\nMULTINOMIALNB fit_prior=False")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

MultinomialNB(alpha = 1.0, class_prior = None, fit_prior = False)
print("\n\nMULTINOMIALNB fit_prior=False ...")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_test2cv)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
print(metrics.classification_report(y_test2, predicion))
confusion_matrix = pd.crosstab(y_test2, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)



mnb=RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0)
print("\n\nRANDOM FOREST")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)



mnb=LinearSVC()
print("\n\nLINEAR SVC")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

mnb=LinearSVC(random_state=0, tol=1e-5)
print("\n\nLINEAR SVC")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
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
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


mnb=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
print("\n\nSGD CLASSIFIER")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


mnb=GridSearchCV(estimator=SVC(),param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')},cv=5, n_jobs=-1)
print("\n\nGRID SEARCH CV")
mnb.fit(x_traincv,y_train)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
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





#----------------------------
"""
print("CROSS")
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, x_traincv, y_train, cv=5)
print(scores)

print("CROSS 2")
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
scores = cross_val_score(clf, x_traincv, y_train, cv=cv)
print(scores)
"""


#https://www.aprendemachinelearning.com/que-es-overfitting-y-underfitting-y-como-solucionarlo/
#https://relopezbriega.github.io/blog/2016/05/29/machine-learning-con-python-sobreajuste/
#https://stackoverflow.com/questions/56207277/am-i-having-an-overfitting-problem-with-my-text-classification
#https://stackoverflow.com/questions/62186526/should-my-model-always-give-100-accuracy-on-training-dataset