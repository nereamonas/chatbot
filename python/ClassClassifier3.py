
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
df=pd.read_csv('../archivos/csv/concatenate.csv',sep=',')

#3- imprimimos las longitudes
print("len total: ",len(df),"\nLen videoconferencia: ",len(df[df.Class=='videoconferencia']),"\nLen cuestionarios: ",len(df[df.Class=='cuestionarios']))

#4- TfIdf Vectorized. Y separamos los elementos para train y test (para hacer pruebas y ver q tan bueno seria, aunq realmente sea todo train)
cv = TfidfVectorizer(min_df=1,stop_words='english')

xtrain, x_test, ytrain, y_test = train_test_split(df["Text"], df["Class"], test_size=0.2) #, random_state=1

print("tamaños: train: 301 dev: 101  test: 101 ")

#5- fit transform


#6- Clasificador

#-------------------------------------------------
#----------------------------------------------------

print("----------------------------------------------------------------------------------")
print('\033[1m',"MULTINOMIALNB",'\033[0m')
accuracy=0.0
for x in [0,1,2,3,4,5,6,7,8,9]:
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2
    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb = MultinomialNB()  #0.930693
    print("\niteracion:",x)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
    print("Resultados de test: ",metrics.classification_report(y_dev, predicion))
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)
    print("accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)
    print("\n")
print('\033[1m',"ACCURACY MEDIO: ",accuracy/10,'\033[0m')

accuracy=0.0

print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"MULTINOMIALNB fit_prior=False",'\033[0m')

for x in [0,1,2,3,4,5,6,7,8,9]:
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=MultinomialNB(fit_prior=False)
    print("\niteracion:",x)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
    print(metrics.classification_report(y_dev, predicion))
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)
    print("accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)
    print("\n")
print('\033[1m',"ACCURACY MEDIO: ",accuracy/10,'\033[0m')
accuracy=0.0

print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"RANDOM FOREST",'\033[0m')

for x in [0,1,2,3,4,5,6,7,8,9]:
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0)
    print("\niteracion:",x)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
    print(metrics.classification_report(y_dev, predicion))
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)
    print("accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)
    print("\n")
print('\033[1m',"ACCURACY MEDIO: ",accuracy/10,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"LINEAR SVC",'\033[0m')

for x in [0,1,2,3,4,5,6,7,8,9]:
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=LinearSVC()
    print("\niteracion:",x)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
    print(metrics.classification_report(y_dev, predicion))
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)
    print("accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)
    print("\n")
print('\033[1m',"ACCURACY MEDIO: ",accuracy/10,'\033[0m')
accuracy=0.0

print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"LINEAR SVC 2",'\033[0m')

for x in [0,1,2,3,4,5,6,7,8,9]:
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=LinearSVC(random_state=0, tol=1e-5)
    print("\niteracion:",x)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    mnb.decision_function(x_devcv)
    print(mnb)
    print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
    print(metrics.classification_report(y_dev, predicion))
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)
    print("accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)
    print("\n")

print('\033[1m',"ACCURACY MEDIO: ",accuracy/10,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"LINEAR SVC 3",'\033[0m')

for x in [0,1,2,3,4,5,6,7,8,9]:
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    print("\niteracion:",x)
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
    print("accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)
    print("\n")
print('\033[1m',"ACCURACY MEDIO: ",accuracy/10,'\033[0m')
accuracy=0.0




print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"Logistic Regression",'\033[0m')

for x in [0,1,2,3,4,5,6,7,8,9]:
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=LogisticRegression(random_state=0)
    print("\niteracion:",x)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
    print(metrics.classification_report(y_dev, predicion))
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)
    print("accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)
    print("\n")
print('\033[1m',"ACCURACY MEDIO: ",accuracy/10,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"SGD CLASSIFIER",'\033[0m')

for x in [0,1,2,3,4,5,6,7,8,9]:
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
    print("\niteracion:",x)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
    print(metrics.classification_report(y_dev, predicion))
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)
    print("accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)
    print("\n")
print('\033[1m',"ACCURACY MEDIO: ",accuracy/10,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"GRID SEARCH CV",'\033[0m')

for x in [0,1,2,3,4,5,6,7,8,9]:
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    mnb=GridSearchCV(estimator=SVC(),param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')},cv=5, n_jobs=-1)
    print("\niteracion:",x)
    mnb.fit(x_traincv,y_train)
    #Sacamos la predicción
    predicion=mnb.predict(x_devcv)
    print("precisión entranamiento: {0: .2f}".format(mnb.score(x_traincv, y_train)))
    print(metrics.classification_report(y_dev, predicion))
    confusion_matrix = pd.crosstab(y_dev, predicion, rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)
    print("accuracy ",metrics.accuracy_score(y_dev, predicion))
    accuracy+=metrics.accuracy_score(y_dev, predicion)
    print("\n")
print('\033[1m',"ACCURACY MEDIO: ",accuracy/10,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"GridSearchCV con mas variables",'\033[0m')

for x in [0,1,2,3,4,5,6,7,8,9]:
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.25)  # 0.25 x 0.8 = 0.2

    # 5- fit transform
    x_traincv = cv.fit_transform(x_train)
    x_devcv = cv.transform(x_dev)
    print("\niteracion:",x)
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
    y_pred = clf.predict(x_devcv)
    print(metrics.classification_report(y_dev, y_pred))
    print()
    print("accuracy ",metrics.accuracy_score(y_dev, y_pred))
    accuracy+=metrics.accuracy_score(y_dev, y_pred)
    print("\n")
print('\033[1m',"ACCURACY MEDIO: ",accuracy/10,'\033[0m')
accuracy=0.0


print("\n\n\n----------------------------------------------------------------------------------")
print('\033[1m',"Ahora aplicamos sobre el dev",'\033[0m')

# 5- fit transform

xtraincv = cv.fit_transform(xtrain)
x_testcv = cv.transform(x_test)
mnb=MultinomialNB(fit_prior=False)
mnb.fit(xtraincv,ytrain)
#Sacamos la predicción
predicion=mnb.predict(x_testcv)
#print(mnb.predict_proba(x_devcv))
print(predicion)
print("precisión entranamiento: {0: .2f}".format(mnb.score(x_testcv, y_test)))
print(metrics.classification_report(y_test, predicion))
confusion_matrix = pd.crosstab(y_test, predicion, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

print('\033[1m',"ACCURACY MEDIO: ",metrics.accuracy_score(y_test, predicion),'\033[0m')




#https://www.aprendemachinelearning.com/que-es-overfitting-y-underfitting-y-como-solucionarlo/
#https://relopezbriega.github.io/blog/2016/05/29/machine-learning-con-python-sobreajuste/
#https://stackoverflow.com/questions/56207277/am-i-having-an-overfitting-problem-with-my-text-classification
#https://stackoverflow.com/questions/62186526/should-my-model-always-give-100-accuracy-on-training-dataset