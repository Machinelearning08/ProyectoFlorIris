############################ProyectoFlorIris########################



######################Datos##########################
import pandas as pd

ruta = '#rutadedescarga'
data = pd.read_csv(ruta)


print(data.head())
data = data.drop('Id', axis=1)
print(data.describe())
print(data.groupby('Species').size())


##################### Visualizacion de los Datos #####################
import matplotlib.pyplot as pyplot
import seaborn as sns

data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# #Histograma

data.hist()
pyplot.show()

fig = data[data.Species == 'Iris-setosa'].plot(kind='scatter',
            x='SepalLengthCm', y='SepalWidthCm', color='Blue', label='setosa')
data[data.Species == 'Iris-setosa'].plot(kind='scatter',
            x='SepalLengthCm', y='SepalWidthCm', color='Green', label='versicolor', ax=fig)
data[data.Species == 'Iris-setosa'].plot(kind='scatter',
            x='SepalLengthCm', y='SepalWidthCm', color='Red', label='virginica', ax=fig)

fig.set_xlabel('Sepalo-Longitud')
fig.set_ylabel('Sepalo-Ancho')
fig.set_title('Sepalo-Longitud vs Ancho')
pyplot.show()


fig = data[data.Species == 'Iris-setosa'].plot(kind='scatter',
            x='PetalLengthCm', y='PetalWidthCm', color='Blue', label='setosa')
data[data.Species == 'Iris-setosa'].plot(kind='scatter',
            x='PetalLengthCm', y='PetalWidthCm', color='Green', label='versicolor', ax=fig)
data[data.Species == 'Iris-setosa'].plot(kind='scatter',
            x='PetalLengthCm', y='PetalWidthCm', color='Red', label='virginica', ax=fig)

fig.set_xlabel('Petalo-Longitud')
fig.set_ylabel('Petalo-Ancho')
fig.set_title('Petalo-Longitud vs Ancho')
pyplot.show()


# sns.pairplot(data, hue=('Species'))


#Algoritmos de clasificacion Machine Learning
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#proceso de preparación de datos para un modelo de aprendizaje automático
array = data.values
X = array[:,0:4]
y = array[:,4]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)



################### Muestra de Resultados por Separado ###################


# resultados = []


# # #Modelo de Regresion Logica


# algoritmo = LogisticRegression()
# algoritmo.fit(X_train, Y_train)
# Y_pred = algoritmo.predict(X_test)
# score_log = accuracy_score(Y_test, Y_pred)
# resultados.append(('LR',score_log))

# #Linear Discriminant Analysis


# algoritmo = LinearDiscriminantAnalysis()
# algoritmo.fit(X_train, Y_train)
# Y_pred = algoritmo.predict(X_test)
# score_LDA = accuracy_score(Y_test, Y_pred)
# resultados.append(('LDA',score_LDA))

# # # Modelo Kneighborns


# algoritmo = KNeighborsClassifier()
# algoritmo.fit(X_train, Y_train)
# Y_pred = algoritmo.predict(X_test)
# score_kn = accuracy_score(Y_test, Y_pred)
# resultados.append(('KNN',score_kn))


# # # Modelo Arboles de decision


# algoritmo = DecisionTreeClassifier()
# algoritmo.fit(X_train, Y_train)
# Y_pred = algoritmo.predict(X_test)
# score_dt = accuracy_score(Y_test, Y_pred)
# resultados.append(('CART',score_dt))

# #Gaussian Naive Bayes 


# algoritmo = GaussianNB()
# algoritmo.fit(X_train, Y_train)
# Y_pred = algoritmo.predict(X_test)
# score_NB = accuracy_score(Y_test, Y_pred)
# resultados.append(('NB',score_NB))


# # # Modelo de Maquinas de Vectores de Soporte

# algoritmo = SVC()
# algoritmo.fit(X_train, Y_train)
# Y_pred = algoritmo.predict(X_test)
# score_svc = accuracy_score(Y_test, Y_pred)
# resultados.append(('SVM',score_svc))


# df = pd.DataFrame(resultados, columns=['algorit', 'Precision'])
# print(df)


################### Muestra de resultados Juntos ####################


modelo = []
modelo.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
modelo.append(('LDA', LinearDiscriminantAnalysis()))
modelo.append(('KNN', KNeighborsClassifier()))
modelo.append(('CART', DecisionTreeClassifier()))
modelo.append(('NB', GaussianNB()))
modelo.append(('SVM', SVC()))

resultado = []
nombres = []
for nombre, modelo in modelo:
  kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
  cv_resultados = cross_val_score(modelo, X_train, Y_train, cv=kfold, scoring='accuracy')
  resultado.append(cv_resultados)
  nombres.append(nombre)
  print('%s: %f (%f)' % (nombre, cv_resultados.mean(), cv_resultados.std()))

################Comparando Algoritmos##################


pyplot.boxplot(resultado, labels=nombres)
pyplot.title('Compracion de algoritmos')
pyplot.show()


