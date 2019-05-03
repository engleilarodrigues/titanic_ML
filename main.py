# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# analise detalhada dos dados: gráficos
'''
sns.heatmap(data_train.isnull(), yticklabels=False, cbar=False) #dados ausentes
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=data_train, palette='BrBG') #numero de sobreviventes dividido pelo sexo
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=data_train, palette='rainbow') #numero de sobreviventes dividido por classes
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass', y='Age', data=data_train, palette='winter') #idades de acordo com a classe
'''

def setAge(cols):
#de acordo com analise empirica seta os dados ausentes da coluna idade, de acordo com a media de idade de cada classe.
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

def cleanData(data_train):

    data_train['Age']= data_train[["Age", "Pclass"]].apply(setAge, axis=1) #completa as idades ausentes
    data_train.drop('Cabin', axis=1, inplace=True) #exclui a coluna Cabine

    data_train.dropna(inplace=True) #remove o unico dado ausente que restou

    #convertendo as informações categóricas
    Sex = pd.get_dummies(data_train['Sex'], drop_first=True)  #considera o primeiro dado como verdadeiro, ou seja, agr teremos uma coluna Male com valores 0 ou 1
    Embark = pd.get_dummies(data_train['Embarked'], drop_first=True) #fazemos o mesmo com a coluna de embarque
    data_train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
    data_train['Male'] = Sex
    data_train['Q'] = Embark['Q']
    data_train['S'] = Embark['S']

    print(data_train.head(5))

    return data_train


def main():
    data_train = pd.read_csv('train.csv')
    cleanData(data_train)

    data_test = pd.read_csv('test.csv')
    cleanData(data_test)

    target = data_train['Survived'].values
    features = data_train[["Pclass", "Age", "SibSp", "Parch", "Fare", "Male", "Q", "S"]].values

    features_test = data_test[["Pclass", "Age", "SibSp", "Parch", "Fare", "Male", "Q", "S"]].values

    #criação do modelo e treino: Regressão Logistica
    RL = LogisticRegression()
    RL.fit(features, target)
    scoreRL = round(RL.score(features, target) * 100, 2)

    # criação do modelo e treino: SVM
    svm = SVC()
    svm.fit(features, target)
    scoreSVM = round(svm.score(features, target) * 100, 2)

    # criação do modelo e treino: KNN
    knn = KNeighborsClassifier()
    knn.fit(features, target)
    scoreKNN = round(knn.score(features, target) * 100, 2)


    models = pd.DataFrame({
        'Model': ['Regressão Logística', 'SVM', 'KNN'],
        'Score': [scoreRL, scoreSVM, scoreKNN]
    })
    print(models.sort_values(by='Score', ascending=False))

    # resultados das predições com os dataset de test
    predictions = svm.predict(features_test)
    results = pd.DataFrame({"PassengerId": data_test["PassengerId"], "Survived": predictions})
    print(results.head())
    filename = "results.csv"
    results.to_csv(filename, index=False)

if __name__ == '__main__':
    main()