from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, ensemble
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import unittest


class Parser:
    def __init__(self, dataToChange):
        self.dataToChange = dataToChange

    def convertToInt(self, field, values):

        for key in values:
            if field == key:
                return values[key]

    def changeDataToNumbers(self, x):
        for name in self.dataToChange:
            if x.name == name:
                x = x.apply(lambda y: self.convertToInt(y, self.dataToChange[name]))

        return x

    def parse(self, dataFrame):
        return dataFrame.apply(lambda x: self.changeDataToNumbers(x))


class Data:
    def __init__(self, fileName, resultColumn, parser, train):
        self.dataFrame = pd.read_csv(fileName)
        self.parser = parser
        self.resultColumn = resultColumn
        self.train = train

    def parse(self):
        self.dataFrame = self.parser.parse(self.dataFrame)

    def printHead(self):
        print(self.dataFrame.head())

    def showCorrelations(self):
        correlations = self.dataFrame.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        colormap = sns.color_palette("BrBG", 10)
        sns.heatmap(correlations,
                    cmap=colormap,
                    annot=True,
                    fmt=".2f")
        ax.set_yticklabels(self.dataFrame.columns)
        plt.show()

    def fillEmptyData(self):
        data = self.dataFrame.sample(frac=1).reset_index(drop=True)
        data.fillna(data.mean(), inplace=True)
        dataCopy = data.copy()
        self.dataFrame = data

    def stratTrain(self):
        self.train.prepareTrainData(self.dataFrame, self.resultColumn)
        self.train.trainSVM()
        self.train.trainRandomForestClassifier()
        self.pickedModelName = self.train.pickBetterModel()

    def predictForOne(self, collegeEntry):
        data = self.parser.parse(collegeEntry.toDataFrame()).to_numpy()

        with open(self.pickedModelName, 'rb') as handle:
            clf = pickle.load(handle)
        return clf.predict(data)
    
    def showComparesion(self):
        self.train.showComparesion()


class Train:
    def __init__(self, handleName):
        self.handleName = handleName
        self.accuracy_compare = {}

    def getHandleName(self, serialization_name):
        return f"{serialization_name}-{self.handleName}.pickle"

    def prepareTrainData(self, dataFrame, resultColumn):
        self.dataFrame = dataFrame
        X = self.dataFrame.drop(resultColumn, axis=1).to_numpy()
        y = self.dataFrame.loc[:, resultColumn].to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

    def trainModel(self, classifier, serialization_name):
        # trenuj model
        classifier.fit(self.X_train, self.y_train)

        # wygeneruj przewidywania modelu dla zbioru testowego
        predictions = classifier.predict(self.X_test)
        with open(self.getHandleName(serialization_name), 'wb') as handle:
            pickle.dump(classifier, handle)

        # dokonaj ewaluacji modelu na podstawie danych testowych
        scores = list(metrics.precision_recall_fscore_support(predictions, self.y_test))
        score_vals = [scores[0][0], scores[1][0], scores[2][0], metrics.accuracy_score(predictions, self.y_test)]
        return score_vals

    def trainSVM(self):
        name = 'SVM'
        accuracy = self.trainModel(svm.SVC(), name)
        self.accuracy_compare[name] = accuracy

        name = 'SVM-gamma-auto'
        accuracy = self.trainModel(svm.SVC(gamma='auto'), name)
        self.accuracy_compare[name] = accuracy

        name = 'SVM-kernel-sigmoid'
        accuracy = self.trainModel(svm.SVC(kernel='sigmoid'), name)
        self.accuracy_compare[name] = accuracy

        for i in range(5):
            name = 'SVM-degree-' + str(i)
            accuracy = self.trainModel(svm.SVC(degree=i), name)
            self.accuracy_compare[name] = accuracy

    def trainRandomForestClassifier(self):
        name = 'RF'
        accuracy = self.trainModel(ensemble.RandomForestClassifier(), name)
        self.accuracy_compare[name] = accuracy

        for i in range(5):
            name = 'RF-min_samples_split-' + str(i)
            accuracy = self.trainModel(ensemble.RandomForestClassifier(min_samples_split=i*2+2), name)
            self.accuracy_compare[name] = accuracy

        name = 'RF-criterion-entropy'
        accuracy = self.trainModel(ensemble.RandomForestClassifier(criterion='entropy'), name)
        self.accuracy_compare[name] = accuracy

        
        name = 'RF-criterion-log_loss'
        accuracy = self.trainModel(ensemble.RandomForestClassifier(criterion='log_loss'), name)
        self.accuracy_compare[name] = accuracy

    def showComparesion(self):
        df_compare = pd.DataFrame(self.accuracy_compare, index=['precision', 'recall', 'f1 score', 'accuracy'])
        df_compare.plot(kind='bar')
        plt.show()
    
    def pickBetterModel(self):
        bestModel = ''
        for model in self.accuracy_compare:
            if bestModel == '':
                bestModel = model
                continue
            
            if self.accuracy_compare[model][3] > self.accuracy_compare[bestModel][3]:
                bestModel = model
        
        return self.getHandleName(bestModel)

    

class CollegeEntry:
    typeSchool = None
    schoolAccreditation = None
    gender = None
    interest = None
    residence = None
    parentAge = None
    parentSalary = None
    houseArea = None
    averageGrades = None
    parentWasInCollege = None

    def toDataFrame(self):
        return pd.DataFrame.from_dict(self.toDictionary())

    def toDictionary(self):
        return {
            'type_school': [self.typeSchool],
            'school_accreditation': [self.schoolAccreditation],
            'gender': [self.gender],
            'residence': [self.residence],
            'interest': [self.interest],
            'parent_age': [self.parentAge],
            'parent_salary': [self.parentSalary],
            'house_area': [self.houseArea],
            'average_grades': [self.averageGrades],
            'parent_was_in_college': [self.parentWasInCollege]
        }

    def toList(self):
        return [
            self.typeSchool,
            self.schoolAccreditation,
            self.gender,
            self.interest,
            self.residence,
            self.parentAge,
            self.parentSalary,
            self.houseArea,
            self.averageGrades,
            self.parentWasInCollege
        ]

class TestData(unittest.TestCase):
    data = Data('test.csv', 'result', Parser({}), Train('test'))
    def testfillEmptyData(self):
        self.data.dataFrame = pd.DataFrame({
            'a': [2.0,2.0,None,2.0],
            'b': [2.0,None,2.0,None],
            'c': [2.0,2.0,2.0,None]
        })
        convertedData = pd.DataFrame({
            'a': [2.0,2.0,2.0,2.0],
            'b': [2.0,2.0,2.0,2.0],
            'c': [2.0,2.0,2.0,2.0]
        })
        self.data.fillEmptyData()
        
        self.assertTrue(convertedData.equals(self.data.dataFrame))

class TestNormalizer(unittest.TestCase):
    
    parser = Parser({'test': {'a': 0, 'b': 1}})
    dataFrame = Parser({'test': {'a': 0, 'b': 1}})

    def testConvertToInt(self):
        data = {
            'testCase': {'xyz': 0, '123': 1},
        }
        convertedData = self.parser.convertToInt('xyz', data['testCase'])
        convertedData2 = self.parser.convertToInt('123', data['testCase'])

        self.assertEqual(0, convertedData)
        self.assertEqual(1, convertedData2)

    def testParse(self):
        data = pd.DataFrame({
            'test': ['a', 'b']
        })

        dataCompare = pd.DataFrame({
            'test': [0, 1]
        })
        convertedData = self.parser.parse(data)
        
        self.assertTrue(convertedData.equals(dataCompare))

class TestTrain(unittest.TestCase):
    def testPickBetterModel(self):
        name = 'test'
        train = Train(name)
        train.accuracy_compare = {
            'a': [0,0,0,0],
            'b': [0,0,0,1],
        }

        self.assertEqual(f"b-{name}.pickle", train.pickBetterModel())

        
        train.accuracy_compare = {
            'a': [1,2,3,4],
            'b': [5,6,7,8],
            'c': [9,10,11,100],
        }

        self.assertEqual(f"c-{name}.pickle", train.pickBetterModel())

unittest.main(exit=False)

parser = Parser({
    'type_school': {'Academic': 0, 'Vocational': 1},
    'school_accreditation': {'A': 0, 'B': 1},
    'gender': {'Male': 0, 'Female': 1},
    'residence': {'Urban': 0, 'Rural': 1},
    'interest': {
        'Not Interested': 0,
        'Uncertain': 1,
        'Less Interested': 2,
        'Quiet Interested': 3,
        'Very Interested': 4}
})

d = Data('college.csv', 'in_college', parser, Train('rick'))
d.parse()
d.fillEmptyData()
d.printHead()
d.stratTrain()
d.showCorrelations()
d.showComparesion()

ce = CollegeEntry()
ce.typeSchool = 'Academic'
ce.schoolAccreditation = 'A'
ce.gender = 'Male'
ce.interest = 'Not Interested'
ce.residence = 'Urban'
ce.parentAge = 49
ce.parentSalary = 4790000
ce.houseArea = 80.2
ce.averageGrades = 67.53
ce.parentWasInCollege = True

if d.predictForOne(ce):
    print('It is LIKELY that this person will go to College')
else:
    print('It is UNLIKELY that this person will go to College')
