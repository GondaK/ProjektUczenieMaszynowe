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
        self.train.trainSVC()
        self.train.trainRandomForestClassifier()

    def predictForOne(self, collegeEntry):
        data = self.parser.parse(collegeEntry.toDataFrame()).to_numpy()

        with open(self.train.getHandleName(), 'rb') as handle:
            clf = pickle.load(handle)
        return clf.predict(data)


class Train:
    def __init__(self, handleName):
        self.handleName = handleName
        self.accuracy_compare = {}

    def getHandleName(self):
        return f"{self.handleName}.pickle"

    def prepareTrainData(self, dataFrame, resultColumn):
        self.dataFrame = dataFrame
        X = self.dataFrame.drop(resultColumn, axis=1).to_numpy()
        y = self.dataFrame.loc[:, resultColumn].to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

    def trainModel(self, classifier, feature_vector_train, label, feature_vector_valid):
        # trenuj model
        classifier.fit(feature_vector_train, label)

        # wygeneruj przewidywania modelu dla zbioru testowego
        predictions = classifier.predict(feature_vector_valid)
        with open(self.getHandleName(), 'wb') as handle:
            pickle.dump(classifier, handle)

        # dokonaj ewaluacji modelu na podstawie danych testowych
        scores = list(metrics.precision_recall_fscore_support(predictions, self.y_test))
        score_vals = [scores[0][0], scores[1][0], scores[2][0], metrics.accuracy_score(predictions, self.y_test)]
        return score_vals

    def trainSVC(self):
        accuracy = self.trainModel(svm.SVC(), self.X_train, self.y_train, self.X_test)
        self.accuracy_compare['SVM'] = accuracy

    def trainRandomForestClassifier(self):
        accuracy = self.trainModel(ensemble.RandomForestClassifier(), self.X_train, self.y_train, self.X_test)
        self.accuracy_compare['RF'] = accuracy

    def showComparesion(self):
        df_compare = pd.DataFrame(self.accuracy_compare, index=['precision', 'recall', 'f1 score', 'accuracy'])
        df_compare.plot(kind='bar')
        plt.show()



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


class TestNormalizer(unittest.TestCase):
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

    def testConvertToInt(self):
        data = {
            'testCase': {'xyz': 0, '123': 1},
        }
        convertedData = parser.convertToInt('xyz', data['testCase'])
        convertedData2 = parser.convertToInt('123', data['testCase'])

        self.assertEqual(0, convertedData)
        self.assertEqual(1, convertedData2)

    def testTraining(self):
        testParser = Parser({
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

        testD = Data('college.csv', 'in_college', testParser, Train('test'))
        testD.parse()
        testD.fillEmptyData()
        testD.stratTrain()
        testCe = CollegeEntry()
        testCe.typeSchool = 'Academic'
        testCe.schoolAccreditation = 'A'
        testCe.gender = 'Male'
        testCe.interest = 'Not Interested'
        testCe.residence = 'Urban'
        testCe.parentAge = 49
        testCe.parentSalary = 4790000
        testCe.houseArea = 80.2
        testCe.averageGrades = 67.53
        testCe.parentWasInCollege = True
        self.assertEqual(True, testD.predictForOne(testCe))


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

unittest.main(exit=False)

d = Data('college.csv', 'in_college', parser, Train('rick'))
d.parse()
d.fillEmptyData()
d.printHead()
d.stratTrain()



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
