from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, ensemble
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('college.csv')
data.head()

print(data.describe())
print(data.info())
print(data.isnull().values.any())

def setTypeSchool(field):
    if field == 'Academic':
        return 1
    if field == 'Vocational':
        return 0
    return field

def setSchoolAccreditation(field):
    if field == 'A':
        return 1
    if field == 'B':
        return 0
    return field

def setGender(field):
    if field == 'Male':
        return 1
    if field == 'Female':
        return 0
    return field

def setInterest(field):
    if field == 'Very Interested':
        return 4
    if field == 'Quiet Interested':
        return 3
    if field == 'Less Interested':
        return 2
    if field == 'Uncertain':
        return 1
    if field == 'Not Interested':
        return 0
    
    return field

def setResidence(field):
    if field == 'Urban':
        return 1
    if field == 'Rural':
        return 0
    return field

def changeDataToNumbers(x):
    if x.name == 'type_school':
        x = x.apply(lambda y: setTypeSchool(y))
        
    if x.name == 'school_accreditation':
        x = x.apply(lambda y: setSchoolAccreditation(y))
        
    if x.name == 'gender':
        x = x.apply(lambda y: setGender(y))
        
    if x.name == 'interest':
        x = x.apply(lambda y: setInterest(y))
        
    if x.name == 'residence':
        x = x.apply(lambda y: setResidence(y))
        
    return x

# data = data.apply(lambda x: setTarget(x))
data = data.apply(lambda x: changeDataToNumbers(x))
data


correlations = data.corr()

fig, ax = plt.subplots(figsize=(10, 10))

colormap = sns.color_palette("BrBG", 10)

sns.heatmap(correlations, 
    cmap=colormap, 
    annot=True, 
    fmt=".2f")
ax.set_yticklabels(data.columns)

plt.show()

data = data.sample(frac=1).reset_index(drop=True)
data.fillna(data.mean(), inplace=True)
dataCopy = data.copy()
data


dataB = data.sort_values('in_college', ascending=False)
# df = dataB.loc[data['is_safe'] == 1, 'bacteria']
# df
dataB


# zbiór danych
#data['age'] = data['age'] / 365
X = data.drop('in_college', axis=1).to_numpy()
X

y = data.loc[:, 'in_college'].to_numpy()
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345) 


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # trenuj model
    classifier.fit(feature_vector_train, label)
    
    # wygeneruj przewidywania modelu dla zbioru testowego
    predictions = classifier.predict(feature_vector_valid)
    with open('rick.pickle', 'wb') as handle:
        pickle.dump(classifier, handle)
    
    # dokonaj ewaluacji modelu na podstawie danych testowych
    scores = list(metrics.precision_recall_fscore_support(predictions, y_test))
    score_vals = [
        scores[0][0],
        scores[1][0],
        scores[2][0]
    ]
    score_vals.append(metrics.accuracy_score(predictions, y_test))
    return score_vals

    # MODEL 1 - regresja logistyczna 
accuracy = train_model(linear_model.LogisticRegression(), X_train, y_train, X_test)
accuracy_compare = {'LR': accuracy}
print ("LR: ", accuracy)

# MODEL 2 - Support Vector Machine
accuracy = train_model(svm.SVC(), X_train, y_train, X_test)
accuracy_compare['SVM'] = accuracy
print ("SVM" , accuracy)

# MODEL 3 - Random Forest Tree 
accuracy = train_model(ensemble.RandomForestClassifier(), X_train, y_train, X_test)
accuracy_compare['RF'] =  accuracy
print ("RF: ", accuracy)



df_compare = pd.DataFrame(accuracy_compare, index = ['precision', 'recall', 'f1 score', 'accuracy'])
df_compare.plot(kind='bar')

with open('rick.pickle', 'rb') as handle:
    clf = pickle.load(handle)

clf.predict([
        [1,0,0,1,0,49,4790000,80.2,67.53,True]
    ]) 
