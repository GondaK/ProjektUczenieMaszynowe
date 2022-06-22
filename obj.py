from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, ensemble
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

class Parser :


class Data:
    def __init__(self, fileName):
        self.dataFrame = pd.read_csv(fileName)



d = Data('college.csv')
d.printHead()