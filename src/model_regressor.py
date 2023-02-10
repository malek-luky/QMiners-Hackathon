import sklearn.linear_model
import sklearn.model_selection

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

from probability_estimator import ProbabilityEstimator

from lukas import DataExtraction

# file_path = "../data/training_data.csv"
# d = Dataset(file_path)
# data = d.get_data()
# target = d.get_target()

class ModelBen():
    def __init__(self) -> None:
        #self.model = sklearn.linear_model.LogisticRegression(random_state=42)
        #self.model = sklearn.neural_network.MLPClassifier(128,random_state=42, max_iter=100)
        self.model = sklearn.linear_model.LogisticRegression(random_state=42)
        self.prepro = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        self.inc = pd.DataFrame()
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()

    def get_probabilities(self, p, a, b, c, d, e, f):
        sample = np.array([p,a,b, c,d,e,f]).reshape(1, -1)
        return self.model.predict_proba(sample)[0][0]

    def get_probabilities2(self, team_a, team_b):
        sample = np.array([team_b, team_b]).reshape(1, -1)
        sample = self.prepro.transform(sample)
        print(self.model.predict(sample))
        return self.model.predict_proba(sample)[0][0]

    def extract_features(self, data):
        example_cnt = len(data)
        example_cnt = 5
        raw_data = np.zeros((example_cnt, 7))
        target = np.zeros(example_cnt)

        for i, row in enumerate(self.inc.iterrows()):
            if i == example_cnt: break
            a, b = DataExtraction().form_prob(self.inc, row[1]["HID"],row[1]["AID"], 5)
            c, d = DataExtraction().form_prob(self.inc, row[1]["HID"],row[1]["AID"], 10)
            e, f = DataExtraction().form_prob(self.inc, row[1]["HID"],row[1]["AID"], 20)
            raw_data[i, 0] = ProbabilityEstimator.get_probability(row[1]["HID"], row[1]["AID"],self.inc)
            raw_data[i, 1] = a
            raw_data[i, 2] = b
            raw_data[i, 3] = c
            raw_data[i, 4] = d
            raw_data[i, 5] = e
            raw_data[i, 6] = f
            target[i] = row[1]["A"]
            if not row[1]["H"] and not row[1]["A"]:
                target[i] = 0
        return raw_data, target

    def extract_features2(self, data):
        example_cnt = len(data)
        #example_cnt = 5
        raw_data = np.zeros((example_cnt, 2))
        target = np.zeros(example_cnt)

        for i, row in enumerate(self.inc.iterrows()):
            if i == example_cnt: break
            raw_data[i][0] = row[1]["HID"]
            raw_data[i][1] = row[1]["AID"]
            target[i] = row[1]["A"]
            if not row[1]["H"] and not row[1]["A"]:
                target[i] = 0
        return raw_data, target

    def fit(self, inc):
        self.inc = inc
        data, target = self.extract_features(self.inc)        
        self.model.fit(data, target)
        print(self.model.score(data, target), "Evaluate!")

    def fit2(self, inc):
        self.inc = inc
        data, target = self.extract_features2(self.inc)
        data = self.prepro.fit_transform(data)
        print(target)
        self.model.fit(data, target)
        print(self.model.score(data, target), "Evaluate!")


    def place_bets(self, opps, summary, inc):
        if self.inc is None:
            self.inc = inc
        else:
            self.inc = self.inc.append(inc)

        # self.fit(self.inc)

        self.model.predict()
        # print(self.model.score(test_data, test_target))
        
        return None

    def evaluate(self, inc):
        raise NotImplementedError()

