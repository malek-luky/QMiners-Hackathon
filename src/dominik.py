import sys
sys.path.append(".")

import numpy as np
import pandas as pd
from probability_estimator import ProbabilityEstimator
from environment import Environment

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from sklearn.linear_model import LogisticRegression, Ridge
import sklearn
import sklearn.compose
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.metrics
import sklearn.ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

from lukas import DataExtraction

class ModelAbstract(ABC):
    """ Tohle má dědit každý náš vlastní model. Je to kvůli jednotnému API. """
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def score(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class AllMatchesModel(ModelAbstract):
    """
    All matches between two teams.
    """
    def fit(self, X, y):
        assert len(X) == len(y)
        self.X = X.astype(int)
        self.y = y.astype(int)
        teams = set()
        for r in X:
            teams.add(r[0])
            teams.add(r[1])
        self.d = {}
        for team in teams:
            if team not in self.d:
                self.d[team] = [None, None]
            self.d[team][0] = (self.X[:, 0] == team)
            self.d[team][1] = (self.X[:, 1] == team)

    def score(self, X, y):
        correct = 0
        preds = np.apply_along_axis(self.predict, 1, X)
        #print("Predcs created")
        for i, t in enumerate(y):
            p = preds[i]
            if p > 0.5 and t == 1:
                correct += 1
            elif p < 0.5 and t == 0:
                correct += 1
            elif t == p:
                correct += 1
        return correct / len(y)

    def predict(self, x):
        team_a = x[0]
        team_b = x[1]
        a_home = np.logical_and(self.d[team_a][0], self.d[team_b][1])
        b_home = np.logical_and(self.d[team_a][1], self.d[team_b][0])
        a_win = np.sum(self.y[a_home] == 0) + np.sum(self.y[b_home] == 1)
        b_win = np.sum(self.y[a_home] == 1) + np.sum(self.y[b_home] == 0)
        tot = a_win + b_win
        if tot == 0:
            return 0.5
        return b_win / tot

class Dominik:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.all_matches_model = None
        # lr = sklearn.pipeline.Pipeline([
        #     ("preprocess", sklearn.compose.ColumnTransformer([
        #         ("onehot",
        #          sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"),
        #          int_columns),
        #         ("scaler", sklearn.preprocessing.RobustScaler(), ~int_columns),
        #     ])),
        #     ('lr2', LogisticRegression(random_state=42, max_iter=1000, C=0.01))
        # ]
        # )
        # self.model = lr

    def __prepare_data(self):
        Xde = []
        y = []
        DE = DataExtraction()

        for i, r in enumerate(self.dataset.iterrows()):
            match = r[1]
            if not match["H"] and not match["A"]:
                continue
            if match["H"]:
                y.append(0)
            # print(match["A"])
            if match["A"]:
                y.append(1)
            assert match["H"] ^ match["A"] or (not match["H"] and not match["A"])

            # continue
            row = np.zeros(16)
            row[0:2] = (match["HID"], match["AID"])
            th, ta = (match["HID"], match["AID"])
            # try:
            #     row[2:4] = DE.winrate_prob(self.dataset, row[0], row[1], 10)
            #     row[4:6] = DE.form_prob(self.dataset, row[0], row[1], 10)
            #     row[6:8] = DE.winrate_prob(self.dataset, row[0], row[1], 5)
            #     row[8:10] = DE.form_prob(self.dataset, row[0], row[1], 5)
            # except ZeroDivisionError:
            #     pass
            row[3] = self.all_matches_model.predict(row[0:2])
            row[4] = self.d[th]["S"]
            row[5] = self.d[ta]["S"]
            row[6] = self.d[th]["PIM"]
            row[7] = self.d[ta]["PIM"]
            row[8] = self.d[th]["PPG"]
            row[9] = self.d[ta]["PPG"]
            row[10] = self.d[th]["FOW"]
            row[11] = self.d[ta]["FOW"]
            row[12] = self.d[th]["SC"]
            row[13] = self.d[ta]["SC"]
            row[14] = match["OddsH"]
            row[15] = match["OddsA"]

            Xde.append(row)

        Xde = np.array(Xde)
        y = np.array(y)
        return Xde, y

    def fit(self, dataset):
        self.dataset = dataset
        self.teams = set()
        for i, r in enumerate(dataset.iterrows()):
            match = r[1]
            self.teams.add(match["HID"])
            self.teams.add(match["AID"])
        attributes = ("S", "PIM", "PPG", "FOW")
        self.d = {}
        for team in self.teams:
            teamh = dataset.loc[(dataset["HID"] == team)]
            teama = dataset.loc[(dataset["AID"] == team)]
            tot = len(teamh) + len(teama)
            for a in attributes:
                if not team in self.d:
                    self.d[team] = {}
                val = (np.sum(teamh[a + "_H"]) + np.sum(teamh[a + "_A"])) / tot
                self.d[team][a] = val
            val = (np.sum(teamh["HSC"]) + np.sum(teamh["ASC"])) / tot
            self.d[team]["SC"] = val

        xx = np.zeros((dataset.shape[0], 2))
        yy = np.ones(dataset.shape[0]) * 0.5

        for i, r in enumerate(dataset.head(1000).iterrows()):
            match = r[1]
            xx[i] = (match["HID"], match["AID"])
            if match["H"]: yy[i] = 0
            if match["A"]: yy[i] = 1

        self.all_matches_model = AllMatchesModel()
        self.all_matches_model.fit(xx, yy)

        Xde, y = self.__prepare_data()

        int_columns = np.all(Xde.astype(int) == Xde, axis=0)

        lr = sklearn.pipeline.Pipeline([
                ("preprocess", sklearn.compose.ColumnTransformer([
                ("onehot",
                sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"),
                int_columns),
                ("scaler", sklearn.preprocessing.RobustScaler(), ~int_columns),
            ])),
                ('lr2', LogisticRegression(random_state=42, max_iter=1000, C=0.01))
            ]
        )

        boost = sklearn.pipeline.Pipeline([
            ("preprocess", sklearn.compose.ColumnTransformer([
                ("onehot",
                 sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"),
                 int_columns),
                ("scaler", sklearn.preprocessing.RobustScaler(), ~int_columns),
            ])),
            ("xgb", xgb.XGBClassifier(objective="binary:logistic",
                                                        colsample_bytree=0.5,
                                                        gamma=0.2,
                                                        learnin_rate=0.01,
                                                        max_depth=2,
                                                        reg_lambda=0.1,
                                                        scale_pos_weight=0.5,
                                                        subsample=0.8,
                                                        silent=True,
                                                        verbosity=0
                                                       )),
        ]
        )

        # mlp = sklearn.pipeline.Pipeline([
        #     ("preprocess", sklearn.compose.ColumnTransformer([
        #         ("onehot",
        #          sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"),
        #          int_columns),
        #         ("scaler", sklearn.preprocessing.RobustScaler(), ~int_columns),
        #     ])),
        #     ("mlp", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(256,), max_iter=500))
        # ]
        # )

        self.model = sklearn.ensemble.StackingClassifier([
            ('lr', lr),
            #('boost', boost),
            #('mlp', mlp)
        ])

        self.model.fit(Xde, y)

    def predict(self, match):
        row = np.zeros(23)
        row[0:2] = (match["HID"], match["AID"])
        th, ta= row[0:2]
        # DE = DataExtraction()
        # try:
        #     row[2:4] = DE.winrate_prob(self.dataset, row[0], row[1], 10)
        #     row[4:6] = DE.form_prob(self.dataset, row[0], row[1], 10)
        #     row[6:8] = DE.winrate_prob(self.dataset, row[0], row[1], 5)
        #     row[8:10] = DE.form_prob(self.dataset, row[0], row[1], 5)
        # except ZeroDivisionError:
        #     pass
        row[10] = self.all_matches_model.predict(row[0:2])
        row[11] = self.d[th]["S"]
        row[12] = self.d[ta]["S"]
        row[13] = self.d[th]["PIM"]
        row[14] = self.d[ta]["PIM"]
        row[15] = self.d[th]["PPG"]
        row[16] = self.d[ta]["PPG"]
        row[17] = self.d[th]["FOW"]
        row[18] = self.d[ta]["FOW"]
        row[19] = self.d[th]["SC"]
        row[20] = self.d[ta]["SC"]
        row[21] = match["OddsH"]
        row[22] = match["OddsA"]

        return self.model.predict_proba([row])