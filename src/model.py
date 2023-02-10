from math import inf

import numpy as np
import pandas as pd
import sklearn.linear_model

from probability_estimator import ProbabilityEstimator
from datetime import datetime
# from tabulate import tabulate
from model_regressor import ModelBen
from lukas import DataExtraction
from dominik import Dominik

seed = 42

np.random.seed(seed)


class Model:
    inc: pd.DataFrame = None
    ben=ModelBen()
    dominik=Dominik()

    def __should_bet_now(self, today, day_match):
        format = "%Y-%m-%d %H:%M:%S"
        today = datetime.strptime(str(today), format)
        day_match = datetime.strptime(str(day_match), format)
        return (day_match - today).days == 1

    def __get_bet_team(self, match, bookmaker_guess:float) -> float:
        #prob = self.ben.get_probabilities2(team_a, team_b)
        prob = self.dominik.predict(match)[0]
        print(prob)
        # a, b = DataExtraction().form_prob(self.inc, team_a, team_b, 10)
        assert 0 <= prob <= 1
        if prob > bookmaker_guess and prob != -1:
            return 2.0
        return 0

    def __simulate(self, tr, data, bet):
        balance = 0
        for row in data.iterrows():
            match = row[1]
            prob_a = self.dominik.predict(match)[0][0]
            prob_b = 1 - prob_a
            bh, ba = 0, 0
            if not match["OddsH"] or not match["OddsA"]:
                continue
            bookmaker_a = 1 / match["OddsH"]
            bookmaker_b = 1 / match["OddsA"]
            if prob_a > tr: # and prob_a > bookmaker_a:
                bh = bet
            if prob_b > tr: #and prob_b > bookmaker_b:
                ba = bet
            if match["H"]:
                balance += bh * match["OddsH"]
                balance -= ba
            if match["A"]:
                balance += ba * match["OddsA"]
                balance -= bh
        return -balance

    def __calculate_tr(self, data, options, bet):
        best = 0
        best_loss = inf
        #for b in range(5, 101, 5):
        #bet = b
        for option in options:
            loss = self.__simulate(option, data, bet)
            if loss < best_loss:
                best_loss = loss
                best = (option, bet)
        return best

    def place_bets(self, opps: pd.DataFrame, summary, inc):
        from random import random
        if self.inc is None:
            self.inc = inc
            self.ben.fit2(self.inc)
            self.dominik.fit(self.inc)
        else:
            self.inc = self.inc.append(inc)
            self.dominik.fit(self.inc.tail(1000))
        #self.ben.fit2(self.inc)
        N = len(opps)
        min_bet = summary.iloc[0].to_dict()['Min_bet']
        max_bet = summary.iloc[0].to_dict()['Max_bet']
        bets = np.zeros((N, 2))
        tr = 0.7
        bet = min_bet
        for idx, row in enumerate(opps.iterrows()):
            match = row[1]
            match_date = match["Date"]
            if not self.__should_bet_now(summary["Date"].loc[0], match_date):
                continue
            bookmaker_a = 1 / match["OddsH"]
            bookmaker_b = 1 / match["OddsA"]
            prob_a = self.dominik.predict(match)[0][0]
            prob_b = 1 - prob_a
            # uncer = 0.8
            # prob_a *= uncer
            # prob_b *= uncer
            # bet = min_bet
            # margin = 0.1
            # ex_a = prob_a * (match["OddsH"] - 1) - prob_b - margin
            # ex_b = prob_b * (match["OddsA"] - 1) - prob_a - margin
            # ex_both = prob_a * (match["OddsH"]) + prob_b * (match["OddsA"]) - 2 - margin
            # options = [ex_a, ex_b, ex_both, 0]
            #print(options)
            options = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            bet = max_bet
            if idx % 20 == -1:
                tr, bet = self.__calculate_tr(self.inc.tail(50), options, bet)
                print(tr, bet)
            tr = 0.7
            if prob_a > tr: # and prob_a > bookmaker_a:
                    bets[idx, 0] = bet
            if prob_b > tr: # and prob_b > bookmaker_b:
                    bets[idx, 1] = bet
            # if option_i == 2:
            #     bets[idx, 0] = max_bet/2
            #     bets[idx, 1] = max_bet/2
            #if match["OddsA"] > 3:
            #    bets[idx, 0] = min_bet
            #if match["OddsH"] > 3:
            #    bets[idx, 1] = min_bet

        #self.print_draws(self.inc)

        return pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)
