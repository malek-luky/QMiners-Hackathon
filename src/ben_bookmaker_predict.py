import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing
from dominik import Dominik

class Model:
    def __init__(self):
        self.team_stats = {}
        for i in range(30): self.team_stats[i] = {"wins": 0, "shots": 0, "won_buly": 0, "penalty": 0, "matches": 0, "odds": 0}
        self.model = sklearn.linear_model.LogisticRegression(max_iter=2000, random_state=42)
        self.dominik = Dominik()
        self.bookmaker_h = sklearn.linear_model.LinearRegression()
        self.bookmaker_a = sklearn.linear_model.LinearRegression()

        self.poly = sklearn.preprocessing.PolynomialFeatures(2)
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        
        self.data = np.empty(shape=(0, 8))
        self.home_wins = np.empty(shape=(0,1))
        self.oddsh = np.empty(shape=(0,1))
        self.oddsa = np.empty(shape=(0,1))
        # affects confidence of the model
        self.reset_factor = 1.4
        self.error, self.n = 0, 0
        self.inc = None

    def reset_stats(self, team_id):
        self.team_stats[team_id]["wins"] /= self.reset_factor
        self.team_stats[team_id]["shots"] /= self.reset_factor
        self.team_stats[team_id]["penalty"] /= self.reset_factor
        self.team_stats[team_id]["matches"] //= self.reset_factor
        self.team_stats[team_id]["odds"] //= self.reset_factor
        
    def add_stats(self, team_id, match, is_home):
        home = ["S_H", "FOW_H", "PIM_H", "OddsH"]
        away = ["S_A", "FOW_A", "PIM_A", "OddsA"]
        if is_home: keys = home
        else: keys = away
        self.team_stats[team_id]["shots"] += match[keys[0]]
        self.team_stats[team_id]["won_buly"] += match[keys[1]]
        self.team_stats[team_id]["penalty"] += match[keys[2]]
        self.team_stats[team_id]["odds"] += match[keys[3]]
        self.team_stats[team_id]["matches"] += 1

    def process_new_matches(self, inc):
        for i, r in enumerate(inc.iterrows()):
            match = r[1]
            th, ta = match["HID"], match["AID"] 
            
            # lower the importance of old matches
            if self.team_stats[th]["matches"] % 30 == 0: self.reset_stats(th)
            if self.team_stats[ta]["matches"] % 30 == 0: self.reset_stats(ta)
                
            if match["H"]: self.team_stats[th]["wins"] += 1 
            self.add_stats(th, match, True)
            
            if match["A"]: self.team_stats[ta]["wins"] += 1 
            self.add_stats(ta, match, False)

            # ignore games with draw
            if (match["H"] or match["A"]):
                mh, ma = self.team_stats[th]["matches"], self.team_stats[ta]["matches"]
                new_row = np.array([
                    [self.team_stats[th]["wins"]/mh, self.team_stats[th]["shots"]/mh, 
                    self.team_stats[th]["won_buly"]/mh, self.team_stats[th]["odds"]/mh,
                    self.team_stats[ta]["wins"]/ma, self.team_stats[ta]["shots"]/ma, 
                    self.team_stats[ta]["won_buly"]/ma, self.team_stats[ta]["odds"]/ma]
                ])
                self.data = np.append(self.data, new_row, axis=0)
                
                if match["H"]: self.home_wins = np.append(self.home_wins, 0)
                else: self.home_wins = np.append(self.home_wins, 1)
                self.oddsh = np.append(self.oddsh, match["OddsH"])
                self.oddsa = np.append(self.oddsa, match["OddsA"])

                    
    def predict(self, th, ta, match):
        mh, ma = self.team_stats[th]["matches"], self.team_stats[ta]["matches"]
        inp = self.poly.transform([[
                self.team_stats[th]["wins"]/mh, self.team_stats[th]["shots"]/mh, 
                self.team_stats[th]["won_buly"]/mh, self.team_stats[th]["odds"]/mh,
                self.team_stats[th]["wins"]/ma, self.team_stats[th]["shots"]/ma, 
                self.team_stats[th]["won_buly"]/ma, self.team_stats[th]["odds"]/ma,
        ]])
        inp = self.scaler.transform(inp)
        #return self.model.predict_proba(inp), self.bookmaker_h.predict(inp), self.bookmaker_a.predict(inp)
        return self.model.predict_proba(inp), self.dominik.predict(match), self.bookmaker_h.predict(inp), self.bookmaker_a.predict(inp)
        #return 0,  self.dominik.predict(match), self.bookmaker_h.predict(inp), self.bookmaker_a.predict(inp)


    def fit(self):
        inp = self.poly.fit_transform(self.data)
        inp = self.scaler.fit_transform(inp)
        self.model.fit(inp, self.home_wins)
        self.dominik.fit(self.inc.tail(1000))
        self.bookmaker_h.fit(inp, self.oddsh)
        self.bookmaker_a.fit(inp, self.oddsa)
        

    def place_bets(self, opps, summary, inc):
        self.process_new_matches(inc)
        if self.inc is None:
            self.inc = inc
        else:
            self.inc = self.inc.append(inc)
        self.fit()
        
        N = len(opps)
        min_bet = summary.iloc[0].to_dict()['Min_bet']
        bets = np.zeros((N, 2))

        for idx, row in enumerate(opps.iterrows()):
            match = row[1]
            match_date = match["Date"]
            
            team_a, team_b = match["HID"], match["AID"]
            ben, dom, poddsh, poddsa = self.predict(team_a, team_b, match)
            #print(prediction)
            dom = dom[0]
            ben = ben[0]
            prediction = np.array([dom[0] + ben[0], dom[1] + ben[1]]) / 2

            delta_h = (match["OddsH"] - poddsh)[0]
            delta_a = (match["OddsA"] - poddsa)[0]

            # bet based on odds prediction
            threshold = 1.8
            if delta_h >= threshold: 
                #print(delta_h, "HHH")
                bets[idx, 0] =+ min_bet*delta_h
            elif delta_a >= threshold+0.5: 
                #print(delta_a, "AAA")
                bets[idx, 1] =+ min_bet*delta_a

            # bet based on win prediction
            bookmaker_h = 1 / match["OddsH"]
            bookmaker_a = 1 / match["OddsA"]
            if prediction[0] >= 0.7 and prediction[0] > bookmaker_h:
                bets[idx, 0] =+ min_bet*(1+prediction[0]) *4
            elif prediction[1] >= 0.7 and prediction[1] > bookmaker_a:
                bets[idx, 1] =+ min_bet*(1+prediction[1]) * 4
        # print(self.error / self.n)
        return pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)


# TODO: sprav nejako normálne ten kód tak aby bol čitateľný a dal sa používať
# skús predikovať kurzy bookmakera