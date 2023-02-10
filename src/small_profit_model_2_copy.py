import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing

class Model:
    def __init__(self):
        self.team_stats = {}
        for i in range(30): self.team_stats[i] = {"wins": 0, "shots": 0, "won_buly": 0, "penalty": 0, "matches": 0 }
        self.model = sklearn.linear_model.LogisticRegression(max_iter=800, random_state=42)
        self.poly = sklearn.preprocessing.PolynomialFeatures(2)
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        
        self.y = []
        self.x = []
        self.reset_factor = 1.2

    def fit(self, inc):
        for i, r in enumerate(inc.iterrows()):
            match = r[1]
            th, ta = match["HID"], match["AID"] 
            
            if self.team_stats[th]["matches"] % 30:
                self.team_stats[th]["wins"] /= self.reset_factor
                self.team_stats[th]["shots"] /= self.reset_factor
                self.team_stats[th]["penalty"] /= self.reset_factor
                self.team_stats[th]["matches"] //= self.reset_factor

            if self.team_stats[ta]["matches"] % 30:
                self.team_stats[ta]["wins"] /= self.reset_factor
                self.team_stats[ta]["shots"] /= self.reset_factor
                self.team_stats[ta]["penalty"] /= self.reset_factor
                self.team_stats[ta]["matches"] //= self.reset_factor

            if match["H"]: self.team_stats[th]["wins"] += 1 
            self.team_stats[th]["shots"] += match["S_H"]
            self.team_stats[th]["won_buly"] += match["FOW_H"]
            self.team_stats[th]["penalty"] += match["PIM_H"]
            self.team_stats[th]["matches"] += 1

            if match["A"]: self.team_stats[ta]["wins"] += 1 
            self.team_stats[ta]["shots"] += match["S_A"]
            self.team_stats[ta]["won_buly"] += match["FOW_A"]
            self.team_stats[ta]["penalty"] += match["PIM_A"]
            self.team_stats[ta]["matches"] += 1
            mh, ma = self.team_stats[th]["matches"], self.team_stats[ta]["matches"]
            
            self.x.append([self.team_stats[th]["wins"]/mh, self.team_stats[th]["shots"]/mh, self.team_stats[th]["won_buly"]/mh,
                    self.team_stats[th]["wins"]/ma, self.team_stats[th]["shots"]/ma, self.team_stats[th]["won_buly"]/ma])

            if match["H"]: self.y.append(0)
            elif not match["H"] and not match["A"]: self.y.append(2)
            else: self.y.append(1)
        
        inp = self.poly.fit_transform(self.x)
        inp = self.scaler.fit_transform(inp)
        self.model.fit(inp, self.y)

           
    def predict(self, th, ta):
        mh, ma = self.team_stats[th]["matches"], self.team_stats[ta]["matches"]
        inp = self.poly.transform([[self.team_stats[th]["wins"]/mh, self.team_stats[th]["shots"]/mh, self.team_stats[th]["won_buly"]/mh,
                self.team_stats[th]["wins"]/ma, self.team_stats[th]["shots"]/ma, self.team_stats[th]["won_buly"]/ma]])
        inp = self.scaler.transform(inp)
        return self.model.predict_proba(inp)

    
    def place_bets(self, opps, summary, inc):
        self.fit(inc)

        N = len(opps)
        min_bet = summary.iloc[0].to_dict()['Min_bet']
        bets = np.zeros((N, 2))
        
        for idx, row in enumerate(opps.iterrows()):
            match = row[1]
            match_date = match["Date"]
            
            team_a, team_b = match["HID"], match["AID"]
            prediciton = self.predict(team_a, team_b)[0]

            # print(prediciton)         
            if prediciton[0] >= 0.90: bets[idx, 0] = min_bet*(1+prediciton[0])
            elif prediciton[1] >= 0.90: bets[idx, 1] = min_bet*(1+prediciton[1])
        return pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)
