from sklearn.linear_model import LogisticRegression, Ridge
import sklearn
import sklearn.compose
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.metrics
import sklearn.ensemble
import xgboost as xgb
from datetime import datetime

cashflow=[]

class Lukas:

    ##################################################################
    # HELP FUNCTIONS
    ##################################################################
    def days_from_match(self, today, day_match):
        format = "%Y-%m-%d %H:%M:%S"
        today = datetime.strptime(str(today), format)
        day_match = datetime.strptime(str(day_match), format)
        return abs((day_match - today).days)

    def matches_during_timeperiod(self, data, team_id, no_of_days):
        matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        today = data.iloc[-1]["Date"]
        for i in range(1,len(matches)):
            match_date = matches.iloc[-i]["Date"]
            if self.days_from_match(today,match_date) > no_of_days:
                break
        return i

    ##################################################################
    # CORE FUNCTIONS 
    ##################################################################

    def team_form(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        home_wins = last_matches.loc[(last_matches['HID']== team_id) & (last_matches['H'] == 1)]
        away_wins = last_matches.loc[(last_matches['AID']== team_id) & (last_matches['A'] == 1)]
        total_wins = len(home_wins) + len(away_wins)
        return total_wins #in last n matches

    def winrate(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        home_wins = last_matches.loc[(last_matches['HID']== team_id) & (last_matches['H'] == 1)]
        away_wins = last_matches.loc[(last_matches['AID']== team_id) & (last_matches['A'] == 1)]
        total_wins = len(home_wins) + len(away_wins)
        total_matches = len(last_matches)
        return total_wins/total_matches #winrate

    def draws(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        draws = last_matches.loc[(last_matches['H'] == 0) & (last_matches['A'] == 0)]
        total_draws = len(draws)
        return total_draws

    def draws_rate(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        draws = last_matches.loc[(last_matches['H'] == 0) & (last_matches['A'] == 0)]
        total_draws = len(draws)
        total_matches = len(last_matches)
        return total_draws/total_matches

    def scored_goals(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        home_goals = last_matches.loc[(last_matches['HID']== team_id)]['HSC']
        away_goals = last_matches.loc[(last_matches['AID']== team_id)]['ASC']
        total_goals = pd.concat([home_goals,away_goals])
        return total_goals.mean()

    def scored_goals_wins(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        home_goals = last_matches.loc[(last_matches['HID']== team_id) & (last_matches['H'] == 1)]['HSC']
        away_goals = last_matches.loc[(last_matches['AID']== team_id) & (last_matches['A'] == 1)]['ASC']
        total_goals = pd.concat([home_goals,away_goals])
        return total_goals.mean()

    def conceded_goals(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        home_goals = last_matches.loc[(last_matches['HID']== team_id)]['ASC']
        away_goals = last_matches.loc[(last_matches['AID']== team_id)]['HSC']
        total_goals = pd.concat([home_goals,away_goals])
        return total_goals.mean()

    def conceded_goals_wins(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        home_goals = last_matches.loc[(last_matches['HID']== team_id) & (last_matches['H'] == 1)]['ASC']
        away_goals = last_matches.loc[(last_matches['AID']== team_id) & (last_matches['A'] == 1)]['HSC']
        total_goals = pd.concat([home_goals,away_goals])
        return total_goals.mean()

    def shots_on_goal(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        shots_home = last_matches.loc[(last_matches['HID']== team_id)]['S_H']
        shots_away = last_matches.loc[(last_matches['AID']== team_id)]['S_A']
        total_pen = pd.concat([shots_home,shots_away])
        return total_pen.mean()

    def conceded_shots(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        shots_home = last_matches.loc[(last_matches['HID']== team_id)]['S_A']
        shots_away = last_matches.loc[(last_matches['AID']== team_id)]['S_H']
        total_pen = pd.concat([shots_home,shots_away])
        return total_pen.mean()

    def penalty_minutes(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        home_pen = last_matches.loc[(last_matches['HID']== team_id)]['PIM_H']
        away_pen = last_matches.loc[(last_matches['AID']== team_id)]['PIM_A']
        total_pen = pd.concat([home_pen,away_pen])
        return total_pen.mean()

    def goals_powerplay(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        home_pen = last_matches.loc[(last_matches['HID']== team_id)]['PPG_H']
        away_pen = last_matches.loc[(last_matches['AID']== team_id)]['PPG_A']
        total_pen = pd.concat([home_pen,away_pen])
        return total_pen.mean()

    def won_pucks(self, data, team_id, matches):
        last_matches = (data.loc[(data['HID']== team_id) | (data['AID']== team_id)])
        last_matches = last_matches.tail(matches)
        home_pen = last_matches.loc[(last_matches['HID']== team_id)]['FOW_H']
        away_pen = last_matches.loc[(last_matches['AID']== team_id)]['FOW_A']
        total_pen = pd.concat([home_pen,away_pen])
        return total_pen.mean()

    ##################################################################
    # FUNCTIONS TO BE CALLED
    ##################################################################
    def winrate_prob(self, data, team1, team2, matches):
        winrate_team1 = self.winrate(data, team1, matches)
        winrate_team2 = self.winrate(data, team2, matches)
        draws = self.draws(data, team1, matches)
        return np.array([winrate_team1, 0, winrate_team2])

    def form_prob(self, data, team1, team2, matches):
        form_team1 = self.team_form(data, team1, matches)
        form_team2 = self.team_form(data, team2, matches)
        team1_prob = form_team1 / (form_team1 + form_team2)
        team2_prob = 1 - team1_prob
        return np.array([team1_prob, 0, team2_prob])

    def match_final_prob(self, data, team1, team2, matches):
        form_team1 = self.team_form(data, team1, matches)
        draws_team1 = self.draws(data, team1, matches)
        form_team2 = self.team_form(data, team2, matches)
        draws_team2 = self.draws(data, team2, matches)
        result = [form_team1, (draws_team1+draws_team2)/2, form_team2]
        result = 1/sum(result)*np.array(result)
        return result

    def match_final_prob_v2(self, data, team1, team2, matches):
        form_team1 = self.team_form(data, team1, matches)
        draws_team1 = self.draws(data, team1, matches)
        form_team2 = self.team_form(data, team2, matches)
        draws_team2 = self.draws(data, team2, matches)
        goal_diff_team1 = self.scored_goals_wins(data, team1, matches) - self.conceded_goals_wins(data, team1, matches)
        goal_diff_team2 = self.scored_goals_wins(data, team2, matches) - self.conceded_goals_wins(data, team2, matches)
        result = [goal_diff_team1*form_team1, ((goal_diff_team1+goal_diff_team2)/2)*(draws_team1+draws_team2)/2, goal_diff_team2*form_team2]
        result = 1/sum(result)*np.array(result)
        return result

    ##################################################################
    # BETTING
    ##################################################################

    def bet_with_draw(self, tip1, tip2):
        """
        DRAW IS NOT A CASE ANYMORE :(
        tip1: [winA, draw, winB] (us)
        tip2: [winA, winB] (bookmaker)
        """
        if sum(tip1) < 0.99 or sum(tip1) > 1.01:
            print("Tip1 Sum =",sum(tip1))
            raise ValueError('tip1 must sum to 1')

        k = 100 # magic constant we need to tune
        draw = 7-abs(tip2[1]-tip2[0])+3.6
        k = 1 / (1/tip2[0] + 1/tip2[1] + 1/draw)
        tip2 = [(1/tip2[0])*k, (1/draw)*k, (1/tip2[1])*k] # now we have the same format as tip1
        if sum(tip2) < 0.99 or sum(tip2) > 1.01:
            print("Tip2 Sum =",sum(tip2))
            raise ValueError('tip2 must sum to 1')
        p = max(tip1) # our maximum probability for a win
        max_index = tip1.argmax() #team we bet on
        P = tip2[max_index]
        diff = p-P # difference between our and bookmaker's probability
        if diff<0: return [0,0] # kurz is not worth it
        final_bet = p*diff # take into account our the probability of winning (bet more when we believe in it more)
        final_bet = (1-tip1[1])*final_bet # take into account the draw probability
        final_bet = k*final_bet # magic :)))
        print(final_bet)
        if max_index == 0: return [final_bet, 0] # home team highest prob  
        elif max_index == 1: return [0, 0] # we don't bet on a draw
        elif max_index == 2: return [0, final_bet] # away team highest prob

    def bet_v1(self, tip1, tip2):
        """
        DRAW IS NOT A CASE ANYMORE :(
        tip1: [winA, winB] (us)
        tip2: [odsA, odsB] (bookmaker)
        """
        print("\nKURZ", tip2)
        if sum(tip1) < 0.99 or sum(tip1) > 1.01:
            print("Tip1 Sum =",sum(tip1))
            raise ValueError('tip1 must sum to 1')
        k = 1 / (1/tip2[0] + 1/tip2[1])
        tip2 = [(1/tip2[0])*k, (1/tip2[1])*k] # now we have the same format as tip1
        if sum(tip2) < 0.99 or sum(tip2) > 1.01:
            print("Tip2 Sum =",sum(tip2))
            raise ValueError('tip2 must sum to 1')
        p = max(tip1) # our maximum probability for a win
        max_index = tip1.argmax() #team we bet on
        P = tip2[max_index]
        diff = p-P # difference between our and bookmaker's probability
        final_bet = p*diff # take into account our the probability of winning (bet more when we believe in it more)
        k = 100 # magic constant we need to tune
        final_bet = k*final_bet # magic :)))
        print("TIP1",tip1)
        print("TIP2",tip2)
        print("DIFF", diff, "SAZKA", final_bet)
        print("")


        if diff<0: return [0,0] # kurz is not worth it
        if max_index == 0: return [final_bet, 0] # home team highest prob  
        elif max_index == 1: return [0, final_bet] # we don't bet on a draw

    def bet(self, tip1, tip2):
        """
        DRAW IS NOT A CASE ANYMORE :(
        tip1: [winA, winB] (us)
        tip2: [odsA, odsB] (bookmaker)
        """
        if sum(tip1) < 0.99 or sum(tip1) > 1.01:
            print("Tip1 Sum =",sum(tip1))
            raise ValueError('tip1 must sum to 1')
        k = 1 / (1/tip2[0] + 1/tip2[1])
        tip2 = [(1/tip2[0])*k, (1/tip2[1])*k] # now we have the same format as tip1
        if sum(tip2) < 0.99 or sum(tip2) > 1.01:
            print("Tip2 Sum =",sum(tip2))
            raise ValueError('tip2 must sum to 1')
        diff = tip1-tip2
        diff[diff<0] = 0
        #print(diff)
        final_bet = np.array([tip1[0]**4*diff[0],tip1[1]**4*diff[1]])
        k = 10000 # magic constant we need to tune
        final_bet = k*final_bet # magic :)))
        #TODO: Vzit v potaz ten jejich poddsa, tyhle vysledky jsou shit

        return final_bet

class AllMatchesModel():
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
        non_int = np.array(list(range(4, 16)))
        self.dataset = None
        self.all_matches_model = None
        lr = sklearn.pipeline.Pipeline([
            ("preprocess", sklearn.compose.ColumnTransformer([
                ("onehot",
                 sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"),
                 ~non_int),
                ("scaler", sklearn.preprocessing.RobustScaler(), non_int),

            ])),
            ("poly", sklearn.preprocessing.PolynomialFeatures(2)),
            ('lr2', LogisticRegression(random_state=42, max_iter=1000, C=0.01))
        ]
        )
        self.model = lr


    def __prepare_data(self):
        Xde = []
        y = []

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

        self.model.fit(Xde, y)

    def predict(self, match):
        row = np.zeros(16)
        row[0:2] = (match["HID"], match["AID"])
        th, ta = (match["HID"], match["AID"])

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

        return self.model.predict_proba([row])
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing

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
        #self.scaler = sklearn.preprocessing.RobustScaler()

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
        #return self.model.predict_proba(inp), [[0, 0]], self.bookmaker_h.predict(inp), self.bookmaker_a.predict(inp)
        return self.model.predict_proba(inp), self.dominik.predict(match), self.bookmaker_h.predict(inp), self.bookmaker_a.predict(inp)
        #return 0,  self.dominik.predict(match), self.bookmaker_h.predict(inp), self.bookmaker_a.predict(inp)


    def fit(self):
        inp = self.poly.fit_transform(self.data)
        inp = self.scaler.fit_transform(inp)
        self.model.fit(inp, self.home_wins)
        self.dominik.fit(self.inc)
        self.bookmaker_h.fit(inp, self.oddsh)
        self.bookmaker_a.fit(inp, self.oddsa)
        

    def place_bets(self, opps, summary, inc):

        N = len(opps)
        min_bet = summary.iloc[0].to_dict()['Min_bet']
        max_bet = summary.iloc[0].to_dict()['Max_bet']
        bets = np.zeros((N, 2))
        if summary["Bankroll"][0]>1500:
            return pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)

        self.process_new_matches(inc)
        if self.inc is None:
            self.inc = inc
        else:
            self.inc = pd.concat([self.inc,inc])
        self.inc = self.inc.tail(1000)
        if len(cashflow)%10==0:
            self.fit()
        
        
        cashflow.append(summary["Bankroll"][0])
        thrust = 1
        for idx, row in enumerate(opps.iterrows()):
            

            # BET ONLY LAST DAY BEFORE THE MATCH
            # if Lukas().days_from_match(summary["Date"].loc[0], row[1]["Date"])==1:
            #     continue
            
            match = row[1]
            match_date = match["Date"]
            
            team_a, team_b = match["HID"], match["AID"]
            ben, dom, poddsh, poddsa = self.predict(team_a, team_b, match)

            dom = dom[0]
            ben = ben[0]
            prediction = np.array([dom[0] + ben[0], dom[1] + ben[1]]) / 2
                    ####################################################
        ########## DOMINIK + BEN V2
        ####################################################
            ##### NIZE JE TO ZAJIMAVE @luk @ver
            delta_h = (match["OddsH"] - poddsh)
            delta_a = (match["OddsA"] - poddsa)
            # bet based on odds prediction

            thrust = 1
            if len(cashflow)>4:
                thrust = 0
                if cashflow[-2] < cashflow[-1]:
                    thrust=1
                    if cashflow[-3] < cashflow[-1]:
                        thrust=2
                        if cashflow[-4] < cashflow[-1]:
                            thrust=3

            threshold = 1.4
            if delta_h >= threshold: 
                #print(delta_h, "HHH")
                bets[idx, 0] += max_bet*thrust/4
            elif delta_a >= threshold: 
                #print(delta_a, "AAA")
                bets[idx, 1] += max_bet*thrust/4

            

            threshold = 1.8
            if delta_h >= threshold: 
                bets[idx, 0] += max_bet*thrust/2
            elif delta_a >= threshold: 
                bets[idx, 1] += max_bet*thrust/2

            # bet based on win prediction
            bookmaker_h = 1 / match["OddsH"]
            bookmaker_a = 1 / match["OddsA"]
            k = 1/(bookmaker_h + bookmaker_a)
            bookmaker_h *= k
            bookmaker_a *= k

            # if prediction[0] > 0.9 and prediction[0] > (bookmaker_h):
            #     bets[idx, 0] += max_bet#*(1+prediction[0])
            # elif prediction[1] > 0.9 and prediction[1] > (bookmaker_a):
            #     bets[idx, 1] += max_bet#*(1+prediction[1])

            # varhany = 0.1
            # if prediction[0] > 0.8 and prediction[0] > (bookmaker_h + varhany):
            #     bets[idx, 0] += max_bet/2#*(1+prediction[0])
            # elif prediction[1] > 0.8 and prediction[1] > (bookmaker_a + varhany):
            #     bets[idx, 1] += max_bet/2#*(1+prediction[1])
            
            # varhany = 0.05
            # if prediction[0] > 0.8 and prediction[0] > (bookmaker_h + varhany):
            #     bets[idx, 0] += max_bet/2#*(1+prediction[0])
            # elif prediction[1] > 0.8 and prediction[1] > (bookmaker_a + varhany):
            #     bets[idx, 1] += max_bet/2#*(1+prediction[1])
            print(prediction[0], prediction[1], bookmaker_h, bookmaker_a)
            varhany = 0.08
            if prediction[0] > 0.7 and prediction[0] > (bookmaker_h + varhany):
                print("here")
                bets[idx, 0] += max_bet/4#*(1+prediction[0])
            elif prediction[1] > 0.7 and prediction[1] > (bookmaker_a + varhany):
                bets[idx, 1] += max_bet/4#*(1+prediction[1])



            #bets[idx] = bets[idx]*thrust
            bets[idx][bets[idx]>max_bet] = max_bet
            bets[idx][bets[idx]<min_bet] = 0
            print(bets[idx])
        # print(self.error / self.n)
        print(thrust)
        return pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)


#             #################################################
#             ######## LUKY
#             #################################################
#             bet = Lukas().bet(prediction,[row[1]["OddsH"], row[1]["OddsA"]])
#             delta_h = (match["OddsH"] - poddsh)[0]
#             delta_a = (match["OddsA"] - poddsa)[0]
#             # print("TU",match["OddsH"],match["OddsA"])
#             # print("ZDE",prediction)
#             # print("ZDE",delta_h, delta_a)
#             # print(poddsh)
#             # print(poddsa)
#             bet[bet>max_bet] = max_bet
#             bet[bet<min_bet] = 0
#             #print("SAZKA: ", bet)
#             bets[idx] = bet

#         return pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)
# """

# """
#         ####################################################
#         ########## DOMINIK + BEN
#         ####################################################
#             ##### NIZE JE TO ZAJIMAVE @luk @ver
#             delta_h = (match["OddsH"] - poddsh)
#             delta_a = (match["OddsA"] - poddsa)
#             print("ZDE",prediction)
#             print("ZDE",delta_h, delta_a)
#             # bet based on odds prediction
#             threshold = 1.8
#             if delta_h >= threshold: 
#                 #print(delta_h, "HHH")
#                 bets[idx, 0] += max_bet#*delta_h
#             elif delta_a >= threshold+0.5: 
#                 #print(delta_a, "AAA")
#                 bets[idx, 1] += max_bet#*delta_a

#             # bet based on win prediction
#             bookmaker_h = 1 / match["OddsH"]
#             bookmaker_a = 1 / match["OddsA"]
#             varhany = 0.05
#             if prediction[0] > 0.8: prediction[0] > (bookmaker_h + varhany):
#                 bets[idx, 0] += max_bet#*(1+prediction[0])
#             elif prediction[1] > 0.8 and prediction[1] > (bookmaker_a + varhany):
#                 bets[idx, 1] += max_bet#*(1+prediction[1])
#         # print(self.error / self.n)
# """

# TODO: sprav nejako normálne ten kód tak aby bol čitateľný a dal sa používať
# skús predikovať kurzy bookmakera
# resime modelech jen uspesnost nebo i winrate na kurz?