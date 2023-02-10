import numpy as np
import pandas as pd
from datetime import datetime

ELO_DICT = dict()

class Elo:

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

    # def new_season(self, data, match):
    #     no_matches = 0
    #     today = match["Date"]
    #         match_date = data.iloc[-i]["Date"]
    #         if self.days_from_match(today,match_date) <60 :
    #             no_matches+=1
    #     if no_matches==0:
    #         print("NOW")
    #     return no_matches

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

    def team_elo(self, data, team):
        for i, row in enumerate(data.iterrows()):

            team_home = row[1]['HID']
            team_away = row[1]['AID']

            # if (self.matches_during_timeperiod(data, team_home, 30) == 0):
            #     print("now")

            #print(row)
            
            #check whether row is in ELO_DICT if not add it
            if team_home not in ELO_DICT:
                ELO_DICT[team_home] = 1000
            if team_away not in ELO_DICT:
                ELO_DICT[team_away] = 1000
            
            #get ELO values
            elo_home = ELO_DICT[team_home]
            elo_away = ELO_DICT[team_away]

            
            # print("\nPred zapasem:",ELO_DICT[team_home], ELO_DICT[team_away])
            # print("Zapas:",team_home, team_away, row[1]['H'],row[1]['A'])

            # prob_home = 1 / (1 + 10**((elo_home-elo_away)/400))
            # prob_away = 1 / (1 + 10**((elo_away-elo_home)/400))
            
            # K = 1
            # if (row[1]['H'] == True):
            #     ELO_DICT[team_home] = elo_home + K*(1-prob_home)
            #     ELO_DICT[team_away] = elo_away + K*(0-prob_away)
            # elif (row[1]['A'] == True):
            #     ELO_DICT[team_home] = elo_home + K*(0-prob_home)
            #     ELO_DICT[team_away] = elo_away + K*(1-prob_away)

            # print("Po zapasu:",ELO_DICT[team_home], ELO_DICT[team_away],"\n")
            #WEa is the expected result for player a
            #Rb and Ra are the ratings of both players.
            #As you can see in a game between two players having 400 points
            #of difference in rating the chances of winning the game are 10:1.

            #
            # BET ONLY LAST DAY BEFORE THE MATCH
            # if DataExtraction().days_from_match(summary["Date"].loc[0], row[1]["Date"])==2:
            #     continue

            # # BET BASED ON FORM
            # team1 = Elo().team_elo(self.inc, row[1]['HID'])
            # team2 = Elo().team_elo(self.inc, row[1]['AID'])
            # bet[bet<min_bet] = 0
            # bet[bet>max_bet] = max_bet
            # bets[i] = bet
        
