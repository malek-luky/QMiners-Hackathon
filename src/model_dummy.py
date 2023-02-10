import numpy as np
import pandas as pd
from datetime import datetime

seed = 42

import numpy as np
import pandas as pd
from datetime import datetime

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
        print(diff)
        final_bet = np.array([tip1[0]**4*diff[0],tip1[1]**4*diff[1]])
        k = 10000 # magic constant we need to tune
        final_bet = k*final_bet # magic :)))
        #TODO: Vzit v potaz ten jejich poddsa, tyhle vysledky jsou shit

        return final_bet

class Model:
    inc:pd.DataFrame = None
    total_bets:int = 0
    season_start:bool = True


    def place_bets(self, opps: pd.DataFrame, summary, inc):
        """
        opps: upcoming matches
        inc: historical matches
        summary: bankroll, min_bet, max_bet
        """
        N = len(opps)
        bets = np.zeros((N, 2))
        #if summary["Bankroll"][0]>1300:
        #   return pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)

        ##################################################################
        # FUNCTIONS FOR MACHINE LEARNING
        ##################################################################
        short_time = 60 # number of days to look back (last month) #TODO: Tune to find a parameter that works well
        long_time = 600 # number of days to look back (3 years) #TODO: Tune to find a parameter that works well

        ##################################################################
        # APPEND LIST
        ##################################################################
        if self.inc is None:
            self.inc = inc
        else:
            self.inc = pd.concat([self.inc,inc])

        ##################################################################
        # PRECALCULATIONS
        ##################################################################
        no_matches_short = Lukas().matches_during_timeperiod(self.inc, opps.iloc[-1]['HID'], short_time)
        no_matches_long = Lukas().matches_during_timeperiod(self.inc, opps.iloc[-1]['HID'], long_time)
        # number of matches is similar for all teams
        
        min_bet = summary.iloc[0].to_dict()['Min_bet']
        max_bet = summary.iloc[0].to_dict()['Max_bet']
        

        ##################################################################
        # PRESEASON CALCULATIONS
        ##################################################################
        
        """
        if self.season_start == True:
            self.season_start = False 
            team_list = (self.inc['HID'].unique())
            col_values = {'Team': team_list, 'Team_form': np.zeros(len(team_list)), 'Winrate': np.zeros(len(team_list)), 'Draws': np.zeros(len(team_list)), 'Draws_rate': np.zeros(len(team_list)), 'Scored_goals': np.zeros(len(team_list)), 'Conceded_goals': np.zeros(len(team_list)), 'Shots_on_goal': np.zeros(len(team_list)), 'Conceded_shots': np.zeros(len(team_list)), 'Penalty_minutes': np.zeros(len(team_list)), 'Shots_on_goal': np.zeros(len(team_list)), 'Goals_powerplay': np.zeros(len(team_list)), 'Won_pucks': np.zeros(len(team_list))}
            data = pd.DataFrame(col_values)
            for idx, row in data.iterrows():
                data.at[idx, 'Team_form'] = Lukas().team_form(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Winrate'] = Lukas().winrate(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Draws'] = Lukas().draws(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Draws_rate'] = Lukas().draws_rate(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Scored_goals'] = Lukas().scored_goals(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Conceded_goals'] = Lukas().conceded_goals(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Shots_on_goal'] = Lukas().shots_on_goal(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Conceded_shots'] = Lukas().conceded_shots(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Penalty_minutes'] = Lukas().penalty_minutes(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Shots_on_goal'] = Lukas().shots_on_goal(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Goals_powerplay'] = Lukas().goals_powerplay(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Won_pucks'] = Lukas().won_pucks(self.inc, row['Team'], no_matches_long)
        """

        ##################################################################
        # CALCULATIONS THROUGHOUT THE SEASON
        ##################################################################
        for i, row in enumerate(opps.iterrows()):

            # BET ONLY LAST DAY BEFORE THE MATCH
            if Lukas().days_from_match(summary["Date"].loc[0], row[1]["Date"])==1:
                continue
            

            # BET BASED ON FORM           
            bet = np.array([0, 0])
            match = row[1]
            if Lukas().team_form(self.inc, row[1]['HID'], 20) > 15 and match["OddsH"]<2:
                bet[0] += 80
            elif Lukas().team_form(self.inc, row[1]['AID'], 20) > 15 and match["OddsA"]<2:
                bet[1] += 80
            bets[i] = bet
            
        
        return pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)
