import numpy as np
import pandas as pd
from lukas import DataExtraction
from datetime import datetime

seed = 42

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
        no_matches_short = DataExtraction().matches_during_timeperiod(self.inc, opps.iloc[-1]['HID'], short_time)
        no_matches_long = DataExtraction().matches_during_timeperiod(self.inc, opps.iloc[-1]['HID'], long_time)
        # number of matches is similar for all teams
        N = len(opps)
        min_bet = summary.iloc[0].to_dict()['Min_bet']
        max_bet = summary.iloc[0].to_dict()['Max_bet']
        bets = np.zeros((N, 2))

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
                data.at[idx, 'Team_form'] = DataExtraction().team_form(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Winrate'] = DataExtraction().winrate(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Draws'] = DataExtraction().draws(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Draws_rate'] = DataExtraction().draws_rate(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Scored_goals'] = DataExtraction().scored_goals(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Conceded_goals'] = DataExtraction().conceded_goals(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Shots_on_goal'] = DataExtraction().shots_on_goal(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Conceded_shots'] = DataExtraction().conceded_shots(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Penalty_minutes'] = DataExtraction().penalty_minutes(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Shots_on_goal'] = DataExtraction().shots_on_goal(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Goals_powerplay'] = DataExtraction().goals_powerplay(self.inc, row['Team'], no_matches_long)
                data.at[idx, 'Won_pucks'] = DataExtraction().won_pucks(self.inc, row['Team'], no_matches_long)
        """

        ##################################################################
        # CALCULATIONS THROUGHOUT THE SEASON
        ##################################################################
        for i, row in enumerate(opps.iterrows()):

            # BET ONLY LAST DAY BEFORE THE MATCH
            if DataExtraction().days_from_match(summary["Date"].loc[0], row[1]["Date"])==2:
                continue

            # BET BASED ON FORM
            prob = DataExtraction().match_final_prob_v2(self.inc, row[1]['HID'], row[1]['AID'],no_matches_short)
            bet = DataExtraction().bet(prob,[row[1]["OddsH"], row[1]["OddsA"]])
            bet = max_bet * np.array(bet)
            bet[bet<min_bet] = 0
            bet[bet>max_bet] = max_bet
            bets[i] = bet
            
        
        return pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)
