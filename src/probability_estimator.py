import pandas as pd


class ProbabilityEstimator:
    @staticmethod
    def __get_matches(team_a, team_b, inc):
        c2 = (inc['HID'] == team_a) & (inc['AID'] == team_b)
        c1 = (inc['HID'] == team_b) & (inc['AID'] == team_a)
        return inc.loc[c1 | c2]

    @staticmethod
    def get_probability(team_a, team_b, inc):
        teams_matches = ProbabilityEstimator.__get_matches(team_a, team_b, inc)
        tot = len(teams_matches)
        a_home_wins = teams_matches.loc[(teams_matches["HID"] == team_a) & teams_matches["H"]]
        a_away_wins = teams_matches.loc[(teams_matches["AID"] == team_a) & teams_matches["A"]]
        b_home_wins = teams_matches.loc[(teams_matches["HID"] == team_b) & teams_matches["H"]]
        b_away_wins = teams_matches.loc[(teams_matches["AID"] == team_b) & teams_matches["A"]]
        a_wins_cnt = len(a_home_wins) + len(a_away_wins)
        b_wins_cnt = len(b_home_wins) + len(b_away_wins)
        assert a_wins_cnt + b_wins_cnt <= tot
        if tot == 0:
            return -1
        return a_wins_cnt / tot
