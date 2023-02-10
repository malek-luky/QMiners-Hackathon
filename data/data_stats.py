import pandas
import seaborn
import datetime
import numpy as np
import matplotlib

class Dataset:
    def __init__(self, file_name) -> None:
        self.data = pandas.read_csv(file_name)
    
    def get_teams(self):
        home, arriving = self.data["HID"], self.data["HID"]
        return pandas.unique(pandas.concat([home, arriving]))
    
    def get_max_date(self):
        pass

    def get_min_date(self):
        pass
    
def frequency_of_matches(data, teams):
    plot_data = pandas.DataFrame(columns=["date", "team_id"])

    for team in teams:
        # all matches where the given team is playing
        matches = data[(data["HID"] == team) | (data["AID"] == team)]
        for match_date in matches["Date"]:
            # vsetkty mesiace majú 31 dní lebo to nerobí rozdie:D
            date = datetime.date.fromisoformat(match_date)
            total_hours = date.year*365+date.month*31+date.day
            plot_data.loc[len(plot_data.index)] = [total_hours, team] 
    seaborn.scatterplot(data=plot_data, x="date", y="team_id", hue="team_id", alpha=.6, palette="dark")
    matplotlib.pyplot.show()


def win_rate_at_home(data, teams):
    plot_data = pandas.DataFrame(columns=["win", "team_id", "home"])

    for team in teams:
        # playing at home
        for i, match in data[(data["HID"] == team)].iterrows():
            plot_data.loc[len(plot_data.index)] = [int(match["H"]), team, True] 

        # playing away
        for i, match in data[(data["AID"] == team)].iterrows():
            plot_data.loc[len(plot_data.index)] = [int(match["A"]), team, False] 
    seaborn.barplot(data=plot_data, x="team_id", y="win", hue="home", alpha=.6, palette="dark")
    matplotlib.pyplot.show()

    
def tiredness(data, teams):
    last_match = dict.fromkeys(range(30), None)
    plot_data = pandas.DataFrame(columns=["shots", "days_after_last_match"])
    for _, match in data.iterrows():
        date = datetime.date.fromisoformat(match["Date"])
        lm_h, lm_a = last_match[match["HID"]], last_match[match["AID"]]
        if lm_h and lm_a:
            delta_h = (date - lm_h).days
            delta_a = (date - lm_a).days
            # if delta_h <=5: plot_data.loc[len(plot_data.index)] = [int(match["H"]), delta_h]
            # if delta_a <=5: plot_data.loc[len(plot_data.index)] = [int(match["A"]), delta_a]
            if delta_h <=5: plot_data.loc[len(plot_data.index)] = [int(match["HSC"]), delta_h]
            if delta_a <=5: plot_data.loc[len(plot_data.index)] = [int(match["ASC"]), delta_a]
        last_match[match["HID"]] = date
        last_match[match["AID"]] = date
    seaborn.stripplot(data=plot_data, x="days_after_last_match", y="shots", alpha=.6, palette="dark")
    # seaborn.barplot(data=plot_data, x="days_after_last_match", y="win", alpha=.6, palette="dark")
    matplotlib.pyplot.show()


def wins_to_something(data, teams):
    team_stats = {}
    for i in range(30): team_stats[i] = {"wins": 0, "shots": 0, "won_buly": 0, "penalty": 0}
    for _, match in data.iterrows():
        if match["H"]: wt = "H"
        else: wt = "A"
        team = match[f"{wt}ID"]
        team_stats[team]["wins"] += 1 
        team_stats[team]["shots"] += match[f"S_{wt}"]
        team_stats[team]["won_buly"] += match[f"FOW_{wt}"]
        team_stats[team]["penalty"] += match[f"PIM_{wt}"]
    
    plot_data = pandas.DataFrame.from_dict(team_stats, orient="index")
    print(plot_data)
    seaborn.scatterplot(data=plot_data, x="won_buly", y="wins", alpha=.6, palette="dark", hue=plot_data.index)
    matplotlib.pyplot.show()


def pairplot(data, team_id):
    data_frame = data
    seaborn.pairplot(data_frame[["OddsH", "FOW_H", "S_H", "PIM_H", "H"]], hue="H")
    matplotlib.pyplot.show()



seaborn.set(style="darkgrid")
data = Dataset("training_data.csv")
teams = data.get_teams()

# frequency_of_matches(data.data, teams)
# win_rate_at_home(data.data, teams)
# pairplot(data.data, 1)
# tiredness(data.data, teams)
wins_to_something(data.data, teams)