import pandas as pd

import sys
sys.path.append(".")

# from model_lukas import Model #TODO: Change here for different model
#from model import Model #TODO: Change here for different model
#from ben_bookmaker_predict import Model #TODO: Change here for different model
#from alll_lukas import Model #TODO: Change here for different model
from d2 import Model #TODO: Change here for different model
#from model_dummy import Model
from environment import Environment

dataset = pd.read_csv('../data/training_data.csv', parse_dates=['Date', 'Open'])
model = Model()
env = Environment(dataset, model, init_bankroll=1000., min_bet=5., max_bet=100.)
evaluation = env.run(start=pd.to_datetime('2014-08-01'))

print(f'Final bankroll: {env.bankroll:.2f}')

