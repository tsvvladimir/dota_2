import pandas as pd
import trueskill

from math import sqrt, log
from trueskill import BETA
from trueskill.backends import cdf

matches_info = pd.read_csv('selected_team_matches.csv').sort(['match_id'])

trueskill.setup(draw_probability=0)

Rates = {}
for index, row in matches_info.iterrows():
    team1 = row['radiant']
    team2 = row['dire']
    if team1 not in Rates:
        Rates[team1] = trueskill.Rating()
    if team2 not in Rates:
        Rates[team2] = trueskill.Rating()
    if row['winner'] == 'RADIANT':
        Rates[team1], Rates[team2] = trueskill.rate_1vs1(Rates[team1], Rates[team2])
    else:
        Rates[team2], Rates[team1] = trueskill.rate_1vs1(Rates[team2], Rates[team1])

print Rates['VP 2']

def win_probability(player_rating, opponent_rating):
    delta_mu = player_rating.mu - opponent_rating.mu
    denom = sqrt(2 * (BETA * BETA) + pow(player_rating.sigma, 2) + pow(opponent_rating.sigma, 2))
    return cdf(delta_mu / denom)

print win_probability(Rates['VP 2'], Rates['EG'])