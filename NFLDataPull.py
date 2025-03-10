import nfl_data_py as nfl
import pandas as pd
import numpy as np

from utils import patched_read_parquet

class DataCreator():
    def __init__(self, years: list, paradigm:str = None):
        self._years = years

        # this is for grabbing the right type of data among: rushing, passing, receiving
        if paradigm is not None:
            self._paradigm = paradigm
        else:
            self._paradigm = "all"

        # FIXME: Temp fix for nfl_data_py
        # this code avoids a numpy error. nfl_data_py will have a new version soon to handle new python and numpy versions
        _read_parquet = pd.read_parquet
        pd.read_parquet = patched_read_parquet
        

    def load_data(self):
        # this maps player ID's from accross multiple platforms. There is also associated player information like Name, College, etc. 
        self.id_map = nfl.import_ids()

        # we want to have depth charts as depth is likely a good predictor of fantasy value. 1st string is utilized more than 3rd string
        self.depth_charts = nfl.import_depth_charts(years = self._years)

        # importing next gen stats data from AWS
        self.ngs_receiving = nfl.import_ngs_data(stat_type = 'receiving', years = self._years).drop_duplicates()

        # pro football reference stats
        self.pfr_receiving = nfl.import_weekly_pfr(s_type = "rec", years = self._years).drop_duplicates()

        # # weekly aggregate stats
        self.weekly = nfl.import_weekly_data(years = self._years, downcast=True).drop_duplicates()

        # I need to standardize the game ID and player ID's accross the dataframes
        self.matchup_data = nfl.import_schedules(self._years)

        # grab play-by-play data to synthesize some base stats. Adding in stats from other platforms after the fact.
        # self.pbp_data = nfl.import_pbp_data(years = self._years)
        self.__load_pbp_data()

        # importing rushing next gen stats data
        self.ngs_rush = nfl.import_ngs_data(stat_type = 'rushing', years = self._years).drop_duplicates()

        # rushing pro football reference stats
        self.pfr_rush = nfl.import_weekly_pfr(s_type = "rush", years = self._years).drop_duplicates()

    def __load_pbp_data(self):
        # TODO: This will fix itself in future iterations of the nfl_data_py project
        if 2024 in self._years:
            no_2024 = [x for x in self._years if x != 2024]
            pbp_data = nfl.import_pbp_data(years=no_2024)
            
            # 2024 needs to be done with a separate flag
            temp_2024_pbp = nfl.import_pbp_data(years=[2024], include_participation=False)
            for c in [c for c in pbp_data.columns.tolist() if c not in temp_2024_pbp.columns.tolist()]:
                temp_2024_pbp[c] = np.nan
            
            self.pbp_data = pd.concat([pbp_data, temp_2024_pbp[pbp_data.columns.tolist()]], ignore_index = True)
        else: 
            self.pbp_data = nfl.import_pbp_data(years = self._years)


    def __load_rosters(self):
        pass

    def clean_data(self):
        # fixing Conklin/Izzo issue
        self.id_map['gsis_id'] = self.id_map['gsis_id'].replace({'00-0034439': '00-0034270', '00-0034270':'11-1111111'})

        # we also need to map non-conventianal city abreviations for consistency
        self.id_map['team'] = self.id_map['team'].replace({
            'LVR':'LV',
            'KCC':'KC',
            'NOS':'NO',
            'TBB':'TB',
            'SFO':'SF',
            'NEP':'NE',
            'LAR':'LA',
            'GBP':'GB',
            'JAC':'JAX'
        })

        # here we grab unique game id's and home/away teams for each game in the season
        self.game_id_map = pd.concat([self.matchup_data[['game_id', 'home_team', 'week']].rename({'home_team':'team_abbr'}, axis=1), self.matchup_data[['game_id', 'away_team', 'week']].rename({'away_team':'team_abbr'}, axis=1)], ignore_index=False)

        # merges the AWS data with the game id data from above
        self.ngs_receiving = self.ngs_receiving.merge(self.game_id_map, left_on=['week', 'team_abbr'], right_on = ['week', 'team_abbr']).rename({'player_gsis_id':'player_id'},axis=1)

        # merges the PFR data with the game id data from above
        self.pfr_receiving = self.pfr_receiving.merge(self.id_map[['pfr_id', 'gsis_id']], left_on='pfr_player_id', right_on = 'pfr_id').rename({'gsis_id':'player_id'},axis=1)
        
        # NOTE: The weekly data does not have the opponent team info for 2022. Thus we would need different logic to grab the game_id. 
        # We actually end up creating all the stats from this dataset by hand so there is no need.

        # merges the depth chart data with the game id data from above
        self.depth_charts = self.depth_charts.merge(self.game_id_map, left_on=['week', 'club_code'], right_on = ['week', 'team_abbr'])

        # since we are going to aggregate the play-by-play data to game-by-game data, we want to fix some features so we can count easier
        d = {}
        # this just makes a countable two point conversion field. One-hot encoding if you will
        d['is_two_point_conversion'] = self.pbp_data['two_point_conv_result'].apply(lambda x: 1 if x == 'success' else 0)
        self.pbp_data = pd.concat([self.pbp_data, pd.DataFrame(d)], axis=1)

        # we're only going to consider the regular season
        self.pbp_data = self.pbp_data[self.pbp_data.week <= 18]

        # just make sure that plays that don't result in a touchdown have data here. Another one-hot (ish)
        self.pbp_data['touchdown'] = self.pbp_data['touchdown'].fillna(0)
        self.pbp_data['interception'] = self.pbp_data['interception'].fillna(0)
        self.pbp_data['fumble_lost'] = self.pbp_data['fumble_lost'].fillna(0)
        self.pbp_data['passing_yards'] = self.pbp_data['passing_yards'].fillna(0)
        self.pbp_data['pass_touchdown'] = self.pbp_data['pass_touchdown'].fillna(0)
        self.pbp_data['rushing_yards'] = self.pbp_data['rushing_yards'].fillna(0)
        self.pbp_data['rush_touchdown'] = self.pbp_data['rush_touchdown'].fillna(0)
        self.pbp_data['receiving_yards'] = self.pbp_data['receiving_yards'].fillna(0)
        # this creates a list of offensive players rather than a string. This helps later when we look at snaps played by each player
        # NOTE: Some of these are missing. 
        self.pbp_data['offense_players'] = self.pbp_data['offense_players'].apply(lambda x: x.split(';') if type(x) == str else x)

        # we are also going to restrict to data that we need in the pbp_data, this will make it easier to look at
        rel_cols_pbp = [
            'play_id',
            'game_id',
            'home_team',
            'away_team',
            'week',
            'posteam',
            'defteam',
            'yardline_100',
            'game_date',
            'game_seconds_remaining',
            'qtr',
            'down',
            'time',
            'desc',
            'play_type',
            'yards_gained',
            'air_yards',
            'yards_after_catch',
            'score_differential',
            'epa',
            'incomplete_pass',
            'interception',
            'penalty',
            'rush_attempt',
            'pass_attempt',
            'touchdown',
            'pass_touchdown',
            'rush_touchdown',
            'two_point_attempt',
            'fumble',
            'fumbled_1_team',
            'fumbled_1_player_id',
            'fumbled_1_player_name',
            'fumbled_2_player_id',
            'fumbled_2_player_name',
            'fumbled_2_team',
            'fumble_lost',
            'complete_pass',
            'passer_player_id',
            'passer_player_name',
            'passing_yards',
            'receiver_player_id',
            'receiver_player_name',
            'receiving_yards',
            'rusher_player_id',
            'rusher_player_name',
            'rushing_yards',
            'fumbled_1_player_id',
            'fumbled_2_player_id',
            'penalty_player_id',
            'penalty_yards',
            'replay_or_challenge',
            'replay_or_challenge_result',
            'penalty_type',
            'offense_players',
            'players_on_play',
            'timeout',
            'is_two_point_conversion',
            'first_down'
        ]

        self.pbp_data = self.pbp_data[rel_cols_pbp].copy()

    def engineer_features(self):

        # get fumbled player
        self.pbp_data['receiver_fumble_lost'] = self.pbp_data.apply(lambda x: self.__get_fumble(x, ptype='rec'), axis=1)
        self.pbp_data['rusher_fumble_lost'] = self.pbp_data.apply(lambda x: self.__get_fumble(x, ptype='rush'), axis=1)
        self.pbp_data['passer_fumble_lost'] = self.pbp_data.apply(lambda x: self.__get_fumble(x, ptype='pass'), axis=1)

        
    
    def __get_fumbles(self, x, ptype = 'rec'):

        if ptype == 'rec':
            rpid = x['receiver_player_id']
        elif ptype == 'rush':
            rpid = x['rusher_player_id']
        elif ptype == 'pass':
            rpid = x['passer_player_id']

        fpid1 = x['fumbled_1_player_id'].values[0]
        fpid2 = x['fumbled_2_player_id'].values[0]
        fteam1 = x['fumbled_1_team']
        fteam2 = x['fumbled_2_team']

        if fpid1 is None:
            fpid1 = ''
        
        if fpid2 is None:
            fpid2 = ''
        
        if fteam1 is None:
            fteam1 = ''
        
        if fteam2 is None:
            fteam2 = ''

        offense = x['posteam']
        # if there was a fumble and the receiver we are looking at was one of the fumble players
        if x['fumble_lost'] == 1 and (fpid1 == rpid or fpid2 == rpid):
            # the only time this is not a fumble for the player in consideration is when
            # there are two fumbles by the offense and the second player is not the first player.
            # e.g. player 1 fumbled and it was picked up by teammate player 2, then player 2 fumbles
            # and it is recovered by the defense
            if (fpid1 == rpid and fpid2 != rpid) and \
                (fteam1 == offense and fteam2 == offense):
                return 0
            else:
                return 1
        else:
            return 0
