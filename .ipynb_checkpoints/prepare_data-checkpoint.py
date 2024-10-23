import pandas as pd
import numpy as np
import nfl_data_py as nfl

from utils import viz_distro

DATA_PATH = './data/'
DATA_FILES = {
    'receiving':'agg_wr_final_10082024.csv',
    'def_points':'def_fpoints_10082024.csv',
    'def_injuries':'inj_defense_10082024.csv',
    'qb_stats':'qb_stats_10082024.csv'
}

DROP_COLUMNS = [
    'avg_cushion', 
    'avg_separation', 
    'twitter_username', 
    'receiving_drop', 
    'receiving_broken_tackles',
    'receiving_rat',
    'receiving_int',
    'receiving_drop_pct',
    'draft_year',
    'draft_round',
    'draft_pick',
    'draft_ovr',
    'target_share_4',
    'snap_percentage_4'
]

lag_columns = ['player_id', 'game_id', 'receiving_yards', 'avg_yac', 'receptions',
       'receiving_touchdowns', 'season', 'week', 'targets_1', 'targets_2', 'targets_3',
       'targets_4', 'total_targets', 'rz_targets', 'garbage_time_fpoints',
       'receiving_fpoints', 'avg_depth_of_target', 'air_yards',
       'max_target_depth', 'fumble_lost', 'receiving_first_downs',
       'receiving_epa', 'receiving_2pt_conversions', 'unrealized_air_yards',
       'racr', 'snap_count_1', 'snap_count_2', 'snap_count_3', 'snap_count_4',
       'total_relevant_snaps', 'snap_percentage_1',
       'snap_percentage_2', 'snap_percentage_3', 'snap_percentage',
       'target_share_1', 'target_share_2', 'target_share_3', 'target_share',
       'air_yards_share', 'wopr']

non_lag_columns = ['player_id', 
                   'game_id', 
                   'team', 
                   'week', 
                   'season',
                   'position', 
                   'receiving_fpoints', 
                   'age', 
                   'height', 
                   'weight', 
                   'depth_team', 
                   'opp_team',
                   'ESPN_projection']

non_numeric = [
    'player_id',
    'game_id',
    'status',
    'position',
    'player_name',
    'team',
    'twitter_username', 
    'college'
    'opp_team'
]

common_club_code_map =     {
        "LVR": "LV",
        "KCC": "KC",
        "NOS": "NO",
        "TBB": "TB",
        "SFO": "SF",
        "NEP": "NE",
        "GBP": "GB",
        "JAC": "JAX",
        "OAK":"LV",
        "STL":"LAR",
        "SL":"LAR",
        "SD":"LAC",
        "SDC":"LAC",
        "RAM":"LAR",
        "LA":"LAR",
        "BLT":"BAL",
        "HST":"HOU",
        "CLV":"CLE",
        "ARZ":"ARI"
    }

years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

def get_rosters(years):
    """ TODO: Include this in the data step... should not be here"""
    r_col_list = ["week", "position", "player_name", "player_id", "team", "status"]
    rosters = pd.DataFrame(columns=r_col_list)
    # this is due to a bug in the NFL data API. This function does not work when you pass it multiple years all at once
    for yr in years:
        t_roster = nfl.import_weekly_rosters(years=[yr])[r_col_list]
        t_roster["season"] = yr
        if yr < 2021:
            t_roster = t_roster[t_roster.week <= 17]
        else:
            t_roster = t_roster[t_roster.week <= 18]
        rosters = pd.concat([rosters, t_roster], ignore_index=True)

    rosters['team'] = rosters['team'].replace(common_club_code_map)

    # we need matchup data as well
    matchup_data = nfl.import_schedules(years)
    matchup_data["away_team"] = matchup_data["away_team"].replace(common_club_code_map)
    matchup_data["home_team"] = matchup_data["home_team"].replace(common_club_code_map)

    # here we grab unique game id's and home/away teams for each game in the season
    game_id_map = pd.concat(
        [
            matchup_data[["game_id", "home_team", "week", "season"]].rename(
                {"home_team": "team_abbr"}, axis=1
            ),
            matchup_data[["game_id", "away_team", "week", "season"]].rename(
                {"away_team": "team_abbr"}, axis=1
            ),
        ],
        ignore_index=False,
    )

    # we need depth charts data as well
    depth_charts = nfl.import_depth_charts(years=years)
    depth_charts["club_code"] = depth_charts["club_code"]
    depth_charts["depth_team"] = depth_charts["depth_team"].fillna(4)
    depth_charts["depth_team"] = depth_charts["depth_team"].astype(int)
    depth_charts = depth_charts[depth_charts['formation'] == 'Offense'].copy()
    depth_charts['depth_position'] = depth_charts.apply(lambda x: x['position'] if x['depth_position'].strip() == '' else x['depth_position'], axis = 1)

    # merges the depth chart data with the game id data from above
    depth_charts = depth_charts.merge(
        game_id_map,
        left_on=["week", "club_code", "season"],
        right_on=["week", "team_abbr", "season"],
    )

    # columns renames
    depth_charts = depth_charts.rename({"club_code": "team"}, axis=1)

    # merge rosters data
    rosters = rosters.merge(
        game_id_map,
        left_on=["week", "team", "season"],
        right_on=["week", "team_abbr", "season"],
    ).drop("team_abbr", axis=1)

    rosters = rosters[((rosters["status"] == "ACT") | (rosters["status"] == "RES")) & (rosters["week"] <= 18)]
    # this sorting will allow us to prioritize active players
    rosters = rosters.sort_values(by = ['game_id', 'player_id', 'status'], ascending = [True, True, True])
    rosters = rosters.drop_duplicates(subset = ['game_id', 'player_id', 'status'])

    temp = depth_charts[['position', 'depth_position', 'depth_team', 'game_id', 'gsis_id']].rename({'gsis_id':'player_id', 'position':'dc_position'}, axis=1)
    final = rosters.merge(temp, on = ['game_id', 'player_id'], how = 'left')
    # some extra filling
    final['depth_position'] = final.apply(lambda x: x['position'] if (x['depth_position'] != x['depth_position'] or x['depth_position'] is None) else x['depth_position'], axis = 1)
    final["depth_team"] = final["depth_team"].fillna(4)

    return final.sort_values(['game_id', 'team', 'position', 'depth_team', 'player_id'])
        

class PrepData():

    def __init__(self, window = True, time_series = False):
        
        self._window = window
        self._time_series = time_series

    def read_data(self, path = DATA_PATH, files = DATA_FILES):
        # load in the datasets
        self._raw_wr_stats = pd.read_csv(path + files['receiving'])
        self._raw_def_points_allowed = pd.read_csv(path + files['def_points'])
        self._raw_def_injuries = pd.read_csv(path + files['def_injuries'])
        self._raw_qb_stats = pd.read_csv(path + files['qb_stats'])

    def _subset(self, position = 'WR'):
        self._raw_wr_stats = self._raw_wr_stats[self._raw_wr_stats['position'] == position]

    def clean_data(self):
        # subset to WR
        self._subset()

        # rename some columns
        self._raw_def_points_allowed.rename({'defteam':'team'}, axis=1, inplace=True)

        # drop columns with a lot of nulls
        self._raw_wr_stats.drop(DROP_COLUMNS, axis=1, inplace = True)
        
        # there aren't many with missing collge/weight/height info, and they are not very impactful, I'm ok dropping them
        self._raw_wr_stats.drop(self._raw_wr_stats[self._raw_wr_stats.college.isna()].index, inplace = True)

        # let's also fill infinities
        self._raw_wr_stats.replace([np.inf, -np.inf], 0, inplace=True)

        # we just want to replace the team depth with the depth most often held by the player
        self._raw_wr_stats["depth_team"] = self._raw_wr_stats.groupby(["player_id", "season"])["depth_team"].transform(
            lambda x: x.fillna(x.mode()[0]) if len(x.mode()) > 0 else x.fillna(np.nan)
        )

        # the rest of these players are low on the depth chart so let's just fill with the max
        self._raw_wr_stats["depth_team"] = self._raw_wr_stats["depth_team"].fillna(self._raw_wr_stats.depth_team.max())


        # let's fill the rest with the player average for that season!
        null_stats = self._raw_wr_stats.isna().sum()/self._raw_wr_stats.shape[0]
        null_cols = null_stats[null_stats>0].sort_values(ascending=False).index.tolist()

        for c in null_cols:
            if c not in ['ESPN_projection'] + non_numeric:
                self._raw_wr_stats[c] = self._raw_wr_stats.groupby(["player_id", "season"])[c].transform(
                    lambda x: x.fillna(x.mean())
                )

        # if not, ten the player average for their career!
        for c in null_cols: 
            if c != ['ESPN_projection'] + non_numeric:
                self._raw_wr_stats[c] = self._raw_wr_stats.groupby(["player_id"])[c].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        # NOTE: This is likely due to the missing 2024 information for players
        # TODO: We could simulate this data with an LLM?
        # if not, the average accross all players
        for c in null_cols:
            if c != ['ESPN_projection'] + non_numeric:
                self._raw_wr_stats[c] = self._raw_wr_stats[c].fillna(self._raw_wr_stats[c].mean())

        print(f"""
        I dropped {len(DROP_COLUMNS)} columns from the dataset.
        I also ...
        """)

    def get_top_n(self, n = 40, roll_window = 2, viz = True):
        """ this function only preserves the top n players on a per week basis. helps us balance the dataset a bit better"""
        self._top_n = n
        self._roll_window = roll_window

        # viz distro
        if viz:
            viz_distro(df = self._raw_wr_stats, col = 'receiving_fpoints', label = 'post-top-n-filter')

        # sort the df by season, week, and score. Then keep only the top n.
        wrs = self._raw_wr_stats.sort_values(by = ['season', 'week', 'receiving_fpoints'], ascending = [True, True, False]).copy()
        wrs = wrs.groupby(['season', 'week']).head(self._top_n)
        
        
        # since we are doing rolling windows, I need the previous m weeks as well, where m is the length of the rolling window
        temp = pd.DataFrame(columns = self._raw_wr_stats.columns)
        for idx, row in wrs.iterrows():
            if row['week'] == 1:
                # week 1 has no previous weeks
                # TODO: How does this affect our model?
                continue
            else:
                week = row['week']
                # if we don't have enough previous weeks, then the first week we can append to the dataset is week 1
                if week - roll_window < 1:
                    final_week = 1
                else:
                    final_week = week - roll_window
                for pweek in range(week-1, final_week-1, -1):
                    # this check ensures that the week is not already in the dataet
                    if len(wrs[(wrs['season'] == row['season']) & \
                        (wrs['week'] == pweek) & \
                        (wrs['player_id'] == row['player_id'])]) == 0:
                        # otherwise, append the week from the raw stats onto the dataframe
                        temp = pd.concat([temp, self._raw_wr_stats[(self._raw_wr_stats['season'] == row['season']) & \
                            (self._raw_wr_stats['week'] == pweek) & \
                            (self._raw_wr_stats['player_id'] == row['player_id'])]], 
                            ignore_index = True)

        # gather new data into a dataframe
        self._wr_stats = pd.concat([wrs, temp], ignore_index = True).sort_values(by=['season', 'player_id', 'week']).drop_duplicates()

        # viz distro
        if viz:
            viz_distro(df = self._wr_stats, col = 'receiving_fpoints', label = 'post-top-n-filter')

    
    def window_data(self):
        # now we need to get the dataset to be averages of all datapoints aside from fantasy points before the current week.
        new_dataset = self._wr_stats[non_lag_columns]

        # this will grab the rolling average data for each player
        rolling_av = self._wr_stats[lag_columns].sort_values(by=['player_id', 'season', 'week'])
        rolling_av = rolling_av.set_index(['week', 'game_id']).groupby(['player_id', 'season']).rolling(self._roll_window, min_periods=self._roll_window).mean().reset_index().rename({'receiving_fpoints':'past_fpoints'}, axis=1)


        # this ensures that the new_week stat associates a players historical stats with the "current" week
        # NOTE: Those with a NaN new week are players who did not play past that "week"
        for idx, row in new_dataset.iterrows():
            week = row.week
            pid = row.player_id
            season = row.season

            # grab the most recent stats for that player, not just the previous week (sometimes players miss weeks)
            temp_week = rolling_av[(rolling_av.season == season) & (rolling_av.player_id == pid) & (rolling_av.week < week)].week.max()
            index = rolling_av[(rolling_av.season == season) & (rolling_av.player_id == pid) & (rolling_av.week == temp_week)].index
            rolling_av.loc[index, 'new_week'] = week

        print(f"The length of the dataset pre-windowing is: {len(new_dataset)}")

        # now we merge rolling_window stats
        self._windowed_data = new_dataset.merge(rolling_av.drop('game_id', axis=1), 
                                left_on=['player_id', 'season', 'week'], 
                                right_on=['player_id', 'season', 'new_week'], 
                                suffixes = ('', '_remove'))
        self._windowed_data.drop([x for x in self._windowed_data.columns if '_remove' in x], axis=1, inplace = True)
        self._windowed_data.dropna(subset = rolling_av.columns.tolist(), inplace = True)
        print(f"The length of the dataset post-windowing is: {len(self._windowed_data)}")

    def add_external_stats(self):

        # add season
        self._raw_def_points_allowed['season'] = self._raw_def_points_allowed['game_id'].apply(lambda x: int(x[:4]))
        self._raw_def_injuries['season'] = self._raw_def_injuries['game_id'].apply(lambda x: int(x[:4]))
        self._raw_qb_stats['season'] = self._raw_qb_stats['game_id'].apply(lambda x: int(x[:4]))

        # now we need to merge in average QBR, Average Defensive Rating, and Current Defensive Injuries
        self._raw_def_points_allowed.sort_values(by=['team','season','week'], inplace=True)
        self._raw_qb_stats.sort_values(by=['posteam', 'passer_player_id', 'season', 'week'], inplace=True)

        # now we get rolling stat windows as we always do
        # NOTE: remember that for each week, this is an average of that week and the week before, so before we merge onto the 
        # training dataset we need to add 1 to the week stat
        def_points_allowed_2 = self._raw_def_points_allowed.set_index(['game_id', 'week']).groupby(['team', 'season']).rolling(self._roll_window, min_periods=self._roll_window).mean().reset_index()

        # because we want to correlate the average of all past performances with the current 
        # def_points_allowed_2['week'] = def_points_allowed_2['week'] + 1
        for _, row in self._raw_def_points_allowed.iterrows():
            week = row.week
            team = row.team
            season = row.season

            # grab the most recent stats for that def, not just the previous week (no bye weeks)
            temp_week = def_points_allowed_2[(def_points_allowed_2.season == season) & (def_points_allowed_2.team == team) & (def_points_allowed_2.week < week)].week.max()
            index = def_points_allowed_2[(def_points_allowed_2.season == season) & (def_points_allowed_2.team == team) & (def_points_allowed_2.week == temp_week)].index
            def_points_allowed_2.loc[index, 'new_week'] = week

        # for QB's we need to do a little extra
        # 1. Similar to the WR stats before, we need to ensure that the current week is associated with the most recently available historical stat. 
        # 2. We need to know if the starter has changed from the previous week
        # 3. We need to get the associated QB stats for the CURRENT starter, not whoever started last week. 
        # 3. NOTE: Should we weight the QBR by recency for that QB? E.g. if the QB hasn't started for 3 weeks their weighted QBR is lower than if they
        #          started last week.

        # for every QB, this is their rolling window
        qb_stats_2 = self._raw_qb_stats.set_index(['game_id', 'week', 'posteam']).groupby(['passer_player_id', 'season']).rolling(self._roll_window, min_periods=self._roll_window).mean().reset_index()
        qb_stats_2.rename({'passer_player_id':'player_id'}, axis=1, inplace=True)
        # for every team/week, this is their starting QB
        # 1. raw roster data
        rosters = get_rosters(years = years)
        # 2. grab QB 1 per week
        temp = rosters[rosters['depth_position'] == 'QB']
        qb_1 = temp.loc[temp.groupby(['game_id', 'team'])['depth_team'].idxmin()][['game_id', 'team', 'player_id', 'season', 'week']]
        # 3. merge onto previous data for that QB from qb_stats_2
        # this ensures that the new_week stat associates a players historical stats with the "current" week
        # NOTE: Those with a NaN new week are players who did not play past that "week"
        for _, row in qb_1.iterrows():
            week = row.week
            pid = row.player_id
            season = row.season

            # grab the most recent stats for that player, not just the previous week (sometimes players miss weeks)
            temp_week = qb_stats_2[(qb_stats_2.season == season) & (qb_stats_2.player_id == pid) & (qb_stats_2.week < week)].week.max()
            index = qb_stats_2[(qb_stats_2.season == season) & (qb_stats_2.player_id == pid) & (qb_stats_2.week == temp_week)].index
            qb_stats_2.loc[index, 'new_week'] = week

        # 5. loop through starter data, fill is_new_qb, needs to be done before window merge
        qb_1_temp = qb_1[['season', 'team', 'player_id', 'week']].copy()
        # this will associate the previous starter with this weeks starter
        qb_1_temp['week'] = qb_1_temp['week'] + 1
        # rename so we can compare
        qb_1_temp = qb_1_temp.rename({'player_id': 'prev_player_id'}, axis = 1)
        # merge into current starter list
        qb_1 = qb_1.merge(qb_1_temp, on = ['season', 'team', 'week'], how = 'left')
        # flag new starters
        qb_1['is_new_qb_starter'] = qb_1.apply(lambda x: 0 if ((x['player_id'] == x['prev_player_id']) or (x['week'] == 1)) else 1, axis=1)
        qb_1.drop('prev_player_id', axis=1, inplace = True)
        
        # TODO: Fill in nulls with ALL historical av for that QB, 
        # if not, fill with season historical average for that team, if not, previous season average, if not, global average
        
        # now we merge rolling_window stats
        # NOTE: A left join here will preserve the QBID for future processing
        qb_1 = qb_1.merge(qb_stats_2.drop('game_id', axis=1), 
                                left_on=['player_id', 'season', 'week'], 
                                right_on=['player_id', 'season', 'new_week'], 
                                suffixes = ('', '_remove'),
                                how = 'left')
        qb_1.drop([x for x in qb_1.columns if '_remove' in x], axis=1, inplace = True)
        # qb_1.dropna(subset = qb_stats_2.columns.tolist(), inplace = True)
        qb_rename_map = {'completions':'hist_qb_completions', 'attempts':'hist_qb_attempts', 'passing_yards':'hist_qb_passing_yards',
       'touchdowns':'hist_qb_touchdowns', 'interceptions':'hist_qb_interceptions', 'completion_percentage':'hist_qb_completion_percentage',
       'yards_per_attempt':'hist_qb_yards_per_attempt', 'td_percentage':'hist_qb_td_percentage', 'interception_percentage':'hist_qb_interception_percentage', 
       'QBR':'hist_qb_QBR','qb_num_snaps':'hist_qb_num_snaps', 'player_id':'qb_player_id'}
        qb_1.rename(qb_rename_map, axis=1, inplace = True)
        qb_stat_cols = ['team', 'game_id'] + list(qb_rename_map.values())
        

        # merge in with training data
        # FIXME: Fix merge columns, use game_id and team
        self._windowed_data = self._windowed_data.merge(def_points_allowed_2[['new_week', 'season', 'team', 'total_qb_fpoints_given_up', 'total_wr_fpoints_given_up']], 
                            how = 'left', 
                            left_on=['opp_team', 'week', 'season'],
                            right_on=['team', 'new_week', 'season'], 
                            suffixes=('', '_remove'))
        
        
        self._windowed_data = self._windowed_data.merge(self._raw_def_injuries[['week', 'season', 'team', 'num_injured_starters']], 
                            how = 'left', 
                            left_on=['opp_team', 'week', 'season'], 
                            right_on=['team', 'week', 'season'], 
                            suffixes=('', '_remove')) \
                .rename({'num_injured_starters': 'def_inj_starters'}, axis = 1)
        
        # merge in QB stats
        self._windowed_data = self._windowed_data.merge(qb_1[qb_stat_cols],
                            how = 'left', 
                            left_on=['game_id', 'team'], 
                            right_on=['game_id', 'team'], 
                            suffixes=('', '_remove'))
        # remove all  remove columns
        self._windowed_data.drop([i for i in self._windowed_data.columns if 'remove' in i],
                       axis=1, inplace=True)
        

        # now if we don't have QB info filled, we have some backup methods: 
        for idx, row in self._windowed_data.iterrows():
            if np.isnan(row['hist_qb_td_percentage']):
                # try filling with global qb average that season
                season = row['season']
                week = row['week']
                pid = row['qb_player_id']

                # season averages (player)
                temp_qb = self._raw_qb_stats[(self._raw_qb_stats['passer_player_id'] == pid) & (self._raw_qb_stats['season'] == season) & (self._raw_qb_stats['week'] < week)].copy()
                temp_qb.drop(['passer_player_id', 'season', 'week', 'game_id', 'posteam'], axis=1, inplace = True)
                
                if not temp_qb.empty:
                    if temp_qb.shape[0] > 1:
                        new_qb_df = temp_qb.mean().to_frame().T
                    else:
                        new_qb_df = temp_qb
                    new_qb_df.rename(qb_rename_map, axis=1, inplace = True)

                    for col in list(qb_rename_map.values())[:-1]:
                        self._windowed_data.loc[idx, col] = new_qb_df[col].values[0]
                    continue
                
                # global average (player)
                temp_qb = self._raw_qb_stats[(self._raw_qb_stats['passer_player_id'] == pid) & (self._raw_qb_stats['season'] <= season) & (self._raw_qb_stats['week'] < week)].copy()
                temp_qb.drop(['passer_player_id', 'season', 'week', 'game_id', 'posteam'], axis=1, inplace = True)

                if not temp_qb.empty:
                    if temp_qb.shape[0] > 1:
                        new_qb_df = temp_qb.mean().to_frame().T
                    else:
                        new_qb_df = temp_qb
                    new_qb_df.rename(qb_rename_map, axis=1, inplace = True)

                    for col in list(qb_rename_map.values())[:-1]:
                        self._windowed_data.loc[idx, col] = new_qb_df[col].values[0]
                    continue  

                # season average (all players)
                temp_qb = self._raw_qb_stats[(self._raw_qb_stats['season'] == season) & (self._raw_qb_stats['week'] < week)].copy()
                temp_qb.drop(['passer_player_id', 'season', 'week', 'game_id', 'posteam'], axis=1, inplace = True)

                if not temp_qb.empty:
                    if temp_qb.shape[0] > 1:
                        new_qb_df = temp_qb.mean().to_frame().T
                    else:
                        new_qb_df = temp_qb
                    new_qb_df.rename(qb_rename_map, axis=1, inplace = True)
                    for col in list(qb_rename_map.values())[:-1]:
                        self._windowed_data.loc[idx, col] = new_qb_df[col].values[0]

                    continue  
                print('why are we here')

