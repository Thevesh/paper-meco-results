import pandas as pd
import numpy as np
import json as j
import os
import time
from dotenv import load_dotenv
from requests_toolbelt import MultipartEncoder
import requests

from helper import generate_slug, get_states, write_parquet

load_dotenv()
PATH = 'src-data/dashboards'

nm = pd.read_csv('src-data/lookup_candidates.csv').drop_duplicates(subset=['candidate_uid'],keep='last')
MAP_NAME = dict(zip(nm.candidate_uid,nm.name))


# Note: Candidates forms the base for Seat Search and Party Search, as well as Parties / Ballots
def make_candidates():
    """Generate candidates dataframe with election results and metadata.

    Reads candidate data from ballot and summary files, maps candidate UIDs to names,
    adds seat and election metadata, and calculates vote percentages and majorities.

    The function:
    1. Maps candidate UIDs to standardized names from lookup_candidates.csv
    2. Adds seat metadata (type, state) and election details
    3. Merges with summary statistics (turnout, rejected votes, majorities)
    4. Writes final dataframe to parquet format

    Returns:
        Number of rows in the final dataframe, representing the number of candidates across all elections
    """

    COL_BALLOT = ['slug','name','type','date','election_name','seat','party','votes','votes_perc','result']

    df = pd.read_parquet('src-data/consol_ballots.parquet').rename(columns={'election':'election_name'})
    df['seat'] = df['seat'] + ', ' + df['state']
    df = df.drop(['state'],axis=1)
    assert len(df[~df.candidate_uid.isin(MAP_NAME.keys())]) == 0, f'Candidate UIDs not found in name map! {df[~df.candidate_uid.isin(MAP_NAME.keys())].candidate_uid.unique()}'
    df['name'] = df['candidate_uid'].map(MAP_NAME)
    df['slug'] = df['candidate_uid'].astype(str).str.zfill(5)
    df.election_name = df.election_name.str.replace('BY-ELECTION','By-Election')
    df['type'] = 'parlimen'
    df.loc[df.seat.str.startswith('N.'),'type'] = 'dun'
    df = df[COL_BALLOT]

    COL_SUMMARY = ['date','election_name','seat','voter_turnout','voter_turnout_perc','votes_rejected','votes_rejected_perc','majority','majority_perc']
    tf = pd.read_parquet('src-data/consol_summary.parquet')\
        .drop(['votes_valid','ballots_not_returned_perc','voters_total'],axis=1)\
        .rename(columns={'ballots_issued':'voter_turnout','voter_turnout':'voter_turnout_perc','election':'election_name'})
    tf['seat'] = tf['seat'] + ', ' + tf['state']
    tf = tf.drop(['state'],axis=1)[COL_SUMMARY]
    tf.election_name = tf.election_name.str.replace('BY-ELECTION','By-Election')

    df = pd.merge(df,tf,on=['date','election_name','seat'],how='right')
    write_parquet(f'{PATH}/elections_candidates',df=df)

    return len(df)


# generate Seat Search, then map majority to Candidate file
def make_seats():
    """Generate seat search dataframe with election results and metadata.

    Reads seat winner data from seat winner file, filters out By-Elections,
    and selects relevant columns.

    The function:
    1. Reads seat winner data from seat winner file
    2. Merges with summary statistics (turnout, rejected votes, majorities)
    3. Writes final dataframe to parquet format

    Returns:
        Number of rows in the final dataframe, representing the number of seats in the dataset
    """
    COL_WINNER = ['slug','seat_name','seat','state','date','election_name','type','party','name']

    lf = pd.read_csv('src-data/lookup_seats.csv')
    MAP_AREA_NAME = dict(zip(zip(lf.election, lf.state, lf.seat), lf.area_name))

    df = pd.read_parquet(f'src-data/consol_ballots.parquet').rename(columns={'election':'election_name'})
    df['name'] = df['candidate_uid'].map(MAP_NAME)
    assert len(df[df.name.isnull()]) == 0, f'Candidate name not found in name map! {df[df.name.isnull()].candidate_uid.unique()}'
    df['type'] = 'parlimen'
    df.loc[df.seat.str.startswith('N.'),'type'] = 'dun'
    df['seat_name'] = df.apply(lambda x: MAP_AREA_NAME.get((x['election_name'], x['state'], x['seat'])), axis=1) + ', ' + df.state
    df['slug'] = df.type.str[:1] + '-' + df.seat_name.apply(generate_slug)
    df.election_name = df.election_name.str.replace('BY-ELECTION','By-Election')
    df = df[df.result.str.contains('won')].copy().drop('result',axis=1)[COL_WINNER]
    assert len(df) == len(df.drop_duplicates(subset=['date','slug'])), 'Duplicate seats!!'

    COL_SUMMARY = ['date','state','seat','majority','majority_perc','voter_turnout','voter_turnout_perc','votes_rejected','votes_rejected_perc']
    sf = pd.read_parquet('src-data/consol_summary.parquet').rename(columns={'voter_turnout':'voter_turnout_perc','ballots_issued':'voter_turnout'})
    sf = sf[COL_SUMMARY]

    df = pd.merge(df,sf,on=['date','state','seat'],how='left')
    assert len(df[df.majority.isnull()]) == 0, f'Imperfect join!! {df[df.majority.isnull()]}'
    df.slug = df.slug.str[2:]
    df['seat'] = df.seat + ', ' + df.state # very important! this is used in the API query
    write_parquet(f'{PATH}/elections_seats_winner',df=df)

    return len(df)


# generate Party Search
def make_parties():
    """Generate party search dataframe with election results and metadata.

    Reads candidate data from candidate file, drops unnecessary columns,
    filters out By-Elections, and selects relevant columns.

    The function:
    1. Reads candidate data from candidate file
    2. Drops unnecessary columns
    3. Filters out By-Elections
    4. Selects relevant columns
    5. Writes final dataframe to parquet format

    Returns:
        Number of rows in the final dataframe, representing the number of seats in the dataset
    """
    col_idx = ['party','type','state','election_name','date']
    df = pd.read_parquet(f'{PATH}/elections_candidates.parquet').rename(columns={'seat':'state'})
    df = df[df.election_name != 'By-Election'] # Remove By-Elections, we are not interested in them
    df = df.drop(['voter_turnout','voter_turnout_perc','votes_rejected','votes_rejected_perc','majority','majority_perc'],axis=1)
    df['state'] = df['state'].apply(lambda x: x.split(',')[1].strip())
    df['seats'] = 0
    df.loc[df.result.str.contains('won'),'seats'] = 1
    df = df.drop(['name','votes_perc','result','slug'],axis=1).groupby(col_idx).sum().reset_index()

    # number of seats and votes per election, by state (sf)
    col_idx_sf = ['election_name','state']
    sf = df[col_idx_sf + ['votes','seats']].copy().groupby(col_idx_sf).sum().reset_index()
    sf.columns = col_idx_sf + ['votes_total','seats_total']
    df = pd.merge(df,sf,on=col_idx_sf,how='left')
    df['votes_perc'] = df.votes / df.votes_total * 100
    df['seats_perc'] = df.seats / df.seats_total * 100

    # number of seats and votes per election, for Malaysia (dfm)
    dfm = df[df.columns[:7]].copy().assign(state='Malaysia').groupby(col_idx).sum().reset_index()
    dfm = pd.merge(dfm,sf.groupby('election_name').sum(numeric_only=True).reset_index(),on='election_name',how='left')
    dfm['votes_perc'] = dfm.votes / dfm.votes_total * 100
    dfm['seats_perc'] = dfm.seats / dfm.seats_total * 100

    df.loc[(df.election_name == 'SE-02') & (df.state == 'Sabah'),'votes_perc'] = df.seats_perc # special case where all seats were uncontested
    df = pd.concat([dfm,df],ignore_index=True)[col_idx + ['seats','seats_total','seats_perc'] + ['votes','votes_perc']]

    return len(df)


# generate Dates master file
def make_dates():
    """Generate dates dataframe with election dates and metadata.

    Reads election dates from lookup_dates.csv, standardizes election names,
    and combines state-level and national-level elections.

    The function:
    1. Reads election dates from lookup_dates.csv
    2. Standardizes election names to 'SE-XX' format
    3. Combines state-level and national-level elections
    4. Writes final dataframe to parquet format

    Returns:
        Number of rows in the final dataframe, representing the number of elections in the dataset
    """
    df = pd.read_csv('src-data/lookup_dates.csv').rename(columns={'election_number':'election'})
    df.election = 'SE-' + df.election.astype(str).str.zfill(2)
    df.loc[df.state == 'Malaysia','election'] = df.election.str.replace('SE','GE')
    df = pd.concat([df[df.state == 'Malaysia'],df[df.state != 'Malaysia']],axis=0,ignore_index=True)
    write_parquet('src-data/dashboards/elections_dates',df=df)

    return len(df)


# generate Election veterans
def make_veterans():
    """Generate veteran candidates dataframe with election results and metadata.

    Reads candidate data from candidate file, filters out By-Elections,
    and selects relevant columns.

    The function:  
    1. Reads candidate data from candidate file
    2. Filters out By-Elections
    3. Selects relevant columns
    4. Writes final dataframe to parquet format

    Returns:
        None.
    """
    states = get_states()

    df = pd.read_parquet(f'{PATH}/elections_candidates.parquet')
    df['state'] = df['seat'].apply(lambda x: x.split(',')[1].strip())
    df['competed'] = 1
    df['won'] = 0
    df.loc[df.result.str.contains('won'),'won'] = 1

    gv = df[df.type == 'parlimen'].copy()[['state','date','name','competed','won']]
    sv = df[df.type == 'dun'].copy()[['state','date','name','competed','won']]

    def get_top_10(tf,state='Malaysia'):
        tf = tf[['name','competed','won']].groupby('name').sum().reset_index()\
        .sort_values(by=['competed','won'],ascending=False).reset_index(drop=True)
        tf = tf[(tf.competed >= tf.competed.iloc[9])\
                & ((tf.won >= tf.won.iloc[9]))\
                & ((tf.won + tf.competed >= 2))]
        tf = tf.assign(state=state)[['state','name','competed','won']].reset_index(drop=True)
        while len(tf) > 15: tf = tf[(tf.competed > tf.competed.iloc[9])]
        return tf

    resp = get_top_10(gv)
    resd = get_top_10(sv)

    for s in states:
        tfp = gv[gv.state == s].copy()
        if len(tfp) > 0: resp = pd.concat([resp,get_top_10(tfp,state=s)],axis=0,ignore_index=True)
        
        tfd = sv[sv.state == s].copy()
        if len(tfd) > 0: resd = pd.concat([resd,get_top_10(tfd,state=s)],axis=0,ignore_index=True)

    write_parquet(f'{PATH}/elections_veteran_parlimen',df=resp)
    write_parquet(f'{PATH}/elections_veteran_dun',df=resd)

    df = pd.DataFrame(columns=['type'])
    df = pd.concat([
        df,
        pd.read_parquet(f'{PATH}/elections_veteran_parlimen.parquet').assign(type='parlimen'),
        pd.read_parquet(f'{PATH}/elections_veteran_dun.parquet').assign(type='dun'),
    ],axis=0,ignore_index=True)
    for c in ['competed','won']: df[c] = df[c].astype(int)
    write_parquet(f'{PATH}/elections_veterans',df=df)


# generate closest and biggest victory margins
def make_slim_big():
    """Generate closest and biggest victory margins dataframe with election results and metadata.

    Reads seat winner data from seat winner file, filters out By-Elections,
    and selects relevant columns.

    The function:
    1. Reads seat winner data from seat winner file
    2. Filters out By-Elections
    3. Selects relevant columns
    4. Writes final dataframe to parquet format

    Returns:
        None.
    """
    states = get_states()

    df = pd.read_parquet(f'{PATH}/elections_seats_winner.parquet').assign(metric='none')
    df = df[df.election_name != 'By-Election'] # Remove By-Elections, we are not interested in them
    df = df[['metric','type','state','election_name','date','seat','party','name','majority']]
    df = df[df.majority > 0].sort_values(by='majority').reset_index(drop=True)

    def get_top_10(tf,state='Malaysia'):
        if len(tf) == 0: return tf
        tf = pd.concat([tf.head(10).assign(metric='slim'),
                        tf.tail(10).assign(metric='big')],
                        axis=0,ignore_index=True)
        tf['state'] = state
        return tf

    res = pd.concat([get_top_10(df[df.type == 'parlimen']),
                    get_top_10(df[df.type == 'dun'])],
                    axis=0,ignore_index=True)

    for s in states:
        res = pd.concat([res,
                        get_top_10(df[(df.type == 'parlimen') & (df.state == s)],state=s),
                        get_top_10(df[(df.type == 'dun') & (df.state == s)],state=s)],
                        axis=0,ignore_index=True)

    write_parquet(f'{PATH}/elections_slim_big',df=res)


if __name__ == '__main__':
    make_candidates()
    make_seats()
    make_parties()
    make_dates()
    make_veterans()
    make_slim_big()

    TINYBIRD = True

    if TINYBIRD:
        print('\nBeginning Tinybird upload:')

        for FILE in [
            'elections_candidates',
            'elections_seats_winner',
            'elections_parties',
            # 'elections_dates',
            # 'elections_veterans',
            # 'elections_slim_big',
        ]:
            time.sleep(10)
            m = MultipartEncoder(
                fields={
                    'parquet': (
                        f'{FILE}.parquet',
                        open(f'src-data/dashboards/{FILE}.parquet', 'rb'),
                        'text/plain'
                    )
                }
            )

            r = requests.post('https://api.us-east.aws.tinybird.co/v0/datasources',
                headers={
                    'Authorization': f'Bearer {os.getenv("TINYBIRD_API_KEY")}',
                    'Content-Type': m.content_type
                },
                params={
                    'name': FILE,
                    'mode': 'replace',
                    'format': 'parquet',
                },
                data=m
            )

            if r.status_code != 200:
                print(r.status_code)
                print(r.text)
            else:
                print(f'\n200 OK: {FILE}')