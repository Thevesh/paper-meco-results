import os
import pandas as pd
import json as j
from glob import glob as g
from datetime import datetime

from helper import get_states, upload_s3, upload_s3_bulk


def make_candidates():
    DATA = {"data": []}

    COL_API_CANDIDATE = ['name','election_name','type','date','seat','party','votes','votes_perc','result']

    df = pd.read_parquet('src-data/dashboards/elections_candidates.parquet')
    for SLUG in sorted(list(df.slug.unique())):
        tf = df[df.slug == SLUG].copy()[COL_API_CANDIDATE].sort_values(by='date',ascending=False)
        tf = tf.to_dict(orient='records')
        tf = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in tf] # proper JSON null
        DATA['data'] = tf
        j.dump(DATA, open(f'api/candidates/{SLUG}.json','w'))


def make_seats():
    DATA = {"data": []}

    COL_API_SEAT = ['election_name','seat','date','party','name','majority','majority_perc']

    df = pd.read_parquet('src-data/dashboards/elections_seats_winner.parquet')
    df.slug = df.type + '-' + df.slug
    for SLUG in df.slug.unique():
        tf = df[df.slug == SLUG].copy()[COL_API_SEAT].sort_values(by='date',ascending=False)
        tf = tf.to_dict(orient='records')
        tf = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in tf] # proper JSON null
        DATA['data'] = tf
        j.dump(DATA, open(f'api/seats/{SLUG}.json','w'))


def make_parties():

    DATA = { "data": []}

    COL_PARTY = ['state', 'type', 'election_name', 'date', 'seats', 'seats_total', 'seats_perc', 'votes', 'votes_perc']

    df = pd.read_parquet('src-data/dashboards/elections_parties.parquet')
    for PARTY in df.party.unique():
        tf = df[df.party == PARTY].copy()

        # loop over parlimen and dun
        for TYPE in tf['type'].unique():
            if not os.path.exists(f'api/parties/{PARTY}/{TYPE}'):
                    os.makedirs(f'api/parties/{PARTY}/{TYPE}')
            tft = tf[tf.type == TYPE].copy()

            # loop over states
            for STATE in tft.state.unique():
                if TYPE == 'dun' and STATE == 'Malaysia':
                    continue
                
                tfts = tft[tft.state == STATE].copy(COL_PARTY).sort_values(by='date',ascending=False)
                res = tfts.to_dict(orient='records')
                res = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in res] # proper JSON null

                DATA['data'] = res
                j.dump(DATA,open(f'api/parties/{PARTY}/{TYPE}/{STATE}.json','w'))


def make_results():
    print('')
    DATA = { "ballot": [], "summary": []}

    COL_API_BALLOT = ['name','party','votes','votes_perc','result']
    COL_API_BALLOT_SUMMARY = ['date','voter_turnout','voter_turnout_perc','votes_rejected','votes_rejected_perc','majority','majority_perc']

    df = pd.read_parquet('src-data/dashboards/elections_candidates.parquet')
    print(f"{df.drop_duplicates(subset=['seat','date']).shape[0]:,.0f} results to create")
    for SEAT in df.seat.unique():
        if not os.path.exists(f'api/results/{SEAT}'):
            os.makedirs(f'api/results/{SEAT}')

        dfs = df[df.seat == SEAT].copy()
        for DATE in dfs.date.unique():
            dfse = dfs[dfs.date == DATE].copy()[COL_API_BALLOT].sort_values(by='votes',ascending=False)
            dfse_b = dfs[dfs.date == DATE].copy()[COL_API_BALLOT_SUMMARY].drop_duplicates()

            res = dfse.to_dict(orient='records')
            res = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in res] # proper JSON null

            res_b = dfse_b.to_dict(orient='records')
            res_b = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in res_b] # proper JSON null
            
            DATA['ballot'] = res
            DATA['summary'] = res_b
            j.dump(DATA,open(f'api/results/{SEAT}/{DATE}.json','w'))


def make_elections():
    COL_COMBO = ['state','type','election_name']
    COL_FINAL = {
        "ballot": ['party','seats','seats_total','seats_perc','votes','votes_perc'],
        "summary": ['voter_turnout','voter_turnout_perc','votes_rejected','votes_rejected_perc'],
        "stats": ['seat','date','party','name','state','majority','majority_perc','voter_turnout','voter_turnout_perc','votes_rejected','votes_rejected_perc']
    }

    dfm = pd.read_parquet('src-data/dashboards/elections_parties.parquet').sort_values(by=['seats_perc','votes_perc'],ascending=False)
    dfs = pd.read_parquet('src-data/dashboards/elections_summary.parquet').fillna(0)
    dft = pd.read_parquet('src-data/dashboards/elections_seats_winner.parquet')
    dft = dft[dft.election_name != 'By-Election']
    dft = pd.concat([dft[dft.type == 'parlimen'].assign(state='Malaysia'),dft],axis=0,ignore_index=True)

    assert len(dfm.drop_duplicates(subset=COL_COMBO)) \
        == len(dfs.drop_duplicates(subset=COL_COMBO)) \
        == len(dft.drop_duplicates(subset=COL_COMBO)), \
        f'Mismatch between 3 components!\
            ballots: {len(dfm.drop_duplicates(subset=COL_COMBO))} \
            summaries: {len(dfs.drop_duplicates(subset=COL_COMBO))} \
            stats: {len(dft.drop_duplicates(subset=COL_COMBO))}'

    df = {
        "ballot": dfm,
        "summary": dfs,
        "stats": dft
    }

    for TYPE in dfm.type.unique():
        tf = dfm[dfm.type == TYPE].copy()
        for STATE in tf.state.unique():
            tf = dfm[(dfm.type == TYPE) & (dfm.state == STATE)].copy().copy()
            for ELECTION in tf.election_name.unique():

                # ensure state folder exists
                if not os.path.exists(f'api/elections/{STATE}'): 
                    os.makedirs(f'api/elections/{STATE}')

                # now loop over the keys
                DATA = { "ballot": [], "summary": [], "stats":[]}
                for KEY in df.keys():
                    tf = df[KEY].copy()
                    tf = tf[(tf.type == TYPE) & (tf.state == STATE) & (tf.election_name == ELECTION)]
                    res = tf[COL_FINAL[KEY]].to_dict(orient='records')
                    res = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in res] # proper JSON null
                    DATA[KEY] = res
                j.dump(DATA,open(f'api/elections/{STATE}/{TYPE}-{ELECTION}.json','w'))


def make_trivia():
    STATES = get_states(my=1)

    sb = pd.read_parquet('src-data/dashboards/elections_slim_big.parquet').sort_values(by='majority')
    vt = pd.read_parquet('src-data/dashboards/elections_veterans.parquet')

    for STATE in STATES:
        df = {
            "slim_big": sb[sb.state == STATE].copy().drop('state',axis=1),
            "veterans_parlimen": vt[(vt.type == 'parlimen') & (vt.state == STATE)].copy().drop(['type','state'],axis=1),
            "veterans_dun": vt[(vt.type == 'dun') & (vt.state == STATE)].copy().drop(['type','state'],axis=1)
        }

        DATA = { "slim_big": [], "veterans_parlimen": [], "veterans_dun":[]}
        for KEY in DATA.keys():
            tf = df[KEY].copy()
            res = tf.to_dict(orient='records')
            res = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in res] # proper JSON null
            DATA[KEY] = res
        j.dump(DATA,open(f'api/trivia/{STATE}.json','w'))


def upload_data(PATH='candidates/*'):
    FILES = g(f'api/{PATH}.json')
    FILES_TO_UPLOAD = sorted([(f,f.replace('api/','')) for f in FILES])

    res = upload_s3_bulk(
        bucket_name='static.electiondata.my',
        files_to_upload=FILES_TO_UPLOAD,
        max_workers=120
    )


def make_upload_dates():
    DATA = {"data": []}
    df = pd.read_csv('src-data/lookup_seats.csv')\
        [['state','election','date']]\
        .drop_duplicates()\
        .sort_values(by=['state','election'])
    df = df[~df.election.str.contains('BY-ELECTION')]
    df = pd.concat([
        df[df.election.str.startswith('GE')].assign(state='Malaysia').drop_duplicates(),
        df
    ],axis=0,ignore_index=True)
    res = df.to_dict(orient='records')
    DATA['data'] = res
    j.dump(DATA,open('api/dates.json','w'))

    print(upload_s3(
        bucket_name='static.electiondata.my',
        source_file_name='api/dates.json',
        cloud_file_name='dates.json'
    ))


if __name__ == '__main__':
    START = datetime.now()
    print(f'\nStart: {START.strftime("%Y-%m-%d %H:%M:%S")}')
    make_candidates()
    make_seats()
    make_parties()
    make_results()
    make_elections()
    make_trivia()
    for PATH in [
        'candidates/*',
        'seats/*',
        'parties/*/*/*',
        'results/*/*',
        'elections/*/*',
        'trivia/*'
    ]:
        upload_data(PATH=PATH)
    make_upload_dates()
    print(f'\nEnd: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'\nDuration: {datetime.now() - START}\n')