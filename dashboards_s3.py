import os
import pandas as pd
import json as j
from glob import glob as g
from datetime import datetime

from helper import upload_s3_bulk


def make_candidates():
    OUTPUT = {"data": []}

    COL_API_CANDIDATE = ['name','election_name','type','date','seat','party','votes','votes_perc','result']

    df = pd.read_parquet('src-data/dashboards/elections_candidates.parquet')
    for SLUG in sorted(list(df.slug.unique())):
        tf = df[df.slug == SLUG].copy()[COL_API_CANDIDATE].sort_values(by='date',ascending=False)
        tf = tf.to_dict(orient='records')
        tf = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in tf] # proper JSON null
        OUTPUT['data'] = tf
        j.dump(OUTPUT, open(f'api/candidates/{SLUG}.json','w'))


def make_seats():
    OUTPUT = {"data": []}

    COL_API_SEAT = ['election_name','seat','date','party','name','majority','majority_perc']

    df = pd.read_parquet('src-data/dashboards/elections_seats_winner.parquet')
    df.slug = df.type + '-' + df.slug
    for SLUG in df.slug.unique():
        tf = df[df.slug == SLUG].copy()[COL_API_SEAT].sort_values(by='date',ascending=False)
        tf = tf.to_dict(orient='records')
        tf = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in tf] # proper JSON null
        OUTPUT['data'] = tf
        j.dump(OUTPUT, open(f'api/seats/{SLUG}.json','w'))


def make_parties():

    OUTPUT = { "data": []}

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

                OUTPUT['data'] = res
                j.dump(OUTPUT,open(f'api/parties/{PARTY}/{TYPE}/{STATE}.json','w'))


def make_results():
    print('')
    DATA = { "ballot": [], "summary": []}

    COL_API_BALLOT = ['name','party','votes','votes_perc','result']
    COL_API_BALLOT_SUMMARY = ['date','voter_turnout','voter_turnout_perc','votes_rejected','votes_rejected_perc','majority','majority_perc']

    df = pd.read_parquet('src-data/dashboards/elections_candidates.parquet')
    print(f"{df.drop_duplicates(subset=['seat','election_name']).shape[0]:,.0f} seats to create")
    for s in df.seat.unique():
        if not os.path.exists(f'api/results/{s}'):
            os.makedirs(f'api/results/{s}')

        dfs = df[df.seat == s].copy()
        for g in dfs.election_name.unique():
            dfse = dfs[dfs.election_name == g].copy()[COL_API_BALLOT].sort_values(by='votes',ascending=False)
            dfse_b = dfs[dfs.election_name == g].copy()[COL_API_BALLOT_SUMMARY].drop_duplicates()

            res = dfse.to_dict(orient='records')
            res = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in res] # proper JSON null

            res_b = dfse_b.to_dict(orient='records')
            res_b = [{k: (None if pd.isna(v) else v) for k,v in record.items()} for record in res_b] # proper JSON null
            
            DATA['ballot'] = res
            DATA['summary'] = res_b
            j.dump(DATA,open(f'api/results/{s}/{g}.json','w'))


def upload_candidates():
    FILES = g('api/candidates/*.json')
    FILES_TO_UPLOAD = sorted([(f,f.replace('api/','')) for f in FILES])

    res = upload_s3_bulk(
        bucket_name='static.electiondata.my',
        files_to_upload=FILES_TO_UPLOAD,
        max_workers=120
    )


def upload_seats():
    FILES = g('api/seats/*.json')
    FILES_TO_UPLOAD = sorted([(f,f.replace('api/','')) for f in FILES])

    res = upload_s3_bulk(
        bucket_name='static.electiondata.my',
        files_to_upload=FILES_TO_UPLOAD,
        max_workers=120
    )


def upload_parties():
    FILES = g('api/parties/*/*/*.json')
    FILES_TO_UPLOAD = sorted([(f,f.replace('api/','')) for f in FILES])

    res = upload_s3_bulk(
        bucket_name='static.electiondata.my',
        files_to_upload=FILES_TO_UPLOAD,
        max_workers=120
    )


def upload_results():
    FILES = g('api/results/*/*.json')
    FILES_TO_UPLOAD = sorted([(f,f.replace('api/','')) for f in FILES])

    res = upload_s3_bulk(
        bucket_name='static.electiondata.my',
        files_to_upload=FILES_TO_UPLOAD,
        max_workers=120
    )


if __name__ == '__main__':
    START = datetime.now()
    print(f'\nStart: {START.strftime("%Y-%m-%d %H:%M:%S")}')
    # make_candidates()
    # make_seats()
    # make_parties()
    make_results()
    # upload_candidates()
    # upload_seats()
    # upload_parties()
    upload_results()
    print(f'\nEnd: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'\nDuration: {datetime.now() - START}\n')