import pandas as pd
import numpy as np
import json as j

from helper import get_states, write_csv_parquet

dates = j.load(open('src-data/dates_master.json'))
STATES = get_states(my=1)


def compile_ballots():
    df = pd.DataFrame()
    for s in STATES + ['PRK']:
        if 'W.P.' in s: continue

        BLIM = 0 if s in ['Malaysia','PRK'] else 1
        ULIM = 1 if s == 'PRK' else 13 if s == 'Sarawak' else 15 if s == 'Sabah' else 16
        TYPE = 'federal' if s == 'Malaysia' else 'prk' if s == 'PRK' else 'state'
        NAME = 'GE' if s == 'Malaysia' else 'BY-ELECTION' if s == 'PRK' else 'SE'

        for e in range(BLIM,ULIM):

            FILENAME = \
                f'ge{e:02d}_ballots.csv' if TYPE == 'federal' else \
                'prk_ballots.csv' if TYPE == 'prk' else \
                f"{s.lower().replace(' ','')}_se{e:02d}_ballots.csv"
            tf = pd.read_csv(f'src-data/{TYPE}/{FILENAME}').rename(columns={'parlimen':'seat','dun':'seat'})

            COL_ORI = [x for x in tf.columns if x not in ['date','election']]
            tf['election'] = f'{NAME}-{e:02d}' if TYPE != 'prk' else NAME
            if TYPE != 'prk': tf['date'] = dates[s][str(e)] 
            tf = tf[['date','election'] + COL_ORI]
            
            # Generate total valid votes for each seat, and then derive percentage of votes for each candidate
            sf = tf[['date','seat','votes']].groupby(['date','seat']).sum().reset_index().rename(columns={'votes':'votes_perc'})
            tf = pd.merge(tf,sf,on=['date','seat'],how='left')
            tf.votes_perc = tf.votes/tf.votes_perc * 100

            # Classify results
            tf['result'] = 'lost'
            tf.loc[tf.votes_perc < 12.5, 'result'] = 'lost_deposit'
            tf['nameseat'] = tf.name + tf.seat
            sf = tf[['date','nameseat','seat','votes']].sort_values(by=['seat','votes']).drop_duplicates(subset=['date','seat'],keep='last')
            tf.loc[tf.nameseat.isin(sf.nameseat.tolist()), 'result'] = 'won'
            tf = tf.drop('nameseat',axis=1)
            tf.loc[(tf.votes == 0) & (tf.votes_perc != 0), 'result'] = 'won_uncontested'

            df = tf.copy() if len(df) == 0 else pd.concat([df,tf],axis=0,ignore_index=True)

    assert len(df.drop_duplicates(subset=['date','election','state','seat'])) == len(df[df.result.str.contains('won')]), 'Number of winners and contests does not match!'
    write_csv_parquet('src-data/consol_ballots',df=df)
    TPYES = {'GE':'federal','SE':'state','BY-ELECTION':'byelection'}
    for k,v in TPYES.items():
        write_csv_parquet(f'src-data/{v}_ballots',df=df[df.election.str.startswith(k)])


def compile_summary():
    df = pd.DataFrame()
    for s in STATES + ['PRK']:
        if 'W.P.' in s: continue

        BLIM = 0 if s in ['Malaysia','PRK'] else 1
        ULIM = 1 if s == 'PRK' else 13 if s == 'Sarawak' else 15 if s == 'Sabah' else 16
        TYPE = 'federal' if s == 'Malaysia' else 'prk' if s == 'PRK' else 'state'
        NAME = 'GE' if s == 'Malaysia' else 'BY-ELECTION' if s == 'PRK' else 'SE'

        for e in range(BLIM,ULIM):

            FILENAME = \
                f'ge{e:02d}_summary.csv' if TYPE == 'federal' else \
                'prk_summary.csv' if TYPE == 'prk' else \
                f"{s.lower().replace(' ','')}_se{e:02d}_summary.csv"
            tf = pd.read_csv(f'src-data/{TYPE}/{FILENAME}').rename(columns={'parlimen':'seat','dun':'seat'})

            COL_ORI = [x for x in tf.columns if x not in ['date','election']]
            tf['election'] = f'{NAME}-{e:02d}' if TYPE != 'prk' else NAME
            if TYPE != 'prk': tf['date'] = dates[s][str(e)] 
            tf = tf[['date','election'] + COL_ORI]

            df = tf.copy() if len(df) == 0 else pd.concat([df,tf],axis=0,ignore_index=True)

    df['votes_valid'] = df.ballots_issued - df.votes_rejected - df.ballots_not_returned

    # Generate majority
    wf = pd.read_csv('src-data/consol_ballots.csv')
    w1 = wf[wf.result.str.contains('won')]
    w2 = wf[~wf.result.str.contains('won')]\
        .sort_values(by=['votes'],ascending=False)\
        .drop_duplicates(subset=['date','election','state','seat'],keep='first')
    assert len(w1) == len(w2) + len(w1[w1.result.str.contains('uncontested')]), 'Number of winners and losers does not match!'
    COL_KEEP = ['date','election','state','seat','votes']
    mf = pd.merge(w1[COL_KEEP + ['result']],w2[COL_KEEP],on=COL_KEEP[:-1],how='left')
    assert len(mf[(mf.votes_y.isnull()) & (~mf.result.str.contains('uncontested'))]) == 0, 'Missing runner-up outside uncontested seats!'
    mf.votes_y = mf.votes_y.fillna(0).astype(int)
    mf['majority'] = mf.votes_x - mf.votes_y
    mf.loc[mf.votes_y == 0, 'majority'] = 0
    mf = mf.drop(['votes_x','votes_y','result'],axis=1)

    df = pd.merge(df,mf,on=COL_KEEP[:-1],how='left')
    assert len(df[df.majority.isnull()]) == 0, f'Imperfect match between ballots and summary!\n{df[df.majority.isnull()]}'
    df['voter_turnout'] = df.ballots_issued / df.voters_total * 100
    df['majority_perc'] = df.majority / df.votes_valid * 100
    for col in ['voter_turnout','majority_perc']:
        df.loc[df.ballots_issued == 0,col] = np.nan
    df['votes_rejected_perc'] = df.votes_rejected / (df.ballots_issued - df.ballots_not_returned) * 100
    df['ballots_not_returned_perc'] = df.ballots_not_returned / df.ballots_issued * 100

    write_csv_parquet('src-data/consol_summary',df=df)
    TPYES = {'GE':'federal','SE':'state','BY-ELECTION':'byelection'}
    for k,v in TPYES.items():
        write_csv_parquet(f'src-data/{v}_summary',df=df[df.election.str.startswith(k)])


def validate():
    bf = pd.read_parquet('src-data/consol_ballots.parquet')
    COL_JOIN = ['date','election','state','seat']
    bf = bf[COL_JOIN + ['votes']]\
        .groupby(COL_JOIN)\
        .sum().reset_index()\
        .rename(columns={'votes':'votes_valid'})

    df = pd.read_parquet('src-data/consol_summary.parquet').rename(columns={'votes_valid':'votes_valid_derived'})
    df = pd.merge(df,bf,on=COL_JOIN,how='left')
    df['check'] = df.votes_valid - df.votes_valid_derived
    df['check_perc'] = df.check.abs() / df.votes_valid * 100
    if len(df[df.check != 0]) > 0:
        df = df.sort_values(by=['date','state','seat']).drop('check_perc',axis=1)
        df = df[~((df.state == 'Sarawak') & (df.election == 'SE-09'))]
        df = df[['check'] + list(df.columns[:-1])]
        df[df.check != 0].to_csv('logs/check.csv',index=False)
        raise Exception(f'Validation failed for {len(df[df.check != 0])} seats!')


if __name__ == '__main__':
    compile_ballots()
    compile_summary()
    validate()