# intended as a sample of how fuzzymatching was implemented

import pandas as pd
from rapidfuzz import process, fuzz


def get_closest_match(x, choices, scorer=fuzz.token_set_ratio, cutoff=88):
    match = process.extractOne(x, choices, scorer=scorer, score_cutoff=cutoff)
    return match[0] if match else None


def fuzzy_merge(df1, df2, left_on, right_on, how='inner', scorer=fuzz.token_set_ratio, cutoff=88):
    df_other = df2.copy()
    df_other[left_on] = [get_closest_match(x, df1[left_on].tolist(), scorer, cutoff)
                         for x in df_other[right_on]]
    return df1.merge(df_other, on=left_on, how=how)


def update_candidates(WRITE=False,BALLOTS=None):
    """
    Update the candidate master with new candidates from the selected ballots file.
    Conducts fuzzy matching to avoid generating new UIDs where possible.
    """

    MASTER = 'src-data/candidates_master.csv'

    res = pd.read_csv(MASTER)
    res_u = res.copy()

    df = pd.read_csv(BALLOTS,usecols=['name'])
    df['id'] = df.index + res['id'].iloc[-1] + 1
    print(f'{len(df)} names to match')

    tf = fuzzy_merge(res_u, df, left_on='name', right_on='name', how='left')
    tf['id_y'] = tf['id_y'].fillna(-1).astype(int)
    tf = tf[tf.id_y > -1]
    print(f'{len(tf)} out of {len(df)} ({len(tf)/len(df):.2%}) names matched')
    UPDATE_ID = dict(zip(tf.id_y,tf.id_x))
    df.loc[df.id.isin(tf.id_y.tolist()), 'id'] = df.id.map(UPDATE_ID)
    UPDATE_UID = dict(zip(df['name'],df['id']))

    res = pd.concat([res,df],axis=0,ignore_index=True)
    res['len'] = res.name.str.len()
    res = res.sort_values(by=['id','len']).drop('len',axis=1).drop_duplicates(subset=['name','id']).reset_index(drop=True)
    if WRITE: 
        res.to_csv(MASTER,index=False)

    bf = pd.read_csv(BALLOTS)
    bf['candidate_uid'] = bf['name'].map(UPDATE_UID)
    if WRITE: 
        bf.to_csv(BALLOTS,index=False)


if __name__ == '__main__':
    BALLOTS = f"src-data/prk/prk_ballots.csv"
    update_candidates(WRITE=True,BALLOTS=BALLOTS)