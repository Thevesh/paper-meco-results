"""Module for fuzzy matching and merging election candidate data."""

import pandas as pd
from rapidfuzz import process, fuzz


def get_closest_match(x, choices, scorer=fuzz.token_set_ratio, cutoff=88):
    """Find the closest match for a string in a list of choices using fuzzy matching.
    
    Args:
        x (str): String to match
        choices (list): List of strings to match against
        scorer: Fuzzy matching scorer function
        cutoff (int): Minimum score threshold for a match
        
    Returns:
        str or None: Best matching string if score exceeds cutoff, None otherwise
    """
    match = process.extractOne(x, choices, scorer=scorer, score_cutoff=cutoff)
    return match[0] if match else None


def fuzzy_merge(
    df1,
    df2,
    left_on,
    right_on,
    how="inner",
    scorer=fuzz.token_set_ratio,
    cutoff=88,
):
    """Merge two dataframes using fuzzy matching on specified columns.
    
    Args:
        df1 (pd.DataFrame): Left dataframe
        df2 (pd.DataFrame): Right dataframe
        left_on (str): Column name in left dataframe to match on
        right_on (str): Column name in right dataframe to match on
        how (str): Type of merge to perform
        scorer: Fuzzy matching scorer function
        cutoff (int): Minimum score threshold for a match
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    df_other = df2.copy()
    df_other[left_on] = [
        get_closest_match(x, df1[left_on].tolist(), scorer, cutoff)
        for x in df_other[right_on]
    ]
    return df1.merge(df_other, on=left_on, how=how)


def update_candidates(write=False, ballots_file=None):
    """Update the candidate master with new candidates from the selected ballots file.
    
    Conducts fuzzy matching to avoid generating new UIDs where possible.
    
    Args:
        write (bool): Whether to write changes to files
        ballots_file (str): Path to ballots file to process
    """
    master_file = "src-data/candidates_master.csv"

    res = pd.read_csv(master_file)
    res_u = res.copy()

    df = pd.read_csv(ballots_file, usecols=["name"])
    df["id"] = df.index + res["id"].iloc[-1] + 1
    print(f"{len(df)} names to match")

    tf = fuzzy_merge(res_u, df, left_on="name", right_on="name", how="left")
    tf["id_y"] = tf["id_y"].fillna(-1).astype(int)
    tf = tf[tf.id_y > -1]
    print(f"{len(tf)} out of {len(df)} ({len(tf)/len(df):.2%}) names matched")
    update_id = dict(zip(tf.id_y, tf.id_x))
    df.loc[df.id.isin(tf.id_y.tolist()), "id"] = df.id.map(update_id)
    update_uid = dict(zip(df["name"], df["id"]))

    res = pd.concat([res, df], axis=0, ignore_index=True)
    res["len"] = res.name.str.len()
    res = (
        res.sort_values(by=["id", "len"])
        .drop("len", axis=1)
        .drop_duplicates(subset=["name", "id"])
        .reset_index(drop=True)
    )
    if write:
        res.to_csv(master_file, index=False)

    bf = pd.read_csv(ballots_file)
    bf["candidate_uid"] = bf["name"].map(update_uid)
    if write:
        bf.to_csv(ballots_file, index=False)


if __name__ == "__main__":
    BALLOTS_FILE = "src-data/prk/prk_ballots.csv"
    update_candidates(write=True, ballots_file=BALLOTS_FILE)
