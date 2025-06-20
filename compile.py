"""Module for compiling and validating election data."""

import json as j
import pandas as pd
import numpy as np

from helper import get_states, write_csv_parquet

with open("src-data/lookup_dates.json", encoding="utf-8") as f:
    DATES = j.load(f)
states = get_states(my=1)


def compile_ballots():
    """Compile ballot data from various election sources."""
    df = pd.DataFrame()
    raw = pd.read_csv('src-data/raw_ballots.csv')
    raw.date = pd.to_datetime(raw.date).dt.date.astype(str)

    for s in states + ["PRK"]:
        if "W.P." in s:
            continue

        blim = 0 if s in ["Malaysia", "PRK"] else 1
        ulim = 1 if s == "PRK" else 13 if s == "Sarawak" else 15 if s == "Sabah" else 16
        election_type = "federal" if s == "Malaysia" else "prk" if s == "PRK" else "state"
        name = "GE" if s == "Malaysia" else "BY-ELECTION" if s == "PRK" else "SE"

        for e in range(blim, ulim):
            election_name = f"{name}-{e:02d}" if election_type != "prk" else name
            tf = raw[raw.election == election_name].copy()
            if election_type == 'state':
                tf = tf[tf.state == s].reset_index(drop=True)
            if election_type != "prk":
                tf["date"] = DATES[s][str(e)]

            # Generate total valid votes for each seat
            # Then derive percentage of votes for each candidate
            sf = (
                tf[["date", "seat", "votes"]]
                .groupby(["date", "seat"])
                .sum()
                .reset_index()
                .rename(columns={"votes": "votes_perc"})
            )
            tf = pd.merge(tf, sf, on=["date", "seat"], how="left")
            tf.votes_perc = tf.votes / tf.votes_perc * 100

            # Classify results
            tf["result"] = "lost"
            tf.loc[tf.votes_perc < 12.5, "result"] = "lost_deposit"
            tf["nameseat"] = tf.name + tf.seat
            sf = (
                tf[["date", "nameseat", "seat", "votes"]]
                .sort_values(by=["seat", "votes"])
                .drop_duplicates(subset=["date", "seat"], keep="last")
            )
            tf.loc[tf.nameseat.isin(sf.nameseat.tolist()), "result"] = "won"
            tf = tf.drop("nameseat", axis=1)
            tf.loc[(tf.votes == 0) & (tf.votes_perc != 0), "result"] = "won_uncontested"

            df = (
                tf.copy()
                if len(df) == 0
                else pd.concat([df, tf], axis=0, ignore_index=True)
            )

    assert len(df.drop_duplicates(subset=["date", "election", "state", "seat"])) == len(
        df[df.result.str.contains("won")]
    ), "Number of winners and contests does not match!"
    write_csv_parquet("src-data/consol_ballots", df=df)
    types = {"GE": "federal", "SE": "state", "BY-ELECTION": "byelection"}
    for k, v in types.items():
        write_csv_parquet(f"src-data/{v}_ballots", df=df[df.election.str.startswith(k)])


def compile_summary():
    """Compile summary data from various election sources."""
    df = pd.DataFrame()
    raw = pd.read_csv('src-data/raw_stats.csv')
    raw.date = pd.to_datetime(raw.date).dt.date.astype(str)

    for s in states + ["PRK"]:
        if "W.P." in s:
            continue

        blim = 0 if s in ["Malaysia", "PRK"] else 1
        ulim = 1 if s == "PRK" else 13 if s == "Sarawak" else 15 if s == "Sabah" else 16
        election_type = "federal" if s == "Malaysia" else "prk" if s == "PRK" else "state"
        name = "GE" if s == "Malaysia" else "BY-ELECTION" if s == "PRK" else "SE"

        for e in range(blim, ulim):
            election_name = f"{name}-{e:02d}" if election_type != "prk" else name
            tf = raw[raw.election == election_name].copy()
            if election_type == 'state':
                tf = tf[tf.state == s].reset_index(drop=True)
            if election_type != "prk":
                tf["date"] = DATES[s][str(e)]

            df = (
                tf.copy()
                if len(df) == 0
                else pd.concat([df, tf], axis=0, ignore_index=True)
            )

    df["votes_valid"] = df.ballots_issued - df.votes_rejected - df.ballots_not_returned

    # Generate majority
    wf = pd.read_csv("src-data/consol_ballots.csv")
    w1 = wf[wf.result.str.contains("won")]
    w2 = (
        wf[~wf.result.str.contains("won")]
        .sort_values(by=["votes"], ascending=False)
        .drop_duplicates(subset=["date", "election", "state", "seat"], keep="first")
    )
    assert len(w1) == len(w2) + len(
        w1[w1.result.str.contains("uncontested")]
    ), "Number of winners and losers does not match!"
    col_keep = ["date", "election", "state", "seat", "votes"]
    mf = pd.merge(w1[col_keep + ["result"]], w2[col_keep], on=col_keep[:-1], how="left")
    assert (
        len(mf[(mf.votes_y.isnull()) & (~mf.result.str.contains("uncontested"))]) == 0
    ), "Missing runner-up outside uncontested seats!"
    mf.votes_y = mf.votes_y.fillna(0).astype(int)
    mf["majority"] = mf.votes_x - mf.votes_y
    mf.loc[mf.votes_y == 0, "majority"] = 0
    mf = mf.drop(["votes_x", "votes_y", "result"], axis=1)

    df = pd.merge(df, mf, on=col_keep[:-1], how="left")
    assert (
        len(df[df.majority.isnull()]) == 0
    ), f"Imperfect match between ballots and summary!\n{df[df.majority.isnull()]}"
    df["voter_turnout"] = df.ballots_issued / df.voters_total * 100
    df["majority_perc"] = df.majority / df.votes_valid * 100
    for col in ["voter_turnout", "majority_perc"]:
        df.loc[df.ballots_issued == 0, col] = np.nan
    df["votes_rejected_perc"] = (
        df.votes_rejected / (df.ballots_issued - df.ballots_not_returned) * 100
    )
    df["ballots_not_returned_perc"] = df.ballots_not_returned / df.ballots_issued * 100

    write_csv_parquet("src-data/consol_summary", df=df)
    types = {"GE": "federal", "SE": "state", "BY-ELECTION": "byelection"}
    for k, v in types.items():
        write_csv_parquet(f"src-data/{v}_summary", df=df[df.election.str.startswith(k)])


def validate():
    """Validate the compiled data for consistency."""
    bf = pd.read_parquet("src-data/consol_ballots.parquet")
    col_join = ["date", "election", "state", "seat"]
    bf = (
        bf[col_join + ["votes"]]
        .groupby(col_join)
        .sum()
        .reset_index()
        .rename(columns={"votes": "votes_valid"})
    )

    df = pd.read_parquet("src-data/consol_summary.parquet").rename(
        columns={"votes_valid": "votes_valid_derived"}
    )
    df = pd.merge(df, bf, on=col_join, how="left")
    df["check"] = df.votes_valid - df.votes_valid_derived
    df["check_perc"] = df.check.abs() / df.votes_valid * 100

    if len(df[df.check != 0]) > 0:
        df = df.sort_values(by=["date", "state", "seat"]).drop("check_perc", axis=1)
        df = df[["check"] + list(df.columns[:-1])]
        df[df.check != 0].to_csv("logs/check.csv", index=False)
        raise ValueError(f"Validation failed for {len(df[df.check != 0])} seats!")


if __name__ == "__main__":
    print("\nCompiling ballots:")
    compile_ballots()
    print("\nCompiling summaries:")
    compile_summary()
    print("\nValidating:")
    validate()
    print('')
