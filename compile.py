"""
Script to produce consol_ballots and consol_stats, the two key source datasets.

Steps:
    1. Read raw ballots files.
    2a. Generate derived data for each candidate
        - Vote percentage
        - Rank
        - Result
    2b. Generate derived data for each seat:
        - Valid votes
        - N candidates
        - Majority
    3. Merge derived data for each seat with raw stats file.
    4. Compute derived statistics:
        - Voter turnout
        - Majority percentage
        - Rejected votes percentage
        - Ballots-not-returned percentage
    5. Validate the data against expected states.
    6. Save consolidated outputs.
    7. Generate subsets for federal, each state, and by-elections

Checks:
    - No invalid states in raw ballots file
    - No invalid states in raw stats file
    - N unique contests (date-election-state-seat combo) in same in ballots and stats
    - No impossible values > 100% in percentages
    - Compliance with V = I - U - R
    - All UIDs for candidates, parties, and coalitions are present in respective lookup files
    - All unique contests are present in seat lookup file
    - Where a state election was held with a general election, the DUN electorates in each
      parliamentary seat sum to its electorate (hard failure from 2008 onward)
"""

import pandas as pd
import numpy as np

from helper import get_states, get_final_cols, write_csv_parquet

# Electorate coherence is enforced from GE-12 (2008) onward; earlier discrepancies are logged
ELECTORATE_HARD_FROM = pd.Timestamp("2008-01-01").date()


def check_electorates(sf: pd.DataFrame) -> pd.DataFrame:
    """
    For every state election held with a general election (same polling date in the state),
    sum the DUN electorates within each parliamentary seat, per lookup_dun_parlimen.csv, and
    compare with that seat's general-election electorate. A seat is only checked when it and
    all its DUNs carry an electorate; otherwise its status says why it could not be.
    Returns one row per parliamentary seat checked or skipped.
    """
    lk = pd.read_csv("data/lookup_dun_parlimen.csv")
    ge = sf[sf.election.str.startswith("GE")]
    se = sf[sf.election.str.startswith("SE")]
    pairs = se.merge(ge[["date", "state", "election"]].drop_duplicates(), on=["date", "state"],
                     suffixes=("", "_ge"))[["state", "election", "election_ge"]].drop_duplicates()

    paired = se.merge(pairs, on=["state", "election"]).assign(code_dun=lambda x: x.seat.str[:4])
    unmapped = paired.merge(lk, on=["state", "election", "code_dun"], how="left").code_parlimen
    assert unmapped.notna().all(), "DUN missing from lookup_dun_parlimen.csv!"

    duns = lk.merge(pairs, on=["state", "election"]).merge(
        se.assign(code_dun=se.seat.str[:4])[["state", "election", "code_dun", "voters_total"]],
        on=["state", "election", "code_dun"], how="left",
    )
    duns["missing"] = duns.voters_total.isna() | (duns.voters_total == 0)
    parl = duns.groupby(["state", "election", "election_ge", "code_parlimen"], as_index=False).agg(
        n_dun=("code_dun", "size"), dun_sum=("voters_total", "sum"), dun_missing=("missing", "sum")
    )
    parl = parl.merge(
        ge.assign(code_parlimen=ge.seat.str[:5])[
            ["date", "state", "election", "code_parlimen", "voters_total"]
        ].rename(columns={"election": "election_ge", "voters_total": "ge_voters"}),
        on=["state", "election_ge", "code_parlimen"], how="left",
    )
    parl["status"] = np.select(
        [parl.ge_voters.isna(), parl.ge_voters == 0, parl.dun_missing > 0],
        ["no GE seat", "GE electorate missing", "DUN electorate missing"], "checked",
    )
    parl["diff"] = np.where(parl.status == "checked", parl.dun_sum - parl.ge_voters, np.nan)
    return parl


def main():
    """
    Compile and validate election data from raw source files.
    Created as a function to prevent unsafe imports.
    """
    states = get_states()
    col_group = ["date", "election", "state", "seat"]

    print("\n\n--------- Generating lookup parquets ----------\n")

    for v in [
        "candidate",
        "party",
        "coalition",
        "coalition_succession",
        "party_succession",
        "dates",
        "prk",
    ]:
        cf = pd.read_csv(f"data/lookup_{v}.csv")
        for c in cf.columns:
            if c in ["date"]:
                cf[c] = pd.to_datetime(cf[c]).dt.date
        write_csv_parquet(f"data/lookup_{v}", cf)

    cf = pd.read_parquet("data/lookup_candidate.parquet")
    assert cf["candidate_rn"].is_unique, "Duplicate values found in candidate_rn column!"
    assert cf["candidate_uid"].is_unique, "Duplicate values found in candidate_uid column!"
    map_rn_uid = dict(zip(cf["candidate_rn"], cf["candidate_uid"]))

    print("\n--------- Compiling ballots ----------\n")

    df = pd.read_csv("data/raw_ballots.csv").rename(columns={"candidate_rn": "candidate_uid"})
    df["candidate_uid"] = df["candidate_uid"].map(map_rn_uid)
    assert len(df[df.candidate_uid.isna()]) == 0, "Missing candidate_uid values!"
    df.date = pd.to_datetime(df.date).dt.date
    assert len(df[~df.state.isin(states)]) == 0, "Invalid state in raw ballots file!"

    grp = df.groupby(col_group)["votes"]

    df = df.assign(
        votes_valid=grp.transform("sum"),
        votes_perc=df["votes"] / grp.transform("sum") * 100,
        rank=grp.rank(method="min", ascending=False).astype(int),
        n_candidates=grp.transform("count"),
    )
    df["majority"] = grp.transform("max") - grp.transform(
        lambda g: g.nlargest(2).iloc[-1] if len(g) >= 2 else 0
    )

    df["result"] = "lost"
    df.loc[df["votes_perc"] < 12.5, "result"] = "lost_deposit"
    df.loc[df["rank"] == 1, "result"] = "won"
    df.loc[df["n_candidates"] == 1, "result"] = "won_uncontested"
    df.loc[(df["votes_valid"] == 0) & (df["n_candidates"] > 1), "result"] = "pending"

    cand = pd.read_parquet("data/lookup_candidate.parquet").drop("candidate_rn", axis=1)
    df = pd.merge(df, cand, on="candidate_uid", how="left").rename(columns={"dob": "age"})
    df.age = pd.to_numeric(df.age.str[:4], errors="coerce").fillna(1786).astype(int)
    df.loc[df.age != 1786, "age"] = pd.to_datetime(df.date).dt.year - df.age
    df.loc[df.age == 1786, "age"] = -1

    write_csv_parquet("data/consol_ballots", df[get_final_cols("ballots")])

    print("\n\n--------- Compiling stats ----------\n")

    sf = pd.read_csv("data/raw_stats.csv")
    sf.date = pd.to_datetime(sf.date).dt.date
    assert len(sf[~sf.state.isin(states)]) == 0, "Invalid state in raw stats file!"

    df = df[
        ["date", "election", "state", "seat", "votes_valid", "majority", "n_candidates"]
    ].drop_duplicates()
    assert len(df) == len(sf), "N ballots not equal to N stats!"

    df = pd.merge(sf, df, on=["date", "election", "state", "seat"], how="left")
    df["voter_turnout"] = df["ballots_issued"] / df["voters_total"] * 100
    df["majority_perc"] = df["majority"] / df["votes_valid"] * 100
    df["votes_rejected_perc"] = (
        df["votes_rejected"] / (df["ballots_issued"] - df["ballots_not_returned"]) * 100
    )
    df["ballots_not_returned_perc"] = df["ballots_not_returned"] / df["ballots_issued"] * 100
    for col in ["voter_turnout", "majority_perc"]:
        df.loc[df.ballots_issued == 0, col] = np.nan
    for c in ["voter_turnout", "majority_perc", "votes_rejected_perc", "ballots_not_returned_perc"]:
        assert len(df[df[c] > 100]) == 0, f"{c} has impossible value > 100%"

    write_csv_parquet("data/consol_stats", df[get_final_cols("stats")])

    print("\n\n--------- Validating files ----------\n")

    df["check"] = df.ballots_issued - df.ballots_not_returned - df.votes_rejected - df.votes_valid
    if len(df[df.check != 0]) > 0:
        df = df.sort_values(by=["date", "state", "seat"])
        df = df[["check"] + list(df.columns[:-1])]
        df[df.check != 0].to_csv("logs/check.csv", index=False)
        raise ValueError(f"Validation failed for {len(df[df.check != 0])} seats!")

    ef = check_electorates(sf)
    flagged = ef[(ef.status != "checked") | (ef["diff"] != 0)]
    flagged.to_csv("logs/check_electorate.csv", index=False)
    hard = flagged[flagged.status == "checked"]
    hard = hard[hard.date >= ELECTORATE_HARD_FROM]
    print(
        f"Electorate coherence: {(ef.status == 'checked').sum():,} parliamentary seats checked; "
        f"{len(flagged):,} flagged in logs/check_electorate.csv, {len(hard)} of them from 2008"
    )
    if len(hard) > 0:
        raise ValueError(f"DUN electorates do not sum to the parliament's for {len(hard)} seats!")

    df = pd.read_parquet("data/consol_ballots.parquet")
    for v in ["party", "coalition", "candidate"]:
        cf = pd.read_csv(f"data/lookup_{v}.csv")
        assert len(df[df[f"{v}_uid"].isin(cf[f"{v}_uid"])]) == len(
            df
        ), f"Missing {v} in lookup file!"

    print("Validation passed!")

    print("\n\n--------- ✨✨✨ DONE ✨✨✨ ----------\n")  # Not AI; I like sparkles after success


if __name__ == "__main__":
    main()
