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
"""

import pandas as pd
import numpy as np

from helper import get_states, write_csv_parquet


def main():
    """
    Compile and validate election data from raw source files.
    Created as a function to prevent unsafe imports.
    """
    states = get_states()
    col_group = ["date", "election", "state", "seat"]

    print("\n --------- Compiling ballots ----------\n")

    df = pd.read_csv("src-data/raw_ballots.csv")
    df.date = pd.to_datetime(df.date).dt.date
    assert len(df[~df.state.isin(states)]) == 0, "Invalid state in raw ballots file!"

    df = df[
        ~((pd.to_datetime(df.date).dt.year == 1988) & (df.election.str.startswith("BY-")))
    ]  # Remove Johor 1988 by-elections

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

    write_csv_parquet(
        "src-data/consol_ballots", df.drop(["votes_valid", "n_candidates", "majority"], axis=1)
    )

    print("\n\n --------- Compiling stats ----------\n")

    sf = pd.read_csv("src-data/raw_stats.csv")
    sf.date = pd.to_datetime(sf.date).dt.date
    assert len(sf[~sf.state.isin(states)]) == 0, "Invalid state in raw stats file!"

    sf = sf[
        ~((pd.to_datetime(sf.date).dt.year == 1988) & (sf.election.str.startswith("BY-")))
    ]  # Remove Johor 1988 by-elections

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

    write_csv_parquet("src-data/consol_stats", df)

    print("\n\n --------- Validating files ----------\n")

    df["check"] = df.ballots_issued - df.ballots_not_returned - df.votes_rejected - df.votes_valid
    if len(df[df.check != 0]) > 0:
        df = df.sort_values(by=["date", "state", "seat"])
        df = df[["check"] + list(df.columns[:-1])]
        df[df.check != 0].to_csv("logs/check.csv", index=False)
        raise ValueError(f"Validation failed for {len(df[df.check != 0])} seats!")

    df = pd.read_parquet("src-data/consol_ballots.parquet")
    for v in ["party", "coalition", "candidate"]:
        cf = pd.read_csv(f"src-data/lookup_{v}.csv")
        assert len(df[df[f"{v}_uid"].isin(cf[f"{v}_uid"])]) == len(
            df
        ), f"Missing {v} in lookup file!"

    print("Validation passed!")

    print("\n\n --------- Generating SFC files ----------\n")

    for v in ["ballots", "stats"]:
        df = pd.read_parquet(f"src-data/consol_{v}.parquet")
        write_csv_parquet(f"src-data/sfc/federal_{v}", df[df.election.str.startswith("GE-")])
        write_csv_parquet(f"src-data/sfc/byelection_{v}", df[df.election.str.startswith("BY-")])

        for state, state_code in zip(get_states(my=0, codes=0), get_states(my=0, codes=1)):
            if "W.P." in state:
                continue
            write_csv_parquet(
                f"src-data/sfc/state_{state_code}_{v}",
                df[(df.state == state) & (df.election.str.startswith("SE-"))],
            )

    print("\n\n --------- Generating lookup parquets ----------\n")

    for v in ["candidate", "party", "coalition", "party_succession", "dates"]:
        cf = pd.read_csv(f"src-data/lookup_{v}.csv")
        for c in cf.columns:
            if c in ["date"]:
                cf[c] = pd.to_datetime(cf[c]).dt.date
        write_csv_parquet(f"src-data/lookup_{v}", cf)

    print("\n\n --------- ✨✨✨ DONE ✨✨✨ ----------\n")  # Not AI; I like sparkles after success


if __name__ == "__main__":
    main()
