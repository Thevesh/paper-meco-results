"""Module for generating dashboard data and visualizations for election results."""

import os
import time
from datetime import datetime
import requests
import pandas as pd
import duckdb

from helper import generate_slug, get_states, write_parquet

PATH = "dashboards"


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
        None
    """
    df = pd.read_csv("data/lookup_dates.csv").rename(columns={"election_number": "election"})
    df.election = "SE-" + df.election.astype(str).str.zfill(2)
    df.loc[df.state == "Malaysia", "election"] = df.election.str.replace("SE", "GE")
    df = pd.concat(
        [df[df.state == "Malaysia"], df[df.state != "Malaysia"]],
        axis=0,
        ignore_index=True,
    )
    print(f"\n{len(df):,.0f} unique elections")
    write_parquet(f"{PATH}/elections_dates", df=df)


def make_candidates():
    """
    This function:
    - Reads and wrangles consolidated ballots file
    - Reads consolidated stats file, and merges it onto the ballots (join on contest identifiers)
    - Saves the combined result to dashboards/elections_candidates.parquet
    The resulting output is used as the base for all other dashboard files.

    Inputs (cleaned and validated):
    - data/consol_ballots.parquet
    - data/consol_stats.parquet

    Outputs:
    - dashboards/elections_candidates.parquet

    Returns:
        None
    """
    # fmt: off
    col_join = ["date", "election_name", "state", "seat"]
    col_ballot = [
        "slug", "name", "type",
    ] + col_join + [
        "party", "party_uid","coalition","coalition_uid",
        "votes", "votes_perc", "result",
    ]
    col_stats = col_join + [
        "voters_total", "n_candidates",
        "voter_turnout", "voter_turnout_perc", 
        "votes_rejected", "votes_rejected_perc", 
        "majority", "majority_perc",
        "ballots_not_returned", "ballots_not_returned_perc",
    ]
    # fmt: on

    df = pd.read_parquet("data/consol_ballots.parquet").rename(
        columns={"candidate_uid": "slug", "election": "election_name"}
    )
    df["type"] = "parlimen"
    df.loc[df.seat.str.startswith("N."), "type"] = "dun"
    df.seat = df.seat + ", " + df.state
    df = df[col_ballot]

    sf = pd.read_parquet("data/consol_stats.parquet").rename(
        columns={
            "voter_turnout": "voter_turnout_perc",
            "ballots_issued": "voter_turnout",
            "election": "election_name",
        }
    )
    sf.seat = sf.seat + ", " + sf.state
    sf = sf[col_stats]

    df = pd.merge(df, sf, on=["date", "election_name", "state", "seat"], how="left")
    df.election_name = df.election_name.replace("BY-ELECTION", "By-Election")
    print(f"\n{len(df.slug.unique()):,.0f} unique candidates")
    write_parquet(f"{PATH}/elections_candidates", df=df)


def make_seats_winner():
    """
    This function
    - Reads and wrangles the candidate file
    - Filters for only winning candidates
    - Generates a slug for each seat

    Inputs:
    - dashboards/elections_candidates.parquet

    Outputs:
    - dashboards/elections_seats_winner.parquet

    Returns:
        int: The number of rows in the final DataFrame, corresponding to the number of contests
    """
    # fmt: off
    col_final = [
        "slug", "seat_name", "seat", "state",
        "date", "election_name", "type",
        "party", "party_uid", "coalition", "coalition_uid", "name",
        "majority", "majority_perc", "voter_turnout", "voter_turnout_perc", "votes_rejected", "votes_rejected_perc",
    ]
    # fmt: on

    df = duckdb.query(
        f"SELECT * FROM read_parquet('{PATH}/elections_candidates.parquet') WHERE result LIKE 'won%'"
    ).df()
    df["seat_name"] = df.seat.str[6:]
    df.loc[df.type == "dun", "seat_name"] = df.seat.str[5:]
    df["slug"] = df.seat.apply(generate_slug)
    df = df[col_final]
    print(f"\n{len(df):,.0f} unique contests")
    write_parquet(f"{PATH}/elections_seats_winner", df=df)


def make_parties_coalitions():
    """
    This function:
    - Reads and processes candidate data
    - Drops unnecessary columns
    - Filters out By-Elections (retaining only general/state elections)
    - Aggregates results by party and coalition
    - Calculates total seats/votes and their percentages per election and state
    - Handles special cases such as uncontested seats
    - Writes final party-level and coalition-level aggregates to parquet files

    Inputs:
    - dashboards/elections_candidates.parquet
    - data/lookup_dates.parquet

    Outputs:
    - dashboards/elections_parties.parquet (results grouped by party)
    - dashboards/elections_coalitions.parquet (results grouped by coalition)

    Returns:
        None
    """

    ds = pd.read_parquet("data/lookup_dates.parquet")
    ds.election_number = "GE-" + ds.election_number.astype(str).str.zfill(2)
    ds.loc[ds.state != "Malaysia", "election_number"] = ds.election_number.str.replace("GE-", "SE-")
    ds = ds.rename(columns={"election_number": "election_name"})
    map_ge_date = dict(zip(ds.election_name, ds.date))

    col_idx = [
        "party_uid",
        "party",
        "coalition",
        "coalition_uid",
        "type",
        "state",
        "election_name",
        "date",
    ]
    df = pd.read_parquet(f"{PATH}/elections_candidates.parquet")
    df = df[df.election_name != "By-Election"]  # Remove By-Elections, we are not interested in them
    df = df.drop(
        [
            "seat",
            "date",
            "voter_turnout",
            "voter_turnout_perc",
            "votes_rejected",
            "votes_rejected_perc",
            "majority",
            "majority_perc",
        ],
        axis=1,
    )
    df["seats_contested"] = 1
    df["seats_won"] = 0
    df.loc[df.result.str.contains("won"), "seats_won"] = 1
    df = pd.merge(df, ds, on=["state", "election_name"], how="left")
    df.loc[df.election_name.str.contains("GE-"), "date"] = df.election_name.map(map_ge_date)
    df = pd.concat(
        [df[df.election_name.str.contains("GE-")].assign(state="Malaysia"), df],
        axis=0,
        ignore_index=True,
    )
    df = (
        df.drop(["name", "votes_perc", "result", "slug"], axis=1)
        .groupby(col_idx)
        .sum()
        .reset_index()
    )

    # add total number of seats and votes per election (sf), then compute percentages
    col_idx_sf = ["election_name", "state"]
    sf = df[col_idx_sf + ["votes", "seats_won"]].copy().groupby(col_idx_sf).sum().reset_index()
    sf.columns = col_idx_sf + ["votes_total", "seats_total"]
    df = pd.merge(df, sf, on=col_idx_sf, how="left")
    df["votes_perc"] = df.votes / df.votes_total * 100
    df["seats_contested_perc"] = df.seats_contested / df.seats_total * 100
    df["seats_won_perc"] = df.seats_won / df.seats_total * 100
    df.loc[(df.election_name == "SE-02") & (df.state == "Sabah"), "votes_perc"] = (
        df.seats_contested_perc
    )  # special case where all seats were uncontested
    df = df[
        col_idx
        + ["seats_contested", "seats_won", "seats_total", "seats_contested_perc", "seats_won_perc"]
        + ["votes", "votes_total", "votes_perc"]
    ]

    df[(df.election_name == "SE-02") & (df.state == "Sabah")].sort_values(
        by="seats_won", ascending=False
    )
    print(f"\n{len(df.party_uid.unique()):,.0f} unique parties")
    print(f"{len(df.coalition_uid.unique()):,.0f} unique coalitions")
    write_parquet(f"{PATH}/elections_parties", df=df)

    df = df.drop(columns=["party_uid", "party"])
    df = df.groupby(col_idx[2:]).sum().reset_index()
    df.seats_total = ((df.seats_contested * 100) / (df.seats_contested_perc)).round(0).astype(int)
    df.votes_total = ((df.votes * 100) / (df.votes_perc)).round(0).astype(int)
    for c in ["votes_total", "seats_total"]:
        assert len(df.drop_duplicates(subset=["election_name", "state"])) == len(
            df.drop_duplicates(subset=["election_name", "state", c])
        )
    write_parquet(f"{PATH}/elections_coalitions", df=df)


def make_election_summaries():
    """
    The function:
        Reads election summary data from consol_summary.parquet,
        then filters out By-Elections,
        then filters out uncontested wins,
        then groups by state and type,
        then calculates voter turnout and rejected votes percentages,
        then writes final dataframe to parquet format.
    Returns:
        Number of rows in the final dataframe, representing the number of general elections
    """
    # fmt: off
    col_idx = ["type","election_name","state"]
    col_summary = [
        "voters_total", "n_candidates",
        "voter_turnout", "voter_turnout_perc", 
        "votes_valid", "votes_rejected", "votes_rejected_perc", 
    ]
    # fmt: on

    df = pd.read_parquet("dashboards/elections_candidates.parquet")
    df["votes_valid"] = df.voter_turnout - df.ballots_not_returned
    df["voters_contested"] = df["voters_total"]
    df.loc[df.result.str.contains("uncontested"), "voters_contested"] = 0
    df = df[(df.result.str.contains("won")) & (~df.election_name.str.contains("By-Election"))]
    df = df[col_idx + col_summary]
    df = pd.concat(
        [df[df.election_name.str.contains("GE-")].assign(state="Malaysia"), df],
        axis=0,
        ignore_index=True,
    )
    df = df.groupby(col_idx).sum().reset_index()
    df["voter_turnout_perc"] = df.voter_turnout / df.voters_total * 100
    df["votes_rejected_perc"] = df.votes_rejected / df.votes_valid * 100
    df = df.sort_values(by=["type", "state", "election_name"], ascending=[False, True, True])
    df = pd.concat(
        [df[df.state == "Malaysia"], df[df.state != "Malaysia"]], axis=0, ignore_index=True
    )
    print(f"\n{len(df):,.0f} unique permutations")
    write_parquet(f"{PATH}/elections_summary", df=df)


if __name__ == "__main__":
    START = datetime.now()
    print(f"\nStart: {START}")
    print("\nGenerating dashboard files:")
    make_dates()
    make_candidates()
    make_seats_winner()
    make_parties_coalitions()
    make_election_summaries()
    print(f"\nEnd: {datetime.now()}")
    print(f"\nDuration: {datetime.now() - START}\n")
