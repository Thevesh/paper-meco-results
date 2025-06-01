"""Module for generating and uploading dashboard data to S3."""

import json as j
import os
from glob import glob as g
from datetime import datetime
import pandas as pd

from helper import get_states, upload_s3, upload_s3_bulk


def make_candidates():
    """Generate candidate data files for API."""
    data = {"data": []}

    col_api_candidate = [
        "name",
        "election_name",
        "type",
        "date",
        "seat",
        "party",
        "votes",
        "votes_perc",
        "result",
    ]

    df = pd.read_parquet("src-data/dashboards/elections_candidates.parquet")
    for slug in sorted(list(df.slug.unique())):
        tf = (
            df[df.slug == slug]
            .copy()[col_api_candidate]
            .sort_values(by="date", ascending=False)
        )
        tf = tf.to_dict(orient="records")
        tf = [
            {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in tf
        ]  # proper JSON null
        data["data"] = tf
        with open(f"api/candidates/{slug}.json", "w", encoding="utf-8") as f:
            j.dump(data, f)


def make_seats():
    """Generate seat data files for API."""
    data = {"data": []}

    col_api_seat = [
        "election_name",
        "seat",
        "date",
        "party",
        "name",
        "majority",
        "majority_perc",
    ]

    df = pd.read_parquet("src-data/dashboards/elections_seats_winner.parquet")
    df.slug = df.type + "-" + df.slug
    for slug in df.slug.unique():
        tf = (
            df[df.slug == slug]
            .copy()[col_api_seat]
            .sort_values(by="date", ascending=False)
        )
        tf = tf.to_dict(orient="records")
        tf = [
            {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in tf
        ]  # proper JSON null
        data["data"] = tf
        with open(f"api/seats/{slug}.json", "w", encoding="utf-8") as f:
            j.dump(data, f)


def make_parties():
    """Generate party data files for API."""
    data = {"data": []}

    col_party = [
        "state",
        "type",
        "election_name",
        "date",
        "seats",
        "seats_total",
        "seats_perc",
        "votes",
        "votes_perc",
    ]

    df = pd.read_parquet("src-data/dashboards/elections_parties.parquet")
    for party in df.party.unique():
        tf = df[df.party == party].copy()

        # loop over parlimen and dun
        for election_type in tf["type"].unique():
            if not os.path.exists(f"api/parties/{party}/{election_type}"):
                os.makedirs(f"api/parties/{party}/{election_type}")
            tft = tf[tf.type == election_type].copy()

            # loop over states
            for state in tft.state.unique():
                if election_type == "dun" and state == "Malaysia":
                    continue

                tfts = (
                    tft[tft.state == state]
                    .copy(col_party)
                    .sort_values(by="date", ascending=False)
                )
                res = tfts.to_dict(orient="records")
                res = [
                    {k: (None if pd.isna(v) else v) for k, v in record.items()}
                    for record in res
                ]  # proper JSON null

                data["data"] = res
                with open(f"api/parties/{party}/{election_type}/{state}.json",
                          "w", encoding="utf-8") as f:
                    j.dump(data, f)


def make_results():
    """Generate result data files for API."""
    print("")
    data = {"ballot": [], "summary": []}

    col_api_ballot = ["name", "party", "votes", "votes_perc", "result"]
    col_api_ballot_summary = [
        "date",
        "voter_turnout",
        "voter_turnout_perc",
        "votes_rejected",
        "votes_rejected_perc",
        "majority",
        "majority_perc",
    ]

    df = pd.read_parquet("src-data/dashboards/elections_candidates.parquet")
    print(
        f"{df.drop_duplicates(subset=['seat','date']).shape[0]:,.0f} results to create"
    )
    for seat in df.seat.unique():
        if not os.path.exists(f"api/results/{seat}"):
            os.makedirs(f"api/results/{seat}")

        dfs = df[df.seat == seat].copy()
        for date in dfs.date.unique():
            dfse = (
                dfs[dfs.date == date]
                .copy()[col_api_ballot]
                .sort_values(by="votes", ascending=False)
            )
            dfse_b = (
                dfs[dfs.date == date].copy()[col_api_ballot_summary].drop_duplicates()
            )

            res = dfse.to_dict(orient="records")
            res = [
                {k: (None if pd.isna(v) else v) for k, v in record.items()}
                for record in res
            ]  # proper JSON null

            res_b = dfse_b.to_dict(orient="records")
            res_b = [
                {k: (None if pd.isna(v) else v) for k, v in record.items()}
                for record in res_b
            ]  # proper JSON null

            data["ballot"] = res
            data["summary"] = res_b
            with open(f"api/results/{seat}/{date}.json", "w", encoding="utf-8") as f:
                j.dump(data, f)


def make_elections():
    """Generate election data files for API."""
    col_combo = ["state", "type", "election_name"]
    col_final = {
        "ballot": [
            "party",
            "seats",
            "seats_total",
            "seats_perc",
            "votes",
            "votes_perc",
        ],
        "summary": [
            "voter_turnout",
            "voter_turnout_perc",
            "votes_rejected",
            "votes_rejected_perc",
        ],
        "stats": [
            "seat",
            "date",
            "party",
            "party_lost",
            "name",
            "state",
            "majority",
            "majority_perc",
            "voter_turnout",
            "voter_turnout_perc",
            "votes_rejected",
            "votes_rejected_perc",
        ],
    }

    # dfm for main ballot by party
    dfm = pd.read_parquet("src-data/dashboards/elections_parties.parquet").sort_values(
        by=["seats_perc", "votes_perc"], ascending=False
    )

    # dfs for summary stats
    dfs = pd.read_parquet("src-data/dashboards/elections_summary.parquet").fillna(0)

    # dft for table of statistics by seat; need to be joined with lf loser frame
    dft = pd.read_parquet("src-data/dashboards/elections_seats_winner.parquet")
    dft = dft[dft.election_name != "By-Election"]
    dft = pd.concat(
        [dft[dft.type == "parlimen"].assign(state="Malaysia"), dft],
        axis=0,
        ignore_index=True,
    )
    lf = pd.read_parquet("src-data/consol_ballots.parquet")
    lf = lf[lf.result != "won"]
    lf.loc[lf.result == "won_uncontested", "party"] = "NEMO"
    lf.seat = lf.seat + ", " + lf.state
    lf = (
        lf[["date", "seat", "party"]]
        .drop_duplicates()
        .rename(columns={"party": "party_lost"})
    )
    lf = lf.groupby(["date", "seat"])["party_lost"].agg(list).reset_index()
    dft = pd.merge(dft, lf, on=["date", "seat"], how="left")

    assert (
        len(dfm.drop_duplicates(subset=col_combo))
        == len(dfs.drop_duplicates(subset=col_combo))
        == len(dft.drop_duplicates(subset=col_combo))
    ), f"Mismatch between 3 components!\
            ballots: {len(dfm.drop_duplicates(subset=col_combo))} \
            summaries: {len(dfs.drop_duplicates(subset=col_combo))} \
            stats: {len(dft.drop_duplicates(subset=col_combo))}"

    df = {"ballot": dfm, "summary": dfs, "stats": dft}

    for election_type in dfm.type.unique():
        tf = dfm[dfm.type == election_type].copy()
        for state in tf.state.unique():
            tf = dfm[(dfm.type == election_type) & (dfm.state == state)].copy().copy()
            for election in tf.election_name.unique():

                # ensure state folder exists
                if not os.path.exists(f"api/elections/{state}"):
                    os.makedirs(f"api/elections/{state}")

                # now loop over the keys
                data = {"ballot": [], "summary": [], "stats": []}
                for key, value in df.items():
                    tf = value.copy()
                    tf = tf[
                        (tf.type == election_type)
                        & (tf.state == state)
                        & (tf.election_name == election)
                    ]
                    res = tf[col_final[key]].to_dict(orient="records")
                    res = [
                        {
                            k: (
                                (None if pd.isna(v) else v)
                                if not isinstance(v, list)
                                else v
                            )
                            for k, v in record.items()
                        }
                        for record in res
                    ]  # proper JSON null
                    res = [
                        {
                            k: [] if isinstance(v, list) and v == ["NEMO"] else v
                            for k, v in record.items()
                        }
                        for record in res
                    ]
                    data[key] = res
                with open(f"api/elections/{state}/{election_type}-{election}.json",
                          "w", encoding="utf-8") as f:
                    j.dump(data, f)


def make_trivia():
    """Generate trivia data files for API."""
    states = get_states(my=1)

    sb = pd.read_parquet("src-data/dashboards/elections_slim_big.parquet").sort_values(
        by="majority"
    )
    vt = pd.read_parquet("src-data/dashboards/elections_veterans.parquet")

    for state in states:
        df = {
            "slim_big": sb[sb.state == state].copy().drop("state", axis=1),
            "veterans_parlimen": vt[(vt.type == "parlimen") & (vt.state == state)]
            .copy()
            .drop(["type", "state"], axis=1),
            "veterans_dun": vt[(vt.type == "dun") & (vt.state == state)]
            .copy()
            .drop(["type", "state"], axis=1),
        }

        data = {"slim_big": [], "veterans_parlimen": [], "veterans_dun": []}
        for key, _ in data.items():
            tf = df[key].copy()
            res = tf.to_dict(orient="records")
            res = [
                {k: (None if pd.isna(v) else v) for k, v in record.items()}
                for record in res
            ]  # proper JSON null
            data[key] = res
        with open(f"api/trivia/{state}.json", "w", encoding="utf-8") as f:
            j.dump(data, f)


def upload_data(file_pattern="candidates/*"):
    """Upload data files to S3."""
    files = g(f"api/{file_pattern}.json")
    files_to_upload = sorted([(f, f.replace("api/", "")) for f in files])

    upload_s3_bulk(
        bucket_name="static.electiondata.my",
        files_to_upload=files_to_upload,
        max_workers=120,
    )


def make_upload_dates():
    """Generate and upload dates data file."""
    data = {"data": []}
    df = (
        pd.read_csv("src-data/lookup_seats.csv")[["state", "election", "date"]]
        .drop_duplicates()
        .sort_values(by=["state", "election"])
    )
    df = df[~df.election.str.contains("BY-ELECTION")]
    df = pd.concat(
        [
            df[df.election.str.startswith("GE")]
            .assign(state="Malaysia")
            .drop_duplicates(),
            df,
        ],
        axis=0,
        ignore_index=True,
    )
    res = df.to_dict(orient="records")
    data["data"] = res
    with open("api/dates.json", "w", encoding="utf-8") as f:
        j.dump(data, f)

    print(
        upload_s3(
            bucket_name="static.electiondata.my",
            source_file_name="api/dates.json",
            cloud_file_name="dates.json",
        )
    )


if __name__ == "__main__":
    START = datetime.now()
    print(f'\nStart: {START.strftime("%Y-%m-%d %H:%M:%S")}')
    make_candidates()
    make_seats()
    make_parties()
    make_results()
    make_elections()
    make_trivia()
    for path in [
        "candidates/*",
        "seats/*",
        "parties/*/*/*",
        "results/*/*",
        "elections/*/*",
        "trivia/*",
    ]:
        upload_data(file_pattern=path)
    make_upload_dates()
    print(f'\nEnd: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"\nDuration: {datetime.now() - START}\n")
