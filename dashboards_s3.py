"""Module for generating and uploading dashboard data to S3."""

import json as j
import os
from glob import glob as g
from datetime import datetime
import pandas as pd

from helper import get_s3_client, upload_single, upload_bulk, generate_slug


def make_candidates():
    """Generate candidate data files for API."""
    data = {"data": []}

    df = pd.read_parquet("dashboards/elections_candidates.parquet")
    df = (
        df.assign(c=1, w=df.result.str.contains("won").astype(int), l=lambda x: 1 - x.w)
        .groupby(["slug", "name"], as_index=False)
        .agg({"c": "sum", "w": "sum", "l": "sum"})
        .sort_values(["c", "w"], ascending=False)
    )
    df = df.to_dict(orient="records")
    df = [
        {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in df
    ]  # proper JSON null
    data["data"] = df
    with open("api/candidates/dropdown.json", "w", encoding="utf-8") as f:
        j.dump(data, f)

    col_api_candidate = [
        "name",
        "election_name",
        "type",
        "date",
        "seat",
        "party",
        "party_uid",
        "coalition",
        "coalition_uid",
        "votes",
        "votes_perc",
        "result",
    ]

    df = pd.read_parquet("dashboards/elections_candidates.parquet")
    df.date = pd.to_datetime(df.date).dt.strftime("%Y-%m-%d")
    for slug in sorted(list(df.slug.unique())):
        tf = df[df.slug == slug].copy()[col_api_candidate].sort_values(by="date", ascending=False)
        tf = tf.to_dict(orient="records")
        tf = [
            {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in tf
        ]  # proper JSON null
        data["data"] = tf
        with open(f"api/candidates/{slug}.json", "w", encoding="utf-8") as f:
            j.dump(data, f)


def make_seats():
    """Generate seat data files for API."""
    data = {
        "desc_en": "",
        "desc_ms": "",
        "voters_total": 0,
        "boundaries": {},
        "lineage": [],
        "data": [],
        "barmeter": {},
        "pyramid": {"ages": [], "male": [], "female": []},
    }

    col_api_seat = [
        "election_name",
        "seat",
        "state",
        "date",
        "party",
        "party_uid",
        "coalition",
        "coalition_uid",
        "name",
        "majority",
        "majority_perc",
        "voter_turnout",
        "voter_turnout_perc",
    ]

    df = pd.read_parquet("dashboards/elections_seats_winner.parquet")
    df = df[
        (df.election_name == "GE-15")
        | (df.election_name == "SE-15")
        | ((df.state == "Sarawak") & (df.election_name == "SE-12"))
    ]
    slugs = df.slug.tolist()

    sf = pd.read_parquet("dashboards/elections_seats_winner.parquet")
    sf.date = pd.to_datetime(sf.date).dt.date.astype(str)
    lf = pd.read_csv("api/lineage_filter.csv")
    lf.current_seat = lf.current_seat.apply(generate_slug)
    lf.seat = lf.seat + ", " + lf.state
    sf = pd.merge(sf, lf, on=["election_name", "state", "seat"], how="left")

    # lf: lineage frame
    lf = pd.read_csv("api/lineage_desc.csv")
    lf = lf[lf.change_en != "Unchanged"]
    lf = lf[~lf.change_en.str.contains("was not renamed, its boundaries were changed")]
    lf.seat = lf.seat.apply(generate_slug)
    lf = lf.rename(columns={"seat": "slug"})

    bf = pd.read_csv("local-scrape/voters_ge15_demog.csv")
    bf = pd.concat(
        [
            bf[~bf.state.str.contains("W.P")],
            bf.groupby(["state", "parlimen"]).sum(numeric_only=True).reset_index(),
        ],
        axis=0,
        ignore_index=True,
    )
    # for c in bf.columns[4:]: bf[c] = (bf[c]/bf['total'] * 100).round(1)
    bf["slug"] = bf.parlimen + ", " + bf.state
    bf.loc[~bf.dun.isnull(), "slug"] = bf.dun + ", " + bf.state
    bf.slug = bf.slug.apply(generate_slug)
    bf["desc_en"] = bf.apply(
        lambda x: f"{x.parlimen} is a federal constituency in {x.state}, with {int(x.total):,} voters as of GE-15 (2022).",
        axis=1,
    )
    bf["desc_ms"] = bf.apply(
        lambda x: f"{x.parlimen} adalah sebuah kawasan persekutuan di {x.state}, dengan {int(x.total):,} pengundi setakat GE-15 (2022).",
        axis=1,
    )
    bf.loc[~bf.dun.isnull(), "desc_en"] = bf.loc[~bf.dun.isnull()].apply(
        lambda x: f"{x.dun} is a state constituency in {x.state}, with {int(x.total):,} voters as of GE-15 (2022).",
        axis=1,
    )
    bf.loc[~bf.dun.isnull(), "desc_ms"] = bf.loc[~bf.dun.isnull()].apply(
        lambda x: f"{x.dun} adalah sebuah kawasan negeri di {x.state}, dengan {int(x.total):,} pengundi setakat GE-15 (2022).",
        axis=1,
    )

    af = pd.read_parquet("local-scrape/voters_ge15_pyramid.parquet")

    for slug in slugs:

        # Election results
        seat_type = sf[sf.slug == slug].type.iloc[0]
        tf = (
            sf[(sf.current_seat == slug) & (sf.type == seat_type)]
            .copy()[col_api_seat]
            .sort_values(by="date", ascending=False)
        )
        tf.seat = tf.seat.str.split(",").str[0]
        tf = tf.to_dict(orient="records")
        tf = [
            {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in tf
        ]  # proper JSON null

        # Lineage descriptions
        tfl = lf[lf.slug == slug].copy().drop("slug", axis=1)
        tfl = tfl.to_dict(orient="records")
        tfl = [
            {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in tfl
        ]  # proper JSON null

        # Demographics
        tfd = bf[bf.slug == slug].iloc[0]
        barmeter = {"votertype": {}, "sex": {}, "age": {}, "ethnic": {}}
        for k in barmeter:
            prefix = k + "_"
            barmeter[k] = {}
            for col in bf.columns:
                if col.startswith(prefix):
                    barmeter[k][col[len(prefix) :]] = int(tfd[col])

        # Pyramid
        tfa = af[af.slug == slug].copy()
        if len(tfa) == 0:
            print(f"No pyramid data for {slug}")
        data["pyramid"]["ages"] = list(range(18, 101))
        data["pyramid"]["male"] = [int(x) for x in tfa.male.tolist()]
        data["pyramid"]["female"] = [int(x) for x in tfa.female.tolist()]

        # Combine into JSON response
        data["desc_en"] = tfd["desc_en"]
        data["desc_ms"] = tfd["desc_ms"]
        data["voters_total"] = int(tfd["total"])
        data["data"] = tf + tfl
        data["data"].sort(key=lambda x: x.get("date", ""), reverse=True)
        data["barmeter"] = barmeter

        with open(f"api/seats/current/{slug}.json", "w", encoding="utf-8") as f:
            j.dump(data, f)


def make_seats_boundaries():
    """
    Generate data for the boundaries key in the seat JSONs
    """
    # read and form consolidated dataframe with latest year
    df = pd.read_csv("local-scrape/lineage_simple.csv").rename(columns={"year": "tileset"})
    df["year"] = df.tileset.astype(str)
    cf = df.copy().drop_duplicates(subset=["state", "seat"])
    cf["tileset"] = cf.state.map({"Sabah": "2019", "Sarawak": "2015"}).fillna("2018")
    cf.year = cf.tileset
    cf.lineage = cf.seat
    df = pd.concat([df, cf], axis=0, ignore_index=True)
    df = df.sort_values(by=["state", "seat", "year"], ascending=[True, True, False]).reset_index(
        drop=True
    )

    df.tileset = "peninsular_" + df.tileset.astype(str) + "_parlimen"
    df.loc[df.seat.str.startswith("N"), "tileset"] = df.tileset.str.replace("parlimen", "dun")
    for s in ["Sabah", "Sarawak"]:
        df.loc[df.state == s, "tileset"] = df.tileset.str.replace("peninsular", s.lower())
    df["slug"] = df.seat + ", " + df.state
    df["slug"] = df.slug.apply(generate_slug)
    df = df[["slug", "state", "seat", "year", "lineage", "tileset"]]

    bf = pd.read_csv("local-scrape/plot_bounds.csv")
    df = df.merge(bf, on="slug", how="left")
    df.zoom = df.zoom.round(2) - 0.5

    files = g("api/seats/current/*.json")
    files = sorted([x for x in files if "dropdown" not in x])

    for f in files:
        data = j.load(open(f, "r", encoding="utf-8"))
        slug = f.replace("api/seats/current/", "").replace(".json", "")
        tf = df[df.slug == slug].copy()
        if len(tf) == 0:
            print(f"{slug} not found")
            continue

        boundaries = {"center": [], "zoom": 0, "polygons": {}}

        data["boundaries"] = boundaries

        data["boundaries"]["center"] = [
            float(tf.center_lon.values[0]),
            float(tf.center_lat.values[0]),
        ]
        data["boundaries"]["zoom"] = float(tf.zoom.values[0])
        for _, row in tf.iterrows():
            year = row["year"]
            tileset = row["tileset"]
            lineage = row["lineage"]
            # lineage may be a comma-separated list, but in this context, it's a single value
            # If lineage is a string with multiple values, split and strip
            if isinstance(lineage, str) and "," in lineage:
                lineage_list = [x.strip() for x in lineage.split(",")]
            else:
                lineage_list = [lineage]
            data["boundaries"]["polygons"][year] = [tileset, lineage_list]

        j.dump(data, open(f, "w", encoding="utf-8"))


def make_seats_lineage():
    """
    Generate data for the lineage key in the seat JSONs
    """
    df = {
        "p": pd.read_csv("local-scrape/lineage_naive_parlimen.csv"),
        "n": pd.read_csv("local-scrape/lineage_naive_dun.csv"),
    }
    col_final = {
        "p": ["year", "parlimen", "area", "overlap_pct", "n_duns", "duns"],
        "n": ["year", "dun", "area", "overlap_pct", "parlimen"],
    }
    for t in ["p", "n"]:
        df[t]["slug"] = df[t].new + ", " + df[t].state
        df[t].slug = df[t].slug.apply(generate_slug)

    files = g("api/seats/current/*.json")
    files = sorted([x for x in files if "dropdown" not in x])

    for f in files:
        data = j.load(open(f, "r", encoding="utf-8"))
        slug = f.replace("api/seats/current/", "").replace(".json", "")

        tf = df[slug[0]][df[slug[0]].slug == slug][col_final[slug[0]]]

        # insert complete lineage
        seat_type = col_final[slug[0]][1]
        for y in tf.year.unique():
            seats = tf[tf.year == y][seat_type].tolist()
            data["boundaries"]["polygons"][f"{y}"][1] = seats

        tf = tf.to_dict(orient="records")
        tf = [
            {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in tf
        ]  # proper JSON null
        data["lineage"] = tf
        j.dump(data, open(f, "w", encoding="utf-8"))


def make_parties():
    """Generate party data files for API."""
    data = {"data": []}

    col_party = [
        "state",
        "type",
        "coalition",
        "coalition_uid",
        "election_name",
        "date",
        "seats_contested",
        "seats_won",
        "seats_total",
        "seats_contested_perc",
        "seats_won_perc",
        "votes",
        "votes_perc",
    ]

    df = pd.read_parquet("dashboards/elections_parties.parquet")
    df.coalition_uid = df.coalition_uid.astype(str).str.zfill(2) + "-" + df.coalition
    tf = pd.read_parquet("dashboards/elections_coalitions.parquet").rename(
        columns={"coalition_uid": "party_uid", "coalition": "party"}
    )
    tf.party_uid = tf.party_uid.astype(str).str.zfill(2) + "-" + tf.party
    tf["coalition"] = tf["coalition_uid"] = "-"
    df = pd.concat(
        [df.assign(party_type="party"), tf.assign(party_type="coalition")],
        axis=0,
        ignore_index=True,
    )
    df.date = pd.to_datetime(df.date).dt.date.astype(str)

    for party in df.party_uid.unique():
        tf = df[df.party_uid == party].copy()
        party_type = "parties" if tf.party_type.iloc[0] == "party" else "coalitions"

        # loop over parlimen and dun
        for election_type in tf["type"].unique():
            if not os.path.exists(f"api/{party_type}/{party}/{election_type}"):
                os.makedirs(f"api/{party_type}/{party}/{election_type}")
            tft = tf[tf.type == election_type].copy()

            # loop over states
            for state in tft.state.unique():
                if election_type == "dun" and state == "Malaysia":
                    continue

                tfts = (
                    tft[tft.state == state].copy(col_party).sort_values(by="date", ascending=False)
                )
                res = tfts.to_dict(orient="records")
                res = [
                    {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in res
                ]  # proper JSON null

                data["data"] = res
                with open(
                    f"api/{party_type}/{party}/{election_type}/{state}.json", "w", encoding="utf-8"
                ) as f:
                    j.dump(data, f)


def make_results():
    """Generate result data files for API."""
    print("")
    data = {"ballot": [], "summary": []}

    # fmt: off
    col_api_ballot = ["name", "party_uid", "party", "coalition_uid", "coalition", "votes", "votes_perc", "result"]
    col_api_ballot_summary = ["date", "voter_turnout", "voter_turnout_perc", "votes_rejected", "votes_rejected_perc", "majority", "majority_perc"]
    # fmt: on

    df = pd.read_parquet("dashboards/elections_candidates.parquet")
    df.date = pd.to_datetime(df.date).dt.date.astype(str)
    print(f"{df.drop_duplicates(subset=['seat','date']).shape[0]:,.0f} results to create")
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
            dfse_b = dfs[dfs.date == date].copy()[col_api_ballot_summary].drop_duplicates()

            res = dfse.to_dict(orient="records")
            res = [
                {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in res
            ]  # proper JSON null

            res_b = dfse_b.to_dict(orient="records")
            res_b = [
                {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in res_b
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
            "party_uid",
            "party",
            "coalition",
            "coalition_uid",
            "seats_contested",
            "seats_won",
            "seats_total",
            "seats_contested_perc",
            "seats_won_perc",
            "votes",
            "votes_total",
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
            "party_uid",
            "party_lost",
            "name",
            "n_candidates",
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
    dfm = pd.read_parquet("dashboards/elections_parties.parquet").sort_values(
        by=["seats_won_perc", "votes_perc"], ascending=False
    )
    dfm.coalition_uid = dfm.coalition_uid.astype(str).str.zfill(2) + "-" + dfm.coalition

    # dfs for summary stats
    dfs = pd.read_parquet("dashboards/elections_summary.parquet").fillna(0)

    # dft for table of statistics by seat; need to be joined with lf loser frame
    dft = pd.read_parquet("dashboards/elections_seats_winner.parquet")
    dft = dft[dft.election_name != "By-Election"]
    dft = pd.concat(
        [dft[dft.type == "parlimen"].assign(state="Malaysia"), dft],
        axis=0,
        ignore_index=True,
    )
    dft.date = pd.to_datetime(dft.date).dt.date
    lf = pd.read_parquet("data/consol_ballots.parquet")
    lf = lf[lf.result != "won"]
    lf.loc[lf.result == "won_uncontested", "party"] = "NEMO"
    lf.seat = lf.seat + ", " + lf.state
    lf = lf[["date", "seat", "party"]].drop_duplicates().rename(columns={"party": "party_lost"})
    lf = lf.groupby(["date", "seat"])["party_lost"].agg(list).reset_index()
    dft = pd.merge(dft, lf, on=["date", "seat"], how="left")
    dft["n_candidates"] = dft["party_lost"].apply(lambda x: len(x) + 1)
    dft.loc[dft.voter_turnout == 0, "n_candidates"] = 1

    assert (
        len(dfm.drop_duplicates(subset=col_combo))
        == len(dfs.drop_duplicates(subset=col_combo))
        == len(dft.drop_duplicates(subset=col_combo))
    ), f"Mismatch between 3 components!\
            ballots: {len(dfm.drop_duplicates(subset=col_combo))} \
            summaries: {len(dfs.drop_duplicates(subset=col_combo))} \
            stats: {len(dft.drop_duplicates(subset=col_combo))}"

    dft.date = pd.to_datetime(dft.date).dt.date.astype(str)
    dfm.date = pd.to_datetime(dfm.date).dt.date.astype(str)
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
                            k: ((None if pd.isna(v) else v) if not isinstance(v, list) else v)
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
                with open(
                    f"api/elections/{state}/{election_type}-{election}.json", "w", encoding="utf-8"
                ) as f:
                    j.dump(data, f)


def upload_data(client=None, bucket=None, file_pattern="candidates/*"):
    """Upload data files to S3."""
    files = g(f"api/{file_pattern}.json")
    files_to_upload = sorted([(f, f.replace("api/", "")) for f in files])

    upload_bulk(client, bucket, files_to_upload, max_workers=120)


def make_upload_dates(client=None, bucket=None):
    """Generate and upload dates data file."""
    data = {"data": []}

    df = (
        pd.read_parquet(
            "dashboards/elections_parties.parquet", columns=["state", "election_name", "date"]
        )
        .drop_duplicates()
        .rename(columns={"election_name": "election"})
    )
    df = df.sort_values(by=["state", "election"]).reset_index(drop=True)
    df = pd.concat(
        [df[df.state == "Malaysia"], df[df.state != "Malaysia"]], axis=0, ignore_index=True
    )
    df.date = pd.to_datetime(df.date).dt.date.astype(str)

    res = df.to_dict(orient="records")
    data["data"] = res
    with open("api/dates.json", "w", encoding="utf-8") as f:
        j.dump(data, f)

    print(upload_single(client, bucket, "api/dates.json", "dates.json"))


if __name__ == "__main__":
    START = datetime.now()
    print(f'\nStart: {START.strftime("%Y-%m-%d %H:%M:%S")}')

    CLIENT = get_s3_client()
    BUCKET = os.getenv("S3_BUCKET")

    make_candidates()
    make_seats()
    make_seats_boundaries()
    make_seats_lineage()
    make_parties()
    make_results()
    make_elections()
    for path in [
        "candidates/*",
        "seats/*/*",
        "parties/*/*/*",
        "coalitions/*/*/*",
        "results/*/*",
        "elections/*/*",
    ]:
        upload_data(CLIENT, BUCKET, file_pattern=path)
    make_upload_dates(CLIENT, BUCKET)
    print(f'\nEnd: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"\nDuration: {datetime.now() - START}\n")
