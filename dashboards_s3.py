"""Module for generating and uploading dashboard data to S3."""

import json as j
import os
from glob import glob as g
from ast import literal_eval as le
from datetime import datetime
import pandas as pd

from helper import get_states, upload_s3, upload_s3_bulk, generate_slug


def make_candidates():
    """Generate candidate data files for API."""
    data = {"data": []}

    df = pd.read_parquet("src-data/dashboards/elections_candidates.parquet")
    df = (df
        .assign(
            c=1,
            w=df.result.str.contains('won').astype(int),
            l=lambda x: 1 - x.w
        )
        .groupby(['slug','name'], as_index=False)
        .agg({'c':'sum', 'w':'sum', 'l':'sum'})
        .sort_values(['c','w'], ascending=False)
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
    data = {"desc_en": "", "desc_ms": "", "voters_total": 0, "data": [], "barmeter": {}, "pyramid": {"ages":[],"male":[],"female":[]}}

    col_api_seat = [
        "election_name",
        "seat",
        "date",
        "party",
        "name",
        "majority",
        "majority_perc",
    ]

    df = pd.read_csv('src-data/lookup_seats_new.csv')
    for r in [('federal','parlimen'),('state','dun')]:
        df.type = df.type.replace(r[0],r[1])
    for r in [('[','["'),(']','"]'),(',','","')]:
        df.lineage = df.lineage.str.replace(r[0],r[1])
    df['lineage'] = df['lineage'].apply(le)
    df['lineage_sort'] = df.lineage.astype(str)
    df.seat = df.seat + ', ' + df.state
    df['slug'] = df.seat.apply(generate_slug)

    tf = df[df['lineage'].apply(lambda x: len(x) == 1)].copy()\
        .sort_values(by=['lineage_sort','date'])\
        .drop_duplicates(subset=['lineage'],keep='last')\
        .drop(columns=['lineage_sort'])
    lineages = tf.lineage.apply(lambda x: x[0])
    tf = pd.concat([
        tf[tf.type == 'parlimen'],
        tf[tf.type == 'dun']
    ],axis=0,ignore_index=True)
    tf = tf[['seat','slug','type']].rename(columns={'seat':'seat_name'}).to_dict(orient='records')
    data['data'] = tf
    with open('api/seats/dropdown.json','w',encoding='utf-8') as f:
        j.dump(data,f)

    map_change = {
        'en': {
            'rename': ' was renamed to ',
            'split': ' was split into ',
            'merge': ' was merged with ',
        },
        'ms': {
            'rename': ' dinamakan semula kepada ',
            'split': ' dibahagikan kepada ',
            'merge': ' digabungkan dengan ',
        }
    }
    lf = pd.read_csv('src-data/lookup_seats_lineage_desc.csv',dtype=str).dropna(how='any')
    lf['change_en'] = lf['from'] + lf['type'].map(map_change['en']) + lf['to'] + ' in the ' + lf['date'].str[:4] + ' redelineation'
    lf['change_ms'] = lf['from'] + lf['type'].map(map_change['ms']) + lf['to'].str.replace(' and ',' dan ') + ' dalam persempadan semula ' + lf['date'].str[:4]
    # for lang in ['en','ms']:
    #     ef.loc[ef.type == 'carve_out',f'change_{lang}'] = map_change[lang]['carve_out_prefix'] + ef[f'change_{lang}']

    sf = pd.read_parquet('src-data/dashboards/elections_seats_winner.parquet')
    sf.slug = sf.seat.apply(generate_slug)

    bf = pd.read_csv('src-data/scrape/voters_ge15_demog.csv')
    bf = pd.concat([
        bf,
        bf.groupby(['state','parlimen']).sum(numeric_only=True).reset_index()
    ],axis=0,ignore_index=True)
    # for c in bf.columns[4:]: bf[c] = (bf[c]/bf['total'] * 100).round(1)
    bf['slug'] = bf.parlimen + ', ' + bf.state
    bf.loc[~bf.dun.isnull(),'slug'] = bf.dun + ', ' + bf.state
    bf.slug = bf.slug.apply(generate_slug)
    bf['desc_en'] = bf.parlimen + ' is a federal constituency in ' + bf.state + ', with ' + bf.total.astype(int).map('{:,}'.format) + ' voters as of GE-15 (2022).'
    bf['desc_ms'] = bf.parlimen + ' adalah sebuah kawasan persekutuan di ' + bf.state + ', dengan ' + bf.total.astype(int).map('{:,}'.format) + ' pengundi setakat GE-15 (2022).'
    bf.loc[~bf.dun.isnull(),'desc_en'] = bf.dun + ' is a state constituency in ' + bf.state + ', with ' + bf.total.astype(int).map('{:,}'.format) + ' voters as of GE-15 (2022).'
    bf.loc[~bf.dun.isnull(),'desc_ms'] = bf.dun + ' adalah sebuah kawasan negeri di ' + bf.state + ', dengan ' + bf.total.astype(int).map('{:,}'.format) + ' pengundi setakat GE-15 (2022).'

    af = pd.read_parquet('src-data/scrape/voters_ge15_pyramid.parquet')

    for lineage in lineages:
        slugs = list(df[df['lineage'].apply(lambda x, l=lineage: l in x)].sort_values(by='date')['slug'])
        if len(slugs) == 0:
            continue
        
        # Election results
        tf = (
            sf[sf.slug.isin(slugs)]
            .copy()[col_api_seat]
            .sort_values(by="date", ascending=False)
        )
        tf = tf.to_dict(orient="records")
        tf = [
            {k: (None if pd.isna(v) else v) for k, v in record.items()} for record in tf
        ]  # proper JSON null

        # Lineage
        tfl = lf[lf.slug == slugs[-1]][['date','change_en','change_ms']].to_dict(orient='records')

        # Demographics
        tfd = bf[bf.slug == slugs[-1]].iloc[0]
        barmeter = {"votertype": {}, "sex": {}, "age": {}, "ethnic": {}}
        for k in barmeter.keys():
            prefix = k + '_'
            barmeter[k] = {}
            for col in bf.columns:
                if col.startswith(prefix):
                    barmeter[k][col[len(prefix):]] = int(tfd[col])

        # Pyramid
        tfa = af[af.slug == slugs[-1]].copy()
        if len(tfa) == 0:
            print(f'No pyramid data for {slugs[-1]}')
        data['pyramid']['ages'] = [x for x in range(18,101)]
        data['pyramid']['male'] = [int(x) for x in tfa.male.tolist()]
        data['pyramid']['female'] = [int(x) for x in tfa.female.tolist()]

        # Combine into JSON response
        data["desc_en"] = tfd['desc_en']
        data["desc_ms"] = tfd['desc_ms']
        data["voters_total"] = int(tfd['total'])
        data["data"] = tf + tfl
        data["data"].sort(key=lambda x: x.get("date", ""), reverse=True)
        data["barmeter"] = barmeter

        with open(f"api/seats/{slugs[-1]}.json", "w", encoding="utf-8") as f:
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
