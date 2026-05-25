"""Upload API data to R2."""

import os
import json as j
import pandas as pd
from datetime import datetime
from glob import glob as g

from dotenv import load_dotenv

from helper import get_r2_client, upload_bulk

load_dotenv()


def make_candidates():
    """Generate candidate data files for API."""
    data = {"data": []}

    df = pd.read_parquet("dashboards/elections_candidates.parquet")
    print(f"Handling {len(df.slug.unique()):,.0f} unique candidates")
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
        print("Wrote candidates/dropdown.json")

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

    df = df[col_api_candidate + ["slug"]].sort_values(by="date", ascending=False)
    df = df.astype(object).where(df.notna(), other=None)  # proper JSON null

    all_data = {
        slug: group.drop(columns="slug").to_dict(orient="records")
        for slug, group in df.groupby("slug", sort=True)
    }

    with open("api/candidates/all.json", "w", encoding="utf-8") as f:
        j.dump(all_data, f)
        print("Wrote candidates/all.json")


def upload_data(client, bucket, file_pattern="candidates/*"):
    """Upload data files matching pattern to R2."""
    files = g(f"api/{file_pattern}.json")
    files_to_upload = sorted([(f, f.replace("api/", "v1/")) for f in files])
    upload_bulk(client, bucket, files_to_upload, max_workers=120)


if __name__ == "__main__":
    START = datetime.now()
    print(f'\nStart: {START.strftime("%Y-%m-%d %H:%M:%S")}\n')

    CLIENT = get_r2_client()
    BUCKET = os.getenv("R2_BUCKET")

    make_candidates()

    # for path in [
    #     "candidates/*",
    #     "seats/*/*",
    #     "parties/*/*/*",
    #     "coalitions/*/*/*",
    #     "results/*/*",
    #     "elections/*/*",
    # ]:
    #     upload_data(CLIENT, BUCKET, file_pattern=path)

    print(f'\nEnd: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"\nDuration: {datetime.now() - START}\n")
