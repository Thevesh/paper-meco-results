"""
Generate lookup_dun_parlimen.csv: the parliamentary constituency that contains each state
constituency, for every state election, as delineated in MECo-2 (paper-meco-maps).

compile.py uses it to check that, where a state election was held with a general election,
the DUN electorates within each parliamentary seat sum to that seat's electorate.

Steps:
    1. Read every {STATE}_SE-xx layer from the MECo-2 geoparquet folder
    2. Keep state, election, code_parlimen, code_dun
    3. Save to data/lookup_dun_parlimen.csv

Dependencies:
    - paper-meco-maps checked out next to this repo (or MECO2 set in the .env file)
"""

import glob
import os
import re

import duckdb
import pandas as pd
from dotenv import load_dotenv

from helper import get_states

load_dotenv()

MECO2 = os.getenv("MECO2", "../paper-meco-maps/data/geoparquet/elections")


def main():
    """Snapshot the DUN -> parliament mapping of every state election layer."""
    state_of = dict(zip(get_states(codes=1), get_states()))
    frames = []
    for path in sorted(glob.glob(f"{MECO2}/*_SE-*.parquet")):
        code, election = re.match(r"([A-Z]{3})_(SE-\d\d)\.parquet", os.path.basename(path)).groups()
        df = duckdb.sql(f"SELECT DISTINCT code_parlimen, code_dun FROM '{path}'").df()
        frames.append(df.assign(state=state_of[code], election=election))
    df = pd.concat(frames)[["state", "election", "code_parlimen", "code_dun"]]
    df = df.sort_values(["state", "election", "code_dun"]).reset_index(drop=True)
    assert not df.duplicated(["state", "election", "code_dun"]).any(), "DUN in two parliaments!"
    df.to_csv("data/lookup_dun_parlimen.csv", index=False)
    print(f"{len(df):,} DUNs across {df.groupby(['state', 'election']).ngroups} state elections")


if __name__ == "__main__":
    main()
