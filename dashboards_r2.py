"""Upload API data to R2."""

import os
from datetime import datetime
from glob import glob as g

from dotenv import load_dotenv

from helper import get_r2_client, upload_bulk

load_dotenv()


def upload_data(client, bucket, file_pattern="candidates/*"):
    """Upload data files matching pattern to R2."""
    files = g(f"api/{file_pattern}.json")
    files_to_upload = sorted([(f, f.replace("api/", "")) for f in files])
    upload_bulk(client, bucket, files_to_upload, max_workers=120)


if __name__ == "__main__":
    START = datetime.now()
    print(f'\nStart: {START.strftime("%Y-%m-%d %H:%M:%S")}')

    CLIENT = get_r2_client()
    BUCKET = os.getenv("R2_BUCKET")

    for path in [
        "candidates/*",
        "seats/*/*",
        "parties/*/*/*",
        "coalitions/*/*/*",
        "results/*/*",
        "elections/*/*",
    ]:
        upload_data(CLIENT, BUCKET, file_pattern=path)

    print(f'\nEnd: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"\nDuration: {datetime.now() - START}\n")
