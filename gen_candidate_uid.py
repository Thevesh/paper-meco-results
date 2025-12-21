"""
Generate 5-character candidate UID based on the integer candidate running number.
Total permutations: 33,554,432
Expected to last at least until year 14000, based on current rate of growth.

Steps:
    1. Read lookup_candidate.csv
    2. Drop candidate_uid if it exists
    3. Insert candidate_uid column
    4. Re-save to lookup_candidate.csv

Dependencies:
    - UID_SECRET in .env file
    - lookup_candidate.csv in src-data folder
"""

import os
import hashlib
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
BASE = 32
WIDTH = 5
MOD = BASE**WIDTH
UID_SECRET = os.environ["UID_SECRET"]


def _derive_params(secret: str) -> tuple[int, int]:
    """Derive hash-based parameters (a, b) from the secret string."""
    h = hashlib.sha256(secret.encode()).digest()
    a = int.from_bytes(h[:4], "big") % MOD | 1  # force odd
    b = int.from_bytes(h[4:8], "big") % MOD
    return a, b


def _to_crockford_base32(n: int) -> str:
    """Convert an integer to a 5-character Crockford base32 string."""
    return "".join(CROCKFORD32[(n >> 5 * i) & 31] for i in reversed(range(WIDTH)))


def candidate_uid(rn: int, secret: str) -> str:
    """Generate a 5-character candidate UID from a running number and secret."""
    if rn <= 0:
        raise ValueError("rn must be a positive integer")
    a, b = _derive_params(secret)
    return _to_crockford_base32((a * rn + b) % MOD)


if __name__ == "__main__":
    print("\n --------- Generating candidate UIDs ----------\n")
    df = pd.read_csv("src-data/lookup_candidate.csv")
    print(f"Loaded {len(df):,.0f} candidates from lookup_candidate.csv.")

    if "candidate_uid" in df.columns:
        print("Removing existing candidate_uid column")
        df = df.drop("candidate_uid", axis=1)

    print("Generating new candidate_uid values")
    df.insert(0, "candidate_uid", df.candidate_rn.apply(lambda x: candidate_uid(x, UID_SECRET)))
    df.to_csv("src-data/lookup_candidate.csv", index=False)
    print("Saved updated lookup_candidate.csv")

    print("\n\n --------- ✨ DONE ✨ ----------\n")
