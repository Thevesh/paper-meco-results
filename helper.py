"""Helper module for file operations, data processing, and S3 interactions."""

import os
import re
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import boto3
from dotenv import load_dotenv

load_dotenv()

# S3 configuration
s3_bucket = os.getenv("S3_BUCKET")
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

TOKEN_API_S3 = (os.getenv("S3_KEY"), os.getenv("S3_SECRET"))


def make_arxiv_tarball(filepath=None, dataviz_path="dataviz/", temp_path="temp_archive"):
    """Generate a tarball for arXiv submission.

    Args:
        filepath (str): Base path for the files
        dataviz_path (str): Path to data visualization files
        temp_path (str): Path for temporary files

    Returns:
        str: Path to the generated tarball
    """
    if not filepath:
        return "No filepath provided"

    # Create temporary directory for modified files
    temp_dir = os.path.join(filepath, temp_path)
    os.makedirs(temp_dir, exist_ok=True)

    # Keep track of arcnames for each file to avoid recomputing
    file_to_arcname = {}

    try:
        # 1. Copy .tex and .bbl files
        for root, _, files in os.walk(filepath):
            for file in files:
                if file.endswith((".tex", ".bbl")):
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path, filepath)
                    dst_path = os.path.join(temp_dir, rel_path)

                    # Create destination directory if it doesn't exist
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                    # For .tex files, modify content
                    if file.endswith(".tex"):
                        with open(src_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Replace dataviz_path in \includegraphics lines
                        modified_lines = []
                        for line in content.split("\n"):
                            if "\\includegraphics" in line and dataviz_path in line:
                                line = line.replace(dataviz_path, "")
                            modified_lines.append(line)

                        with open(dst_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(modified_lines))
                    else:
                        shutil.copy2(src_path, dst_path)

                    # Store the arcname for this file (relative to filepath)
                    file_to_arcname[dst_path] = rel_path

        # 2. Copy only .eps and .pdf files from dataviz_path to root of temp_dir
        dataviz_full_path = os.path.join(filepath, dataviz_path)
        if os.path.exists(dataviz_full_path):
            for root, _, files in os.walk(dataviz_full_path):
                for file in files:
                    if file.endswith((".eps", ".pdf")):
                        src_path = os.path.join(root, file)
                        # Copy directly to temp_dir root, not preserving directory structure
                        dst_path = os.path.join(temp_dir, os.path.basename(file))
                        shutil.copy2(src_path, dst_path)
                        # Store the arcname for this file (just the basename)
                        file_to_arcname[dst_path] = os.path.basename(file)

        # Create tarball
        tar_path = os.path.join(filepath, "arxiv.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for file_path, arcname in file_to_arcname.items():
                if f"{temp_path}/{temp_path}" not in file_path:
                    tar.add(file_path, arcname=arcname)

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return tar_path


def upload_s3(bucket_name=None, source_file_name=None, cloud_file_name=None):
    """Upload a file to S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket
        source_file_name (str): Path to the source file
        cloud_file_name (str): Name to use in the cloud

    Returns:
        str: Status message indicating success or failure
    """
    if not cloud_file_name:
        cloud_file_name = source_file_name
    try:
        time_start = time.time()
        s3 = boto3.client(
            "s3",
            aws_access_key_id=TOKEN_API_S3[0],
            aws_secret_access_key=TOKEN_API_S3[1],
        )
        s3.upload_file(source_file_name, bucket_name, cloud_file_name)
        duration = f"{time.time() - time_start:.1f} seconds"
        return f"SUCCESS ({duration}): {bucket_name}/{cloud_file_name}"
    except boto3.exceptions.S3UploadFailedError as e:
        return f"FAILURE: {source_file_name}\n\n{e}"


def upload_s3_single(bucket_name, source_file_name, cloud_file_name):
    """Upload a single file to S3.

    Args:
        bucket_name (str): Name of the S3 bucket
        source_file_name (str): Path to the source file
        cloud_file_name (str): Name to use in the cloud

    Returns:
        tuple: (source_file_name, success, message)
    """
    try:
        time_start = time.time()
        s3 = boto3.client(
            "s3",
            aws_access_key_id=TOKEN_API_S3[0],
            aws_secret_access_key=TOKEN_API_S3[1],
        )
        s3.upload_file(source_file_name, bucket_name, cloud_file_name)
        duration = f"{time.time() - time_start:.1f} seconds"
        message = f"SUCCESS ({duration}): {bucket_name}/{cloud_file_name}"
        return source_file_name, True, message
    except boto3.exceptions.S3UploadFailedError as e:
        message = f"FAILURE: {bucket_name}/{source_file_name}\n\n{e}"
        return source_file_name, False, message


def upload_s3_bulk(bucket_name, files_to_upload, max_workers=50):
    """Upload multiple files to S3 in parallel.

    Args:
        bucket_name (str): S3 bucket name
        files_to_upload (list): List of tuples (source_file_name, cloud_file_name)
        max_workers (int): Number of concurrent uploads

    Returns:
        list: List of tuples containing failed uploads (source_file, error_message)
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(upload_s3_single, bucket_name, source_file, cloud_file): (
                source_file,
                cloud_file,
            )
            for source_file, cloud_file in files_to_upload
        }

        for future in as_completed(future_to_file):
            source_file, cloud_file = future_to_file[future]
            source_file_name, success, message = future.result()
            results[source_file_name] = (success, message)
            print(message)

    failed_uploads = [
        (source_file, message.split(": ", 1)[1][9:])
        for source_file, (success, message) in results.items()
        if not success
    ]
    return failed_uploads


def write_csv_parquet(filepath, df=None):
    """Write dataframe to both CSV and Parquet formats.

    Args:
        filepath (str): Base path for the output files
        df (pd.DataFrame): DataFrame to write
    """
    df.to_csv(f"{filepath}.csv", index=False)
    df.to_parquet(f"{filepath}.parquet", index=False, compression="brotli")
    print(f"Wrote CSV + Parquet: {filepath}")


def write_parquet(filepath, df=None):
    """Write dataframe to Parquet format.

    Args:
        filepath (str): Base path for the output file
        df (pd.DataFrame): DataFrame to write
    """
    df.to_parquet(f"{filepath}.parquet", index=False, compression="brotli")
    print(f"Wrote Parquet: {filepath}")


def write_csv(filepath, df=None):
    """Write dataframe to CSV format.

    Args:
        filepath (str): Base path for the output file
        df (pd.DataFrame): DataFrame to write
    """
    df.to_csv(f"{filepath}.csv", index=False)
    print(f"Wrote CSV: {filepath}")


def generate_slug(x):
    """Generate URL-friendly slug from string.

    Args:
        x (str): Input string

    Returns:
        str: URL-friendly slug
    """
    slug = re.sub(r"[^a-zA-Z0-9\s]", "", x)
    slug = slug.replace(" ", "-").lower()
    return slug


def get_states(my: int = 0, codes: int = 0) -> List[str]:
    """Get list of Malaysian states.

    Args:
        my (int): Whether to include Malaysia (country)
        code (int): Whether to return full name (0), text code (1), or integer code (2)
    Returns:
        List[str]: List of states as full name, text codes, or integer codes
    """
    data = {
        "Malaysia": ["MYS", 0],
        "Perlis": ["PLS", 1],
        "Kedah": ["KDH", 2],
        "Kelantan": ["KTN", 3],
        "Terengganu": ["TRG", 4],
        "Pulau Pinang": ["PNG", 5],
        "Perak": ["PRK", 6],
        "Pahang": ["PHG", 7],
        "Selangor": ["SGR", 8],
        "W.P. Kuala Lumpur": ["KUL", 9],
        "W.P. Putrajaya": ["PJY", 10],
        "Negeri Sembilan": ["NSN", 11],
        "Melaka": ["MLK", 12],
        "Johor": ["JHR", 13],
        "W.P. Labuan": ["LBN", 14],
        "Sabah": ["SBH", 15],
        "Sarawak": ["SWK", 16],
    }
    state_names = list(data.keys())
    state_codes = [data[x][0] for x in state_names]
    state_order = [data[x][1] for x in state_names]
    if codes == 0:
        return state_names[1 - my :]
    if codes == 1:
        return state_codes[1 - my :]
    if codes == 2:
        return state_order[1 - my :]
    raise ValueError("Invalid code type")


def capitalize_sentence(sentence):
    """Capitalize first word and title case remaining words in sentence.

    Args:
        sentence (str): Input sentence

    Returns:
        str: Properly capitalized sentence
    """
    words = sentence.split()
    return " ".join([words[0].upper()] + [word.title() for word in words[1:]])


def compute_summary(
    df,
    election=None,
    exclude_state=None,
    include_state=None,
    col_groupby=["coalition", "party"],
    col_seat="seat",
):
    """
    Compute summary statistics for a given election and (optionally) selected/excluded states.

    Args:
        df (pandas.DataFrame): DataFrame containing election data. Must have columns: 'election', 'state', 'votes', and all in col_groupby.
        election (str, optional): Election identifier string. If provided, filters rows to only this election.
        include_state (list of str, optional): List of state names to include. If provided, retains only these states.
        exclude_state (list of str, optional): List of state names to exclude. Ignored if null OR if include_state is specified.
        col_groupby (list of str, optional): Columns to group by. Default is ['coalition', 'party'].

    Returns:
        pandas.DataFrame: DataFrame with columns: col_groupby, 'votes' (comma formatted), and 'votes_perc' (rounded % of total votes per group).
    """
    if election:
        df = df[df.election == election]
    if include_state:
        df = df[df.state.isin(include_state)]
    elif exclude_state:
        df = df[~df.state.isin(exclude_state)]
    df["votes_perc"] = df["votes"] / df["votes"].sum() * 100
    df["seats"] = 0

    wf = df.sort_values(by="votes", ascending=True).drop_duplicates(subset=col_seat, keep="last")
    df.loc[df.index.isin(wf.index), "seats"] = 1

    rf = (
        df[col_groupby + ["votes", "votes_perc", "seats"]]
        .groupby(col_groupby)
        .sum()
        .sort_values(by=[col_groupby[0], "votes"], ascending=False)
    )
    rf.votes_perc = rf.votes_perc.round(1)
    rf["votes"] = rf["votes"].apply(lambda x: f"{x:,}")
    return rf


def get_final_cols(file_type="ballots"):
    """Get final columns for a given file type. Extra layer of validation that all cols are present.

    Args:
        file_type (str): Type of file ("ballots" or "stats")

    Returns:
        list: List of columns
    """
    if file_type == "ballots":
        return [
            "date",
            "election",
            "state",
            "seat",
            "ballot_order",
            "candidate_uid",
            "name_on_ballot",
            "name",
            "sex",
            "ethnicity",
            "age",
            "party_on_ballot",
            "party_uid",
            "party",
            "coalition_uid",
            "coalition",
            "votes",
            "votes_perc",
            "rank",
            "result",
        ]
    elif file_type == "stats":
        return [
            "date",
            "election",
            "state",
            "seat",
            "voters_total",
            "ballots_issued",
            "ballots_not_returned",
            "votes_rejected",
            "votes_valid",
            "majority",
            "n_candidates",
            "voter_turnout",
            "majority_perc",
            "votes_rejected_perc",
            "ballots_not_returned_perc",
        ]
