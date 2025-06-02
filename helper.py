"""Helper module for file operations, data processing, and S3 interactions."""

import json
import os
import re
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, datetime
from typing import List, Dict, Any

import boto3
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# S3 configuration
s3_bucket = os.getenv("S3_BUCKET")
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

TOKEN_API_S3 = (os.getenv("S3_KEY"), os.getenv("S3_SECRET"))


def make_arxiv_tarball(
    filepath=None, dataviz_path="dataviz/", temp_path="temp_archive"
):
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


def read_s3(bucket=None):
    """Get latest view of S3 bucket contents.
    
    Args:
        bucket (str): Name of the S3 bucket
        
    Returns:
        pd.DataFrame: DataFrame containing bucket contents
    """
    base_url = f"https://{bucket}.s3.ap-southeast-1.amazonaws.com/"

    df = pd.read_xml(base_url)
    df = df.dropna(subset=["Key"])[["Key", "LastModified"]]
    df.columns = ["key", "modified"]
    df.modified = pd.to_datetime(df.modified).astype(str).str[:19]
    df.modified = pd.to_datetime(df.modified) + timedelta(hours=8)
    df["url"] = base_url + df.key
    df = df.sort_values(by="modified", ascending=False).reset_index(drop=True)

    return df


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


def get_states(my: int = 0) -> List[str]:
    """Get list of Malaysian states.
    
    Args:
        my (int): Whether to include only Malaysian states
        
    Returns:
        List[str]: List of state names
    """
    states = [
        "Johor",
        "Kedah",
        "Kelantan",
        "Melaka",
        "Negeri Sembilan",
        "Pahang",
        "Perak",
        "Perlis",
        "Pulau Pinang",
        "Sabah",
        "Sarawak",
        "Selangor",
        "Terengganu",
        "W.P. Kuala Lumpur",
        "W.P. Labuan",
        "W.P. Putrajaya",
    ]
    return states if my == 0 else ['Malaysia'] + states


def capitalize_sentence(sentence):
    """Capitalize first word and title case remaining words in sentence.
    
    Args:
        sentence (str): Input sentence
        
    Returns:
        str: Properly capitalized sentence
    """
    words = sentence.split()
    return " ".join([words[0].upper()] + [word.title() for word in words[1:]])


def upload_to_s3(file_path: str, s3_key: str) -> None:
    """Upload file to S3.
    
    Args:
        file_path (str): Path to file to upload
        s3_key (str): S3 key to upload to
    """
    try:
        s3_client.upload_file(file_path, s3_bucket, s3_key)
    except Exception as e:
        print(f"Failed to upload {file_path} to {s3_key}: {str(e)}")


def download_from_s3(s3_key: str, file_path: str) -> None:
    """Download file from S3.
    
    Args:
        s3_key (str): S3 key to download from
        file_path (str): Path to save file to
    """
    try:
        s3_client.download_file(s3_bucket, s3_key, file_path)
    except Exception as e:
        print(f"Failed to download {s3_key} to {file_path}: {str(e)}")


def list_s3_files(prefix: str) -> List[str]:
    """List files in S3 bucket with given prefix.
    
    Args:
        prefix (str): Prefix to filter files by
        
    Returns:
        List[str]: List of S3 keys
    """
    try:
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]
    except Exception as e:
        print(f"Failed to list files with prefix {prefix}: {str(e)}")
        return []


def read_json(file_path: str) -> Dict[str, Any]:
    """Read JSON file.
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        Dict[str, Any]: JSON data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], file_path: str) -> None:
    """Write data to JSON file.
    
    Args:
        data (Dict[str, Any]): Data to write
        file_path (str): Path to write to
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_timestamp() -> str:
    """Get current timestamp in ISO format.
    
    Returns:
        str: Current timestamp
    """
    return datetime.now().isoformat()
