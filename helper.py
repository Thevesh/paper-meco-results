import pandas as pd
import re
import os
import tarfile
import shutil
import boto3
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date,timedelta, datetime
import time

load_dotenv()
TOKEN_API_S3 = (os.getenv('S3_KEY'),os.getenv('S3_SECRET'))


# generate tarball for arxiv submission
def make_arxiv_tarball(FILEPATH=None, DATAVIZ_PATH='dataviz/',TEMP_PATH='temp_archive'):
    if not FILEPATH:
        return "No FILEPATH provided"

    # Create temporary directory for modified files
    temp_dir = os.path.join(FILEPATH, TEMP_PATH)
    os.makedirs(temp_dir, exist_ok=True)

    # Keep track of arcnames for each file to avoid recomputing
    file_to_arcname = {}

    try:
        # 1. Copy .tex and .bbl files
        for root, _, files in os.walk(FILEPATH):
            for file in files:
                if file.endswith(('.tex', '.bbl')):
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path, FILEPATH)
                    dst_path = os.path.join(temp_dir, rel_path)
                    
                    # Create destination directory if it doesn't exist
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    
                    # For .tex files, modify content
                    if file.endswith('.tex'):
                        with open(src_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Replace DATAVIZ_PATH in \includegraphics lines
                        modified_lines = []
                        for line in content.split('\n'):
                            if '\\includegraphics' in line and DATAVIZ_PATH in line:
                                line = line.replace(DATAVIZ_PATH, '')
                            modified_lines.append(line)
                        
                        with open(dst_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(modified_lines))
                    else:
                        shutil.copy2(src_path, dst_path)
                    
                    # Store the arcname for this file (relative to FILEPATH)
                    file_to_arcname[dst_path] = rel_path

        # 2. Copy only .eps and .pdf files from DATAVIZ_PATH to root of temp_dir
        dataviz_full_path = os.path.join(FILEPATH, DATAVIZ_PATH)
        if os.path.exists(dataviz_full_path):
            for root, _, files in os.walk(dataviz_full_path):
                for file in files:
                    if file.endswith(('.eps', '.pdf')):
                        src_path = os.path.join(root, file)
                        # Copy directly to temp_dir root, not preserving directory structure
                        dst_path = os.path.join(temp_dir, os.path.basename(file))
                        shutil.copy2(src_path, dst_path)
                        # Store the arcname for this file (just the basename)
                        file_to_arcname[dst_path] = os.path.basename(file)

        # Create tarball
        tar_path = os.path.join(FILEPATH, 'arxiv.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            for file_path, arcname in file_to_arcname.items():
                if f'{TEMP_PATH}/{TEMP_PATH}' not in file_path:
                    tar.add(file_path, arcname=arcname)

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return tar_path


# upload file to bucket (S3)
def upload_s3(bucket_name=None, source_file_name=None, cloud_file_name=None):
    if not cloud_file_name: cloud_file_name = source_file_name
    try:
        time_start_temp = time.time()
        s3 = boto3.client(
            's3',
            aws_access_key_id=TOKEN_API_S3[0],
            aws_secret_access_key=TOKEN_API_S3[1]
        )
        s3.upload_file(source_file_name, bucket_name, cloud_file_name)
        duration = "{:.1f}".format(time.time() - time_start_temp) + ' seconds'
        res = f'SUCCESS ({duration}): {bucket_name}/{cloud_file_name}'
    except Exception as e:
        res = f'FAILURE: {source_file_name}\n\n{e}'
    
    return res


# upload single file to S3, works as base for bulk upload
def upload_s3_single(bucket_name, source_file_name, cloud_file_name):
    """
    Upload a single file to S3.
    Returns a tuple of (source_file_name, success, message).
    """
    try:
        time_start_temp = time.time()
        s3 = boto3.client(
            's3',
            aws_access_key_id=TOKEN_API_S3[0],
            aws_secret_access_key=TOKEN_API_S3[1]
        )
        s3.upload_file(source_file_name, bucket_name, cloud_file_name)
        duration = "{:.1f}".format(time.time() - time_start_temp) + ' seconds'
        message = f'SUCCESS ({duration}): {bucket_name}/{cloud_file_name}'
        return source_file_name, True, message
    except Exception as e:
        message = f'FAILURE: {bucket_name}/{source_file_name}\n\n{e}'
        return source_file_name, False, message
    

def upload_s3_bulk(bucket_name, files_to_upload, max_workers=50):
    """
    Upload multiple files to S3 in parallel.
    Args:
        bucket_name (str): S3 bucket name.
        files_to_upload (list): List of tuples (source_file_name, cloud_file_name).
        max_workers (int): Number of concurrent uploads.
    Returns:
        dict: Mapping of source file names to (success, message).
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(upload_s3_single, bucket_name, source_file, cloud_file): (source_file, cloud_file)
            for source_file, cloud_file in files_to_upload
        }

        for future in as_completed(future_to_file):
            source_file, cloud_file = future_to_file[future]
            source_file_name, success, message = future.result()
            results[source_file_name] = (success, message)
            print(message)

    failed_uploads = [(source_file, message.split(": ", 1)[1][9:]) for source_file, (success, message) in results.items() if not success]
    return failed_uploads


# get latest view of bucket (S3)
def read_s3(BUCKET=None):
    BASE_URL = f'https://{BUCKET}.s3.ap-southeast-1.amazonaws.com/'

    df = pd.read_xml(BASE_URL)
    df = df.dropna(subset=['Key'])[['Key','LastModified']]
    df.columns = ['key','modified']
    df.modified = pd.to_datetime(df.modified).astype(str).str[:19]
    df.modified = pd.to_datetime(df.modified) + timedelta(hours=8)
    df['URL'] = BASE_URL + df.key
    df = df.sort_values(by='modified',ascending=False).reset_index(drop=True)

    return df


# write csv and parquet simultaneously
def write_csv_parquet(FILEPATH, df=None):
    df.to_csv(f'{FILEPATH}.csv', index=False)
    df.to_parquet(f'{FILEPATH}.parquet', index=False, compression='brotli')
    print(f'Wrote CSV + Parquet: {FILEPATH}')


# write parquet with customisation
def write_parquet(FILEPATH, df=None):
    df.to_parquet(f'{FILEPATH}.parquet', index=False, compression='brotli')
    print(f'Wrote Parquet: {FILEPATH}')
  

# write CSV with customisation
def write_csv(FILEPATH, df=None):
    df.to_csv(f'{FILEPATH}.csv', index=False)
    print(f'Wrote CSV: {FILEPATH}')
    

# slug generator
def generate_slug(x):
    slug = re.sub(r'[^a-zA-Z0-9\s]', '', x)
    slug = slug.replace(' ', '-').lower()
    return slug


# get list of states
def get_states(my=0):
    stt = [
        'Perlis', 'Kedah', 'Kelantan', 'Terengganu', 
        'Pulau Pinang', 'Perak', 'Pahang', 'Selangor', 
        'W.P. Kuala Lumpur', 'W.P. Putrajaya', 
        'Negeri Sembilan', 'Melaka', 'Johor', 
        'W.P. Labuan', 'Sabah', 'Sarawak'
    ] # election arrangement
    if my == 1:
        stt = ['Malaysia'] + stt

    return stt


# capitalise every word in sentence
def capitalize_sentence(sentence):
    words = sentence.split()
    return ' '.join([words[0].upper()] + [word.title() for word in words[1:]])


