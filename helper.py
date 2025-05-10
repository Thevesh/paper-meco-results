import pandas as pd
import re

import boto3
from datetime import date,timedelta, datetime
import time

TOKEN_API_S3 = ('YOUR_ACCESS_KEY', 'YOUR_SECRET_KEY')


# generate tarball for arxiv submission
def make_arxiv_tarball(FILEPATH=None,DATAVIZ_PATH='dataviz/'):
    import os
    import tarfile
    import shutil
    from pathlib import Path

    if not FILEPATH:
        return "No FILEPATH provided"

    # Create temporary directory for modified files
    temp_dir = os.path.join(FILEPATH, 'temp_archive')
    os.makedirs(temp_dir, exist_ok=True)

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

        # Create tarball
        tar_path = os.path.join(FILEPATH, 'arxiv.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
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


