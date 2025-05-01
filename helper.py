import pandas as pd
import re

import boto3
from datetime import date,timedelta, datetime
import time

TOKEN_API_S3 = ('YOUR_ACCESS_KEY', 'YOUR_SECRET_KEY')


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