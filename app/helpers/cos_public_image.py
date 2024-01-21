import os
import re
import ibm_boto3
from ibm_botocore.client import Config, ClientError

cred_cos = {
        "COS_APIKEY": os.environ["COS_APIKEY"],
        "COS_INSTANCE_CRN": os.environ["COS_INSTANCE_CRN"],
        "endpoint_url_private": os.environ["endpoint_url_private"],
        "endpoint_url_public": os.environ["endpoint_url_public"],
        "COS_APIKEY_PUB": os.environ["COS_APIKEY_PUB"],
        "COS_INSTANCE_CRN_PUB": os.environ["COS_INSTANCE_CRN_PUB"],
    }

PUBLIC_BUCKET = "temp-img-hosting"
PRIVATE_BUCKET = "cos-parsing-bni-onboard-1"


def cos_init(bucket_type, creds=cred_cos):
    COS_ENDPOINT = "https://"+creds['endpoint_url_public']

    if bucket_type == "private":
        apikey = creds['COS_APIKEY']
        instance = creds['COS_INSTANCE_CRN']
    else:
        apikey = creds['COS_APIKEY_PUB']
        instance = creds['COS_INSTANCE_CRN_PUB']

    res = ibm_boto3.resource("s3",
    ibm_api_key_id=apikey,
    ibm_service_instance_id=instance,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT)

    return res, COS_ENDPOINT

def upload_file_obj(bucket_name, item_name, bucket_type):
    res, COS_ENDPOINT = cos_init(bucket_type)
    print("Starting upload item to bucket: {0}, key: {1}".format(bucket_name, item_name))
    with open('filename.png', 'rb') as item_bin:
        try:
            #res.Bucket(bucket_name).upload_file(item_name, item_name)
            res.Bucket(bucket_name).Object(item_name).upload_fileobj(item_bin)
            print("uploaded file: ", item_name)
            return f"{COS_ENDPOINT}/{bucket_name}/{item_name}"
        except Exception as e:
            print("Unable to retrieve file contents: {0}".format(e))


def download_file_obj(bucket_name, item_name, bucket_type):
    res, COS_ENDPOINT = cos_init(bucket_type)
    obj = res.Bucket(bucket_name).Object(item_name)
    
    #print(type(obj))
    with open('filename.png', 'wb') as data:
        obj.download_fileobj(data)

    

async def replace_urls(passage: str, PRIVATE_BUCKET=PRIVATE_BUCKET, PUBLIC_BUCKET=PUBLIC_BUCKET) -> str:
    url_pattern = re.compile(r'https://cos-parsing-bni-onboard-1.s3.jp-tok.\S+\.png')
    url_found = re.findall(url_pattern, passage)

    if len(url_found) == 0:
        return passage
    else:
        for url in url_found:
            item_name = "/".join(url.split("/")[-2:])
            print(item_name)
            download_file_obj(PRIVATE_BUCKET, item_name, "private")
            pub_url = upload_file_obj(PUBLIC_BUCKET, item_name, "public")
            replacement = f"<img src='{pub_url}' style='max-width: 300px; height: auto;'>"
            passage = re.sub(url, replacement, passage)

    return passage
