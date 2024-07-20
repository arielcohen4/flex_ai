import requests

from flex_ai.common.enums import DatasetType
BASE_URL = "https://api.getflex.ai"
# BASE_URL = "http://localhost:8080"
# send api key in the header
def generate_dataset_upload_urls(api_key:str, dataset_id:str):
    url = f"{BASE_URL}/v1/datasets/generate_upload_urls"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"id": dataset_id}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    data = response.json()
    return data["train_upload_url"], data["eval_upload_url"]

def create_dataset(api_key:str, dataset_id: str, name:str, train_rows_count:int, eval_rows_count:int, max_tokens:int, total_tokens:int, type: DatasetType):
    url = f"{BASE_URL}/v1/datasets/create_dataset"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"id": dataset_id, "name": name, "train_rows_count": train_rows_count, "eval_rows_count": eval_rows_count, "max_tokens": max_tokens, "total_tokens": total_tokens, "type": type.value}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()
    return data[0]