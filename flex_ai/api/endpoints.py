from typing import List, Literal
import requests
from flex_ai.common.classes import LoraCheckpoint
from flex_ai.settings import BASE_URL

def get_endpoint(api_key:str, id:str):
    url = f"{BASE_URL}/v1/endpoints"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"id": id}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()
    return data[0]

def create_multi_lora_endpoint(api_key:str, name:str, lora_checkpoints: List[LoraCheckpoint], compute: Literal["T4", "A100-40GB", "A100-80GB", "A10G", "A100-80GB", "L4"] = "A100-40GB"):
    url = f"{BASE_URL}/v1/endpoints/create_multi_lora_endpoint"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"name": name, "lora_checkpoints": lora_checkpoints, "compute": compute}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()
    return data