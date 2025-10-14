import json
import requests

def upload_to_jsonbin(json_data, api_key, bin_id=None):
    """
    upload JSON to JSONBin.io
    Parameters
    ----------
    json_data: dict
        data to upload
    api_key: str
        JSONBin Master Key
    bin_id: str (optional)
    Returns
    -------
    response: dict
    """
    headers = {
        "Content-Type": "application/json",
        "X-Master-Key": api_key
    }
    
    if bin_id:
        url = f"https://api.jsonbin.io/v3/b/{bin_id}"
        response = requests.put(url, headers=headers, json=json_data)
    else:
        url = "https://api.jsonbin.io/v3/b"
        response = requests.post(url, headers=headers, json=json_data)

    if response.status_code not in (200, 201):
        raise Exception(f"Upload failed: {response.text}")
    return response.json()
