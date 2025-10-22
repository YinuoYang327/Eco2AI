import json
import requests
import warnings

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
    res_json = response.json()
    if "id" in res_json:
        bin_id = res_json["id"]
    elif "metadata" in res_json and "id" in res_json["metadata"]:
        bin_id = res_json["metadata"]["id"]
    else:
        warnings.warn(f"Unexpected JSONBin response format: {res_json}")
        return {"response": res_json}

    jsonbin_url = f"https://jsonbin.io/{bin_id}"
    return {"id": bin_id, "url": jsonbin_url, "response": res_json}
    
