import requests
import os
import argparse
import sys

def upload_file(gateway_url, api_key, group_id, local_file_path, remote_file_name=None, content_type=None):
    """
    Automates the process of requesting a pre-signed URL and uploading a file.
    """
    if not os.path.exists(local_file_path):
        print(f"Error: Local file '{local_file_path}' not found.")
        return False
    
    file_name = remote_file_name or os.path.basename(local_file_path)
    
    # 1. Request pre-signed URL
    print(f"[*] Requesting pre-signed URL for '{file_name}'...")
    payload = {
        "group_id": group_id,
        "file_key": file_name,
        "action": "upload"
    }
    if content_type:
        payload["content_type"] = content_type
        
    headers = {
        "X-Internal-Token": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.post(f"{gateway_url}/api/internal/storage/generate_presigned_url", json=payload, headers=headers)
        resp.raise_for_status()
        url_data = resp.json()
        target_url = url_data["url"]
    except Exception as e:
        print(f"[-] Failed to get pre-signed URL: {e}")
        if 'resp' in locals() and resp.status_code == 422:
            print(f"    Detail: {resp.text}")
        return False

    # 2. Perform the upload
    print(f"[*] Starting upload to OBS...")
    upload_headers = {}
    if content_type:
        upload_headers["Content-Type"] = content_type
        
    try:
        with open(local_file_path, 'rb') as f:
            # We use Put because it's a pre-signed URL
            put_resp = requests.put(target_url, data=f, headers=upload_headers)
            put_resp.raise_for_status()
            print("[+] Upload successful!")
            return True
    except Exception as e:
        print(f"[-] Upload failed: {e}")
        if 'put_resp' in locals():
            print(f"    Server response: {put_resp.text}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-click test tool for Cloud Storage Gateway")
    parser.add_argument("--gateway", default="http://101.245.110.98:8000", help="Gateway URL")
    parser.add_argument("--key", default="test_secret_key", help="Internal API Key")
    parser.add_argument("--group", default="federated_group_1", help="Group ID")
    parser.add_argument("--file", required=True, help="Path to local file to upload")
    parser.add_argument("--type", help="Content-Type (e.g. image/png)")
    
    args = parser.parse_args()
    
    success = upload_file(args.gateway, args.key, args.group, args.file, content_type=args.type)
    sys.exit(0 if success else 1)
