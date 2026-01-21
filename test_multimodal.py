#!/usr/bin/env python3
"""Test multimodal endpoint"""
import requests
import json

# Test text-only processing
print("Testing text-only multimodal processing...")
print("=" * 60)

url = "http://127.0.0.1:5000/api/multimodal/process"
payload = {
    "text": "There is a large pothole on Main Street causing damage to vehicles. It needs immediate repair."
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse:")
    print(json.dumps(response.json(), indent=2))
    
    if response.status_code == 200:
        print("\n✓ Multimodal endpoint is working!")
        data = response.json()['data']
        print(f"\n  - Text: {data['text']}")
        print(f"  - Modality: {data['modality']}")
        print(f"  - Embedding dimension: {len(data['embedding'])}")
    else:
        print("\n✗ Error occurred")
        
except Exception as e:
    print(f"\n✗ Failed to connect: {e}")
    print("\nMake sure the Flask server is running:")
    print("  ./venv/Scripts/python app.py")
