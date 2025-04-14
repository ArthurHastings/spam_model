import requests
import json

url = "http://localhost:5001/invocations"

data = {
    "dataframe_records": [
        {"text": "Congratulations! You won a free gift. Click here to claim!"}
    ]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, data=json.dumps(data), headers=headers)
predictions = response.json()

print(predictions)
