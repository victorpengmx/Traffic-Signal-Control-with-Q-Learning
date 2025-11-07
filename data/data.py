import requests

url = "https://datamall2.mytransport.sg/ltaodataservice/TrafficFlow"
headers = {
    "AccountKey": "eYTZ2jlDRlS4jX9/XnnvBQ==",
    "accept": "application/json"
}

response = requests.get(url, headers=headers)
data = response.json()

# Get the actual traffic flow dataset link
dataset_url = data["value"][0]["Link"]
print("Download link:", dataset_url)
