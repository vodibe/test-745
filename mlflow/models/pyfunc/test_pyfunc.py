import requests

headers = {
    'Content-Type': 'application/json',
}

json_data = {
    'dataframe_split': {
        'data': [
            [
                6.7,
                3.3,
                5.7,
                2.5,
            ],
        ],
    },
}

response = requests.post('http://localhost:5011/invocations', headers=headers, json=json_data)

print(response.content)