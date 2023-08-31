import requests

def convert_to_stl(file_data):
    url = "https://cadconvert.fly.dev/convert/"
    headers = {'accept': '*/*'}
    files = {'file': ('file.step', file_data, 'application/vnd.ms-pki.stl')}  # Change the filename and content type
    params = {'inputformat': 'step', 'outputformat': 'stl'}  # Add parameters for conversion

    response = requests.request("POST", url, params=params, headers=headers, files=files)

    if response.status_code != 200:
        raise Exception(f"Error occurred: {response.text}")

    return response.content