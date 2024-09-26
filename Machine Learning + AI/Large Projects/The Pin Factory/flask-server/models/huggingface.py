import requests
import os

API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload, api_url):
    response = requests.post(api_url, headers=headers, data=payload)
    return response.json()

def make_hf_call(model, input, call_type):

    api_url =  f"https://api-inference.huggingface.co/models/{model}"

    if call_type == "image":
        payload = construct_image_payload(input) 
    else:
        pass

    data = query(payload, api_url)

    return data


def construct_image_payload(img_filename):
    with open(img_filename, "rb") as f:
        return f.read()
        


if __name__ == "__main__":
    model = "google/vit-base-patch16-224"
    current_file_path = os.path.abspath(__file__)

    image_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), 'images', 'img.png')
    call_type = "image"
    data = make_hf_call(model, image_path, call_type)
    print(data)
    
