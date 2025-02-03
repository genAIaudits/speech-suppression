# https://docs.modelslab.com/image-generation/realtime-stable-diffusion/text2img

import requests
import json
import pandas as pd, os, sys, time
# https://www.geeksforgeeks.org/python-import-from-parent-directory/
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)

url = "https://modelslab.com/api/v6/realtime/text2img"

def get_image(key, prompt):
    payload = json.dumps({
        "key" : key,
        "prompt": prompt,
        "negative_prompt": "bad quality",
        "width": "512",
        "height": "512",
        "safety_checker": False,
        "seed": None,
        "samples":1,
        "base64":False,
        "webhook": None,
        "track_id": None
    })

    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    return response.text


def main():
    short_synopsis_data = pd.read_csv("Data/TV Shows/short_TMDB_data.csv").sample(1)
    prompt_body = "Make a show poster based on the following synopsis: "
    key = ""
    
    for show_name, episode_title, synopsis in short_synopsis_data[["show_name", "episode_title", "episode-overview"]].to_numpy():
        print("Show:", show_name)
        print("Episode:", episode_title)
        print("Synopsis:", synopsis, "\n")

        print(get_image(key, prompt=prompt_body + synopsis))
    
    
if __name__ == "__main__":
    main()