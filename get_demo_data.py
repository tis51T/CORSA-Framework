import pandas as pd
import os
from tqdm import tqdm
import json
import re

FOLDER_PATH = "./demo_data/"

def turn_back_to_json(ai_response):
    if isinstance(ai_response, dict):
        return ai_response
    if not isinstance(ai_response, str):
        return None
    match = re.search(r'```(?:json)?(.*?)```', ai_response, re.DOTALL)
    if match:
        ai_response = match.group(1).strip()
    else:
        ai_response = ai_response.strip('`').strip()
    try:
        return json.loads(ai_response)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        return None

def encode_polarity(p):
    return "positive" if p == "POS" else "negative" if p == "NEG" else "neutral"

def encode_type(t):
    return "person" if t == "[PER]" else "location" if t == "[LOC]" else "other"

def process_aspects(df):
    # Convert genai_out to dict if needed
    df["genai_out"] = df["genai_out"].apply(turn_back_to_json)
    df.dropna(subset=["genai_out"], inplace=True)

    # Build aspect/polarity/type merged list
    new_aspects_list = [
    {
        " ".join(a["term"]) if isinstance(a, dict) and "term" in a else str(a): 
        a["polarity"] if isinstance(a, dict) and "polarity" in a else None
        for a in aspects if isinstance(a, dict) or isinstance(a, str)
    }
    for aspects in df["aspects"]
]
    merged_aspects_list = [
        [
            { key: [encode_polarity(new_aspects.get(key)), encode_type(genai_out.get(key))] }
            for key in set(new_aspects.keys()) | set(genai_out.keys())
        ]
        for new_aspects, genai_out in zip(new_aspects_list, df["genai_out"])
    ]
    df["label_aspects"] = merged_aspects_list

    # Build words, aspects, sentiments columns
    words, aspects, sentiments = [], [], []
    for item in merged_aspects_list:
        temp_w, temp_a, temp_s = [], [], []
        for aspect in item:
            for key, (sent, asp) in aspect.items():
                temp_w.append(key)
                temp_s.append(sent)
                temp_a.append(asp)
        words.append(temp_w)
        sentiments.append(temp_s)
        aspects.append(temp_a)

    # Build new DataFrame
    new_df = pd.DataFrame({
        "text": df["words"].apply(lambda x: " ".join(x)),
        "image": df["image_id"].apply(lambda x: f"./twitter2017_images/{x}"),
        "words": words,
        "aspects": aspects,
        "sentiments": sentiments
    })
    return new_df

def create_dataset(version="full"):
    set_shapes = [100, 10, 10]
    for i, set_type in enumerate(["train", "dev", "test"]):
        # Load raw data and process aspects
        df = pd.read_json(os.path.join(FOLDER_PATH, f"{set_type}.json"))
        new_df = process_aspects(df)
        new_df = new_df.sample(frac=1).reset_index(drop=True)  # Shuffle

        # Select demo subset if needed
        if version == "demo":
            demo_df = new_df.head(set_shapes[i]).copy()
        else:
            demo_df = new_df.copy()

        # Copy images to demo folder
        image_idx = demo_df["image"].apply(lambda x: x.split("/")[-1]).tolist()
    
        for img_id in tqdm(image_idx, desc=f"Create demo for {set_type} set"):
            src_img = f"./data/twitter_data/twitter2017_images/{img_id}"
            dst_img = f"./demo_data/images/{img_id}"
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            with open(src_img, "rb") as img_file:
                img_data = img_file.read()
            with open(dst_img, "wb") as demo_img_file:
                demo_img_file.write(img_data)

        demo_df["image"] = demo_df["image"].apply(lambda x: x.replace("./twitter2017_images", "./demo_data/images"))
        outpath = os.path.join(FOLDER_PATH, f"{set_type}_labeled_{version}.json")
        print(f"Saving to {outpath}")
        demo_df.to_json(outpath, orient="records", lines=True)

if __name__ == "__main__":
    create_dataset(version="demo")