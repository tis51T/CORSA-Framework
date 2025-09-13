import pandas as pd
import os
from tqdm import tqdm
# import base64
# Google Generative AI SDK
import base64
from google import genai
from google.genai import types


import json


def main():
    json_path = "./data/twitter_data/twitter2017_segmented/"
    set_shapes = [30, 10, 10]

    for i, set_type in enumerate(["train", "dev", "test"]):
        df = pd.read_json(os.path.join(json_path, f"{set_type}.json"))
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

        demo_df = df.head(set_shapes[i])

        words = demo_df["words"].tolist()
        aspects = demo_df["aspects"].tolist()

        sentences = []
        terms_list = []

        for j in range(len(words)):
            sentence = " ".join(words[j])
            sentences.append(sentence)

            terms = []
            for k in range(len(aspects[j])):
                terms.append(" ".join(aspects[j][k]["term"]))
            terms_list.append(terms)

        outs = []
        for j in range(len(terms_list)):
            try:
                genai_output = generate(sentence=sentences[j], terms=terms_list[j])
                print(f"Sentence: {sentences[j]} - Terms: {terms_list[j]} - GenAI Output: {genai_output}")
                outs.append(genai_output)
            except Exception as e:
                outs.append(None)

        demo_df["genai_output"] = outs
        demo_df.to_json("./demo_data/" + f"{set_type}_demo.json", orient="records", lines=True)

        image_idx = demo_df["image_id"].tolist()
        
        # for img_id in tqdm(image_idx, desc=f"Create demo for {set_type} set"):
        #     # Read the source image
        #     with open(f"./data/twitter_data/twitter2017_images/{img_id}", "rb") as img_file:
        #         img_data = img_file.read()

        #     # Create the destination directory if it doesn't exist
        #     os.makedirs(f"./demo_data/images/{set_type}/", exist_ok=True)

        #     # Write the image to the destination directory
        #     with open(f"./demo_data/images/{set_type}/{img_id}", "wb") as demo_img_file:
        #         demo_img_file.write(img_data)
        
        # print(f"Saved {set_type} demo data with shape: {demo_df.shape}")

def generate(sentence, terms):
    client = genai.Client(
        api_key="AIzaSyBTjcqmChEUS6jxMglJPowgJnDXHvLZMto",
    )

    full_prompt = f"""
    Expect you are a data labeler, I will give you a sentence and terms in sentence. Your task is classifying these terms into three aspect: [PER] for person, [LOC] for location, and [OTH] for other things. Your output must be followed these constraints:
        - Return only one label, no reasoning
        - Output format: 'label': [PER]/[PLACE]/[THING]; and being dictionary format

        Input: 
            "sentence": {sentence}, 
            "terms":{terms}
            
        Output:
    """

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=full_prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
    )

    full_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        # print(chunk.text, end="")
        full_text += chunk.text

    return full_text

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     main()