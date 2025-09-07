import pandas as pd
import os
os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm

def main():
    tsv_path = "./data/twitter_data/twitter2017/"
    set_shapes = [100, 50, 50]

    for i, set_type in enumerate(["train", "dev", "test"]):
        df = pd.read_csv(os.path.join(tsv_path, f"{set_type}.tsv"), sep="\t").drop(["index"], axis=1)
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

        demo_df = df.head(set_shapes[i])
        demo_df.to_json("./demo_data/" + f"{set_type}_demo.json", orient="records", lines=True)

        if "#2 ImageID" in demo_df.columns:
            image_idx = demo_df["#2 ImageID"].values
        else:
            image_idx = demo_df["#1 ImageID"].values
            
        for img_id in tqdm(image_idx, desc=f"Create demo for {set_type} set"):
            # Read the source image
            with open(f"./data/twitter_data/twitter2017_images/{img_id}", "rb") as img_file:
                img_data = img_file.read()

            # Create the destination directory if it doesn't exist
            os.makedirs(f"./demo_data/images/{set_type}/", exist_ok=True)

            # Write the image to the destination directory
            with open(f"./demo_data/images/{set_type}/{img_id}", "wb") as demo_img_file:
                demo_img_file.write(img_data)
        
        print(f"Saved {set_type} demo data with shape: {demo_df.shape}")

if __name__ == "__main__":
    main()