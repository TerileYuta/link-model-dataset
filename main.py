import pandas as pd
import numpy as np
from ast import literal_eval   
import os

from config import Config
from similarity import get_cosine_similarity
from openai_api import openai_generate
from util import create_file_path, load_text, print_relations_dict

categories_prompt = load_text(Config.prompt_path)

def ai_suggestion(text_1, text_2):
    response = openai_generate(
        categories_prompt,
        f"A : {text_1}\nB : {text_2}"
    )
    print(f"AI : {response}")

def annotation(sample_1, sample_2, triplets):
    sample_1_word = sample_1['word'].replace("\n", "")
    sample_2_word = sample_2['word'].replace("\n", "")

    print(f"A:({sample_1['id']}) {sample_1_word}")
    print("↓")
    print(f"B:({sample_2['id']}) {sample_2_word}")

    print_relations_dict(Config.relation_dict)

    ai_suggestion(
        sample_1_word, 
        sample_2_word
    )
    
    key_input = input("> ")

    if key_input != "" and key_input != "-":
        key_list = key_input.split(" ")

        for key in key_list:
            key = int(key)
            if key >= Config.relation_num: continue

            triplets = np.vstack([triplets, [[int(sample_1['id']), key, int(sample_2['id'])]]])

            # 対義の場合は双方向
            if key == "7":
                triplets = np.vstack([triplets, [[int(sample_2['id']), key, int(sample_1['id'])]]])

    elif key_input == "-":
        annotation(sample_2, sample_1, triplets)

    return triplets


def main():
    print("データをロード中です。")

    target_df = pd.read_csv("./token/unique.csv", header=0, names=["id", "type", "word", "token"])
    target_df["token"] = target_df["token"].apply(literal_eval)

    while True:
        try:
            triplets_files = [f for f in os.listdir(Config.dataset_path) if f.endswith(".npy")]
            triplets_file_path = triplets_files[0]
            triplets = np.load(triplets_file_path)
        except:
            triplets = np.empty((0, 3), dtype=int)

            if not os.path.exists(Config.dataset_path):
                os.mkdir(Config.dataset_path)
            
            triplets_file_path = os.path.join(Config.dataset_path, f"{create_file_path()}.npy")

        target = target_df.sample().iloc[0]

        similars = get_cosine_similarity(target_df["token"].tolist(), target["token"])

        for similar in similars[1:]:
            similar_word = target_df.iloc[similar]

            triplets = annotation(target, similar_word, triplets)

            np.save(triplets_file_path, triplets)

            print("--------------------------------------------------")


if __name__ == "__main__":
    main()