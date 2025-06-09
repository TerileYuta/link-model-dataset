import pandas as pd
import ast
from datetime import datetime

def load_text(file_path):
    contents = ""
    with open(file_path, "r",encoding="utf-8") as f:
        contents = f.read()

    return contents

def create_file_path():
    return datetime.now().strftime("%Y%m%d%H%M")

def print_relations_dict(weight_dict):
    print("-" * 50)
    
    for index, (key, value) in enumerate(weight_dict.items()):
        print(f"{index} : {key} ({value})")

    print("-" * 50)