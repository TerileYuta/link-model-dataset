
class Config:
    relation_dict = {
        "共起": "同じ文/段落に出現",
        "参照": "AがBを参照",
        "説明": "AがBを説明する",
        "同値": "まったく同じ意味",
        "抱含": "AはBに含まれる",
        "従属": "AはBに従属する",
        "類似": "ほぼ同じ式（数字がかわっただけ）",
        "対義": "意味が反対",
    }

    relation_num = len(relation_dict)

    dataset_path = "./dataset"
    triplets_file_path = f"{dataset_path}/triplets.npy"
    prompt_path = "./prompts/categories.txt"
