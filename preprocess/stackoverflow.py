import pandas as pd
import os
import torch
from datasets import Dataset

def preprocess_stackoverflow(input_csv: str, output_dir: str, seed: int = 42):
    # 读取CSV
    df = pd.read_csv(input_csv)

    # 映射标签
    string_to_int = {
        'open': 0,
        'not a real question': 1,
        'off topic': 1,
        'not constructive': 1,
        'too localized': 1
    }
    df['OpenStatusInt'] = df['OpenStatus'].map(string_to_int)

    # 特征工程
    df['BodyLength'] = df['BodyMarkdown'].apply(lambda x: len(x.split(" ")))
    df['TitleLength'] = df['Title'].apply(lambda x: len(x.split(" ")))
    df['TitleConcatWithBody'] = df.apply(lambda x: x.Title + " " + x.BodyMarkdown, axis=1)
    df['NumberOfTags'] = df.apply(
        lambda x: len([x[col] for col in ['Tag1','Tag2','Tag3','Tag4','Tag5'] if not pd.isna(x[col])]),
        axis=1,
    )
    df['PostCreationDate'] = pd.to_datetime(df['PostCreationDate'])
    df['OwnerCreationDate'] = pd.to_datetime(df['OwnerCreationDate'], format='mixed')
    df['DayDifference'] = (df['PostCreationDate'] - df['OwnerCreationDate']).dt.days

    # 选择 tabular 特征
    tabular_feature_list = [
        'ReputationAtPostCreation',
        'BodyLength',
        'TitleLength',
        'NumberOfTags',
        'DayDifference',
    ]

    # 转换为 Hugging Face Dataset
    data_dict = {
        'text': df.TitleConcatWithBody.tolist(),
        'tabular': df[tabular_feature_list].values,
        'label': df.OpenStatusInt.tolist(),
    }
    dataset = Dataset.from_dict(data_dict)

    # shuffle + split
    n_samples = len(dataset)
    split_idx1 = int(n_samples * 0.8)
    split_idx2 = int(n_samples * 0.9)
    shuffled = dataset.shuffle(seed=seed)

    train_dataset = shuffled.select(range(split_idx1))
    val_dataset   = shuffled.select(range(split_idx1, split_idx2))
    test_dataset  = shuffled.select(range(split_idx2, n_samples))

    # 标准化 (fit on train)
    mean_train = torch.mean(torch.tensor(train_dataset['tabular'], dtype=torch.float32), dim=0)
    std_train  = torch.std(torch.tensor(train_dataset['tabular'], dtype=torch.float32), dim=0)

    def standard_scale(example):
        example['tabular'] = ((torch.tensor(example['tabular']) - mean_train) / std_train).tolist()
        return example

    train_dataset = train_dataset.map(standard_scale)
    val_dataset   = val_dataset.map(standard_scale)
    test_dataset  = test_dataset.map(standard_scale)

    # 保存到 Arrow 格式
    dataset_dict = {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }
    # 提取数据名（去掉路径和扩展名）
    data_name = os.path.splitext(os.path.basename(input_csv))[0]

    # 拼接最终保存根目录
    save_root = os.path.join(output_dir, data_name)

    # 确保根目录存在
    os.makedirs(save_root, exist_ok=True)

    # 保存每个 split
    for split_name, split_data in dataset_dict.items():
        split_save_path = os.path.join(save_root, split_name)
        os.makedirs(split_save_path, exist_ok=True)  # 确保 split 目录存在
        split_data.save_to_disk(split_save_path)

    print(f"✅ 数据已保存到 {output_dir} (Arrow 格式)")
    return dataset_dict


if __name__ == "__main__":
    preprocess_stackoverflow(
        input_csv="/data/lj/task/BERT-LoRA-TensorRT/data/train-sample.csv",
        output_dir="../data/"
    )
    print(os.path.abspath("../data/"))
