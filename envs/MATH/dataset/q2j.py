import pandas as pd
import json
import numpy as np

def parquet_to_jsonl(parquet_file, jsonl_file):
    """
    将 Parquet 文件转换为 JSONL 文件
    :param parquet_file: 输入的 Parquet 文件路径
    :param jsonl_file: 输出的 JSONL 文件路径
    """
    try:
        # 读取 Parquet 文件
        df = pd.read_parquet(parquet_file)

        # 将 DataFrame 逐行写入 JSONL
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):  # 每行转换为字典
                f.write(json.dumps(record, ensure_ascii=True) + "\n")  # 写入 JSONL

        print(f"转换完成: {parquet_file} → {jsonl_file}")

    except Exception as e:
        print(f"发生错误: {e}")

def csv_to_jsonl(csv_file, jsonl_file):
    """
    将 CSV 文件转换为 JSONL 文件
    :param csv_file: 输入的 CSV 文件路径
    :param jsonl_file: 输出的 JSONL 文件路径
    """
    try:
        # 读取 CSV 文件
        df = pd.read_csv(csv_file).replace([np.nan, ''], None)

        # 将 DataFrame 逐行写入 JSONL
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):  # 每行转换为字典
                f.write(json.dumps(record, ensure_ascii=True) + "\n")  # 写入 JSONL

        print(f"转换完成: {csv_file} → {jsonl_file}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 直接在代码中指定输入和输出文件路径
    input_file = "/localnvme/application/sc_new/changkaiyan/openr/envs/MATH/dataset/AIME25/data/train-00000-of-00001-243207c6c994e1bd.parquet"
    output_file = "/localnvme/application/sc_new/changkaiyan/openr/envs/MATH/dataset/AIME25/data/test.jsonl"

    # 调用转换函数
    parquet_to_jsonl(input_file, output_file)
    # csv_to_jsonl(input_file, output_file)
