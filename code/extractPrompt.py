import json
import os
import random
from collections import defaultdict


def process_file(input_file, output_file, NUM, filter_by):
    # 1. 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 2. 统计指定字段的不同种类，并过滤掉 answer 为空的项
    filter_dict = defaultdict(list)
    for item in data:
        if item['answer'] != '':  # 只添加 answer 非空的项
            filter_value = item[filter_by]
            filter_dict[filter_value].append(item)

    # 3. 随机挑选每种指定字段的 NUM 项（或全部，如果数量小于 NUM）
    result = []
    for filter_value, items in filter_dict.items():
        if len(items) <= NUM:
            result.extend(items)
        else:
            result.extend(random.sample(items, NUM))

    # 4. 保存结果到新 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    print(f"筛选后的数据已保存到 '{output_file}'")


def extract(sample_num, filter_by):
    input_folder = 'prompt'  # 存放原始 JSON 文件的文件夹
    output_folder = 'extract'  # 存放处理后 JSON 文件的文件夹

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有 JSON 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            # 去掉扩展名，重新生成输出文件名
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_folder, f"extracted_{base_name}_{filter_by}_{sample_num}.json")
            input_file = os.path.join(input_folder, filename)
            process_file(input_file, output_file, sample_num, filter_by)
