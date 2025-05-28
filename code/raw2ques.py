import json
import os


def extract_questions(data):
    questions = []

    def recursive_extract(item):
        if isinstance(item, dict):
            if 'id' in item:
                questions.append(item)
            for value in item.values():
                recursive_extract(value)
        elif isinstance(item, list):
            for element in item:
                recursive_extract(element)

    recursive_extract(data)

    # 过滤掉没有 "解析" key 的项目
    filtered_questions = [q for q in questions if "解析" in q]

    return filtered_questions

if __name__ == '__main__':

    # 获取 gz-data 文件夹中所有的 *_questions.json 文件
    input_folder = 'gz-data'
    output_folder = 'questions'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历 gz-data 文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('_questions.json'):
            # 构造输入和输出文件路径
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace('_questions.json', '.json')
            output_path = os.path.join(output_folder, output_filename)

            # 读取原始JSON文件
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取所有题目
            all_questions = extract_questions(data)

            # 将结果写入新的JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_questions, f, ensure_ascii=False, indent=2)

            print(f"已处理 {filename}: 提取了 {len(all_questions)} 个题目并保存到 {output_filename}")

    print("所有文件处理完成")