import json
import os
import re
import time


def remove_all_spaces(text):
    if isinstance(text, str):
        return ''.join(text.split())
    elif isinstance(text, dict):
        return {remove_all_spaces(k): remove_all_spaces(v) for k, v in text.items()}
    elif isinstance(text, list):
        return [remove_all_spaces(item) for item in text]
    else:
        return text


def replace_nbsp_with_blanks(text):
    return re.sub(r'\xa0{2,}', '（）', text)


def format_question(question):
    try:
        if question['类型'] == '多项选择':
            content = question.get('内容', '')
            if all(char in content for char in ['A', 'B', 'C', 'D']):
                text = remove_all_spaces(question.get('文本', ''))
                imgs = question.get('图片', [])
                type = question.get('类型', '')
                formatted_question = f"{text}\n该题的类型为{type}\n请结合输入的图片，直接回答正确选项的字母，并且对每个选项进行分析，正确选项可能不止一个。"
                return {
                    'question_id': question.get('id', ''),
                    'question': formatted_question,
                    'answer': remove_all_spaces(question.get('答案', '')),
                    'difficulty': question.get('难度', ''),
                    'imgs': imgs,
                    'type': type,
                    'analysis': question.get('解析', ''),
                    'kaodian': question.get('考点梳理', []),
                    'recom': question.get('类题推荐', [])
                }
            else:
                question['类型'] = '填空题'
        if question['类型'] == '选择题':
            text = remove_all_spaces(question.get('文本', ''))
            content = remove_all_spaces(question.get('内容', {}))
            imgs = question.get('图片', [])
            type = question.get('类型', '')
            formatted_content = '\n'.join([f"{key}. {value}" for key, value in content.items()])
            formatted_question = f"{text}\n{formatted_content}\n该题的类型为{type}\n请结合输入的图片，直接回答正确选项的字母，并且对每个选项进行分析，正确选项只有一个。"
            return {
                'question_id': question.get('id', ''),
                'question': formatted_question,
                'answer': remove_all_spaces(question.get('答案', '')),
                'difficulty': question.get('难度', ''),
                'imgs': imgs,
                'type': type,
                'analysis': question.get('解析', ''),
                'kaodian': question.get('考点梳理', []),
                'recom': question.get('类题推荐', [])
            }
        elif question['类型'] == '填空题':
            text = replace_nbsp_with_blanks(question.get('文本', ''))
            imgs = question.get('图片', [])
            type = question.get('类型', '')
            formatted_question = f"{text}\n该题的类型为{type}\n请结合输入的图片，给出此题的解析以及题目中需要填写的答案。"
            return {
                'question_id': question.get('id', ''),
                'question': formatted_question,
                'answer': remove_all_spaces(question.get('答案', '')),
                'difficulty': question.get('难度', ''),
                'imgs': imgs,
                'type': type,
                'analysis': question.get('解析', ''),
                'kaodian': question.get('考点梳理', []),
                'recom': question.get('类题推荐', [])
            }
        elif question['类型'] == '作图题' or question['类型'] == '连线题' or question['类型'] == '填表绘图题':
            text = replace_nbsp_with_blanks(question.get('文本', ''))
            imgs = question.get('图片', [])
            type = question.get('类型', '')
            formatted_question = f"{text}\n该题的类型为{type}\n请根据以上要求结合输入的图片给出此题的作图步骤作为解析以及最后的答案(无需画图)。"
            return {
                'question_id': question.get('id', ''),
                'question': formatted_question,
                'answer': question.get('答案', ''),
                'difficulty': question.get('难度', ''),
                'imgs': imgs,
                'type': type,
                'analysis': question.get('解析', ''),
                'kaodian': question.get('考点梳理', []),
                'recom': question.get('类题推荐', [])
            }
        elif question['类型'] == '计算题' or question['类型'] == '解答题':
            text = replace_nbsp_with_blanks(question.get('文本', ''))
            imgs = question.get('图片', [])
            type = question.get('类型', '')
            formatted_question = f"{text}\n该题的类型为{type}\n请给出每个小问对应的解题步骤以及每个小问最后的答案。根据以上要求结合输入的图片，生成解析和答案"
            return {
                'question_id': question.get('id', ''),
                'question': formatted_question,
                'answer': question.get('答案', ''),
                'difficulty': question.get('难度', ''),
                'imgs': imgs,
                'type': type,
                'analysis': question.get('解析', ''),
                'kaodian': question.get('考点梳理', []),
                'recom': question.get('类题推荐', [])
            }
        else:
            text = replace_nbsp_with_blanks(question.get('文本', ''))
            imgs = question.get('图片', [])
            type = question.get('类型', '')
            formatted_question = f"{text}\n该题的类型为{type}\n请结合输入的图片，给出每个小问对应的解析以及每个小问需要填写的答案。"
            return {
                'question_id': question.get('id', ''),
                'question': formatted_question,
                'answer': remove_all_spaces(question.get('答案', '')),
                'difficulty': question.get('难度', ''),
                'imgs': imgs,
                'type': type,
                'analysis': question.get('解析', ''),
                'kaodian': question.get('考点梳理', []),
                'recom': question.get('类题推荐', [])
            }
    except Exception as e:
        print(f"处理题目时出错: {str(e)}")
        return None


if __name__ == '__main__':

    # 设置输入和输出文件夹
    input_folder = 'questions'
    output_folder = 'prompt'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历 questions 文件夹中的所有 JSON 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            # 构造输入和输出文件路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取上一个脚本的输出文件
            with open(input_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)

            # 格式化题目
            formatted_questions = [format_question(q) for q in questions]
            formatted_questions = [q for q in formatted_questions if q is not None]

            # 去重
            unique_questions = []
            seen_questions = set()
            for q in formatted_questions:
                question_text = q['question']
                if question_text not in seen_questions:
                    seen_questions.add(question_text)
                    unique_questions.append(q)

            # 将结果写入新的JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(unique_questions, f, ensure_ascii=False, indent=2)

            print(f"处理文件 {filename}:")
            print(f"  原始题目数量: {len(formatted_questions)}")
            print(f"  去重后题目数量: {len(unique_questions)}")

    print("所有文件处理完成")