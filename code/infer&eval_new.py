import time
import google
import requests
import retrying
import re
import os
import json
import base64
import openai
import dashscope
from dashscope import MultiModalConversation
import google.generativeai as genai
import PIL.Image
from tqdm import tqdm

from eval_old import evaluate_json
from extractPrompt import extract
from prompt.prompt import EVALUATION_SYSTEM_PROMPT
from Img2Text import image_to_text
from ques2prompt import format_question

GOOGLE_API_KEY = ''
genai.configure(api_key=GOOGLE_API_KEY, transport='rest')


@retrying.retry(wait_fixed=2000, stop_max_attempt_number=6, retry_on_exception=lambda e: isinstance(e, (
google.api_core.exceptions.InternalServerError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError)))
def GeminiPro(image_paths, item, isInputKaodian, isInputRec):
    prompt = f"{EVALUATION_SYSTEM_PROMPT}\n\n{item['question']}\n\n"
    model = genai.GenerativeModel('gemini-1.5-flash')
    images = [PIL.Image.open(path).convert('RGB') for path in image_paths]

    if isInputKaodian and item['kaodian']:
        prompt += "考点梳理：\n"
        for i, point in enumerate(item['kaodian'], 1):
            prompt += f"{i}. {point['标题']}：{point['内容']}\n"
        prompt += "\n"

    if isInputRec and item['recom']:
        if item['answer']['答案']:
            origin_answer = item['answer']['答案']
            time.sleep(2)
        else:
            # 处理答案图片路径数组
            image_paths = item['answer']['答案图片']
            processed_texts = []

            for path in image_paths:
                try:
                    # 添加 "gz-data/" 前缀
                    full_path = 'gz-data/' + path
                    # 调用 image_to_text 函数
                    text = image_to_text(full_path)
                    processed_texts.append(text)
                except Exception as e:
                    print(f"处理图片 {path} 时出错: {str(e)}")
                    continue  # 跳过这个图片，继续处理下一个

            # 用 "\n" 连接所有处理后的文本
            origin_answer = "\n".join(processed_texts)
        prompt += "类题推荐：\n"
        if item['recom']:
            first_rec = item['recom'][0]
            formatted_rec = format_question(first_rec)
            recPrompt = '问题：\n' + formatted_rec['question'] + '\n答案：\n' + origin_answer
            prompt += recPrompt + "\n"
        prompt += "\n"
    contents = [*images, prompt]
    response = model.generate_content(contents, stream=False)
    response.resolve()
    return response.text


# 设置您的API密钥
openai.api_key = ''

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}"
}
@retrying.retry(wait_fixed=2000, stop_max_attempt_number=6, retry_on_exception=lambda e: isinstance(e, (requests.exceptions.ReadTimeout,requests.exceptions.ConnectionError)))
def gpt4(image_paths, item, isInputKaodian, isInputRec, model_name):
    if model_name == 'gpt4':
        model = "gpt-4-vision-preview"
    else:
        model = "gpt-4o"
    prompt = f"问题：\n{item['question']}\n\n"

    if isInputKaodian and item['kaodian']:
        prompt += "考点梳理：\n"
        for i, point in enumerate(item['kaodian'], 1):
            prompt += f"{i}. {point['标题']}：{point['内容']}\n"
        prompt += "\n"

    if isInputRec and item['recom']:
        if item['answer']['答案']:
            origin_answer = item['answer']['答案']
            time.sleep(2)
        else:
            # 处理答案图片路径数组
            image_paths = item['answer']['答案图片']
            processed_texts = []

            for path in image_paths:
                try:
                    # 添加 "gz-data/" 前缀
                    full_path = 'gz-data/' + path
                    # 调用 image_to_text 函数
                    text = image_to_text(full_path)
                    processed_texts.append(text)
                except Exception as e:
                    print(f"处理图片 {path} 时出错: {str(e)}")
                    continue  # 跳过这个图片，继续处理下一个

            # 用 "\n" 连接所有处理后的文本
            origin_answer = "\n".join(processed_texts)
        prompt += "类题推荐：\n"
        if item['recom']:
            first_rec = item['recom'][0]
            formatted_rec = format_question(first_rec)
            recPrompt = '问题：\n' + formatted_rec['question'] + '\n答案：\n' + origin_answer
            prompt += recPrompt + "\n"
        prompt += "\n"
    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": EVALUATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 3000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()
    # 获取模型生成的内容
    raw_content = res['choices'][0]['message']['content']

    # 清理内容，移除可能的 ```json 和 ``` 标记
    cleaned_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_content.strip())

    # 打印清理后的内容
    return cleaned_content


dashscope.api_key = ""


@retrying.retry(wait_fixed=2000, stop_max_attempt_number=6, retry_on_exception=lambda e: isinstance(e, (
requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError)))
def qwen(image_paths, item, isInputKaodian, isInputRec):
    model = "qwen-vl-plus"

    prompt = f"问题：\n{item['question']}\n\n你的输出必须以json的格式，并且可以被json.loads解析且包含analysis和answer两个key,且这两个key的value都必须是字符串格式。"

    if isInputKaodian and item['kaodian']:
        prompt += "考点梳理：\n"
        for i, point in enumerate(item['kaodian'], 1):
            prompt += f"{i}. {point['标题']}：{point['内容']}\n"
        prompt += "\n"

    if isInputRec and item['recom']:
        if item['answer']['答案']:
            origin_answer = item['answer']['答案']
            time.sleep(2)
        else:
            # 处理答案图片路径数组
            image_paths = item['answer']['答案图片']
            processed_texts = []

            for path in image_paths:
                try:
                    # 添加 "gz-data/" 前缀
                    full_path = 'gz-data/' + path
                    # 调用 image_to_text 函数
                    text = image_to_text(full_path)
                    processed_texts.append(text)
                except Exception as e:
                    print(f"处理图片 {path} 时出错: {str(e)}")
                    continue  # 跳过这个图片，继续处理下一个

            # 用 "\n" 连接所有处理后的文本
            origin_answer = "\n".join(processed_texts)
        prompt += "类题推荐：\n"
        if item['recom']:
            first_rec = item['recom'][0]
            formatted_rec = format_question(first_rec)
            recPrompt = '问题：\n' + formatted_rec['question'] + '\n答案：\n' + origin_answer
            prompt += recPrompt + "\n"
        prompt += "\n"

    content = [{"text": prompt}]

    for image_path in image_paths:
        ab_img_path = 'file://' + os.path.abspath(image_path)
        content.append({
            "image": ab_img_path
        })

    messages = [
        {
            'role': 'system',
            'content': [{"text": EVALUATION_SYSTEM_PROMPT}]
        },
        {
            'role': 'user',
            'content': content
        }
    ]

    response = MultiModalConversation.call(model=model, messages=messages)

    # 获取模型生成的内容
    raw_content = response['output']['choices'][0]['message']['content'][0]['text']

    # 清理内容，移除可能的 ```json 和 ``` 标记
    cleaned_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_content.strip())

    return cleaned_content


def extract_content(text, key):
    pattern = f'"{key}":\s*"((?:[^"\\\\]|\\\\.)*)"'
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def create_manual_dict(text):
    analysis = extract_content(text, "analysis")
    answer = extract_content(text, "answer")
    return {
        "analysis": analysis,
        "answer": answer
    }


def process_questions(input_file, output_file, isInputKaodian, isInputRec, model_name):
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    results = []
    for item in tqdm(questions, desc='Processing Questions'):
        question = item['question']
        image_paths = item['imgs']
        if item['answer']['答案']:
            origin_answer = item['answer']['答案']
            time.sleep(2)
        else:
            # 处理答案图片路径数组
            image_paths = item['answer']['答案图片']
            processed_texts = []

            for path in image_paths:
                try:
                    # 添加 "gz-data/" 前缀
                    full_path = 'gz-data/' + path
                    # 调用 image_to_text 函数
                    text = image_to_text(full_path)
                    processed_texts.append(text)
                except Exception as e:
                    print(f"处理图片 {path} 时出错: {str(e)}")
                    continue  # 跳过这个图片，继续处理下一个

            # 用 "\n" 连接所有处理后的文本
            origin_answer = "\n".join(processed_texts)
            if origin_answer == '':
                continue
        new_image_paths = []
        for path in image_paths:
            path = 'gz-data/' + path
            new_image_paths.append(path)

        try:
            MAX_RETRIES = 3
            RETRY_DELAY = 2  # 每次重试之间等待的秒数
            for retry in range(MAX_RETRIES):
                try:
                    if model_name == 'gemini':
                        answer = GeminiPro(new_image_paths, item, isInputKaodian, isInputRec)
                    elif model_name == 'gpt4' or model_name == 'gpt4o':
                        answer = gpt4(new_image_paths, item, isInputKaodian, isInputRec, model_name)
                    elif model_name == 'qwen':
                        answer = qwen(new_image_paths, item, isInputKaodian, isInputRec)
                    else:
                        print('请输入正确的模型名称')
                        break

                    try:
                        answer_dict = json.loads(answer)
                    except json.JSONDecodeError:
                        print("JSON解析失败，尝试手动提取内容")
                        answer_dict = create_manual_dict(answer)
                        if not answer_dict["analysis"] and not answer_dict["answer"]:
                            raise ValueError("无法提取有效的 analysis 和 answer")

                    break  # 如果成功，跳出重试循环

                except json.JSONDecodeError as e:
                    print(f"JSON解析错误 (尝试 {retry + 1}/{MAX_RETRIES}): {e}")
                    if retry < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)  # 等待一段时间后重试
                    else:
                        print(f"在 {MAX_RETRIES} 次尝试后仍然失败，跳过此项")
                        continue  # 跳过当前循环的剩余部分，进入下一次循环
                except Exception as e:
                    print(f"其他错误 (尝试 {retry + 1}/{MAX_RETRIES}): {e}")
                    if retry < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        print(f"在 {MAX_RETRIES} 次尝试后仍然失败，跳过此项")
                        continue

            # 如果所有重试都失败，这里的 answer_dict 将不会被定义
            # 你可能想要在这里添加一个检查
            if 'answer_dict' not in locals():
                print("无法处理此项，继续下一个")
                continue

            # 现在你可以访问字典中的键值对
            llm_analysis = answer_dict.get("analysis", "")
            llm_answer = answer_dict.get("answer", "")
            if isinstance(llm_analysis, str) and isinstance(llm_answer, str):
                results.append({
                    'question_id': item['question_id'],
                    'type': item['type'],
                    'difficulty': item['difficulty'],
                    'question': question,
                    'original_answer': origin_answer,
                    'llm_answer': llm_answer,
                    'original_analysis': item['analysis'],
                    'llm_analysis': llm_analysis
                })
        except Exception as e:
            print(f"Error processing question: {e}")
            continue

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("推理完成，结果保存至", output_file)


if __name__ == '__main__':
    models = ['gemini', 'gpt4', 'gpt4o', 'qwen']  # 模型列表
    subjects = ['biology', 'chemistry', 'geography', 'history', 'math', 'physics']  # 学科列表
    filter_bys = ['type', 'difficulty']  # 筛选条件
    sample_num = 1  # 抽取数量
    isInputKaodian = True  # 是否输入考点梳理
    isInputRec = True  # 是否输入类题推荐

    for filter_by in filter_bys:
        # 提取
        print(f'开始按{filter_by}抽取')
        extract(sample_num, filter_by)

        # 推理
        print('开始推理')
        for model_name in models:
            for subject in subjects:
                input_file = f'extract/extracted_{subject}.json'  # 输入文件名
                output_file = f'results/{subject}_result_{model_name}_{filter_by}.json'  # 输出文件名
                process_questions(input_file, output_file, isInputKaodian, isInputRec, model_name)

        # 评估
        print('开始评估')
        for model_name in models:
            for subject in subjects:
                output_file = f'results/{subject}_result_{model_name}_{filter_by}.json'  # 输出文件名
                eval_output_file = f'eval_result/{subject}_evaluation_results_{model_name}_{sample_num}_{filter_by}.txt'
                evaluate_json(output_file, eval_output_file)
