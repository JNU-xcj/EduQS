import json
from collections import defaultdict

import jieba
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk

nltk.download('punkt')

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
model = BertModel.from_pretrained('./bert-base-chinese').to(device)


def calculate_cosine_similarity_bert(sentence1, sentence2):
    sentence1 = ensure_string(sentence1)
    sentence2 = ensure_string(sentence2)

    def encode_sentence(sentence):
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        # 使用attention_mask来只平均有效的token表示
        # last_hidden_state = outputs.last_hidden_state
        # attention_mask = inputs['attention_mask'].unsqueeze(-1)
        # sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
        # sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        # return sum_embeddings / sum_mask
        # 提取CLS token的表示作为句子向量
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 第0个token即CLS token

        # 进行标准化
        cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)

        return cls_embedding

    vec1 = encode_sentence(sentence1)
    vec2 = encode_sentence(sentence2)
    similarity = cosine_similarity(vec1.cpu(), vec2.cpu())
    return similarity[0][0].item()


def ensure_string(input_data):
    return str(input_data)


def calculate_bleu(reference, candidate):
    reference = ensure_string(reference)
    candidate = ensure_string(candidate)

    reference_tokens = list(jieba.cut(reference))
    candidate_tokens = list(jieba.cut(candidate))
    weights = (0.25, 0.25, 0.25, 0.25)  # 1-gram to 4-gram weights
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], candidate_tokens, weights=weights, smoothing_function=smoothing)


def calculate_rouge(reference, candidate):
    reference = ensure_string(reference)
    candidate = ensure_string(candidate)

    if not candidate or not reference:
        return {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}

    rouge = Rouge()
    # 使用jieba分词，然后用空格连接
    reference = ' '.join(jieba.cut(reference))
    candidate = ' '.join(jieba.cut(candidate))
    try:
        scores = rouge.get_scores(candidate, reference)
        return scores[0]
    except Exception as e:  # 处理可能的其他异常情况
        return {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}


def evaluate_json(file_path, output_file):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    question_types = defaultdict(lambda: defaultdict(int))
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for item in tqdm(data, desc='评估'):
        question_type = item['type']
        difficulty = item.get('difficulty', 'unknown')  # 如果没有difficulty字段，默认为'unknown'
        question_types[question_type][difficulty] += 1

        # 评估解析
        original_analysis = item.get('original_analysis', '')
        llm_analysis = item.get('llm_analysis', '')

        if original_analysis and llm_analysis:
            metrics[question_type][difficulty]['analysis_cosine'].append(
                calculate_cosine_similarity_bert(original_analysis, llm_analysis))
            metrics[question_type][difficulty]['analysis_bleu'].append(calculate_bleu(original_analysis, llm_analysis))
            metrics[question_type][difficulty]['analysis_rouge'].append(
                calculate_rouge(original_analysis, llm_analysis))

        # 评估答案
        original_answer = item.get('original_answer', '')
        llm_answer = item.get('llm_answer', '')

        # if question_type in ['选择题', '多项选择']:
        #     original = ''.join(c for c in original_answer if c.isalpha()).upper()
        #     gemini = ''.join(c for c in llm_answer if c.isalpha()).upper()
        #     metrics[question_type][difficulty]['answer_correct'].append(original == gemini)

        if question_type == '选择题':
            # 选择题：提取原始答案和模型生成答案中的第一个大写字母进行比较
            original = next((c for c in original_answer if c.isupper()), '')
            gemini = next((c for c in llm_answer if c.isupper()), '')
            metrics[question_type][difficulty]['answer_correct'].append(original == gemini)

        elif question_type == '多项选择':
            # 多项选择：提取出所有大写字母并排序后进行比较
            original = ''.join(sorted(c for c in original_answer if c.isalpha())).upper()
            gemini = ''.join(sorted(c for c in llm_answer if c.isalpha())).upper()
            metrics[question_type][difficulty]['answer_correct'].append(original == gemini)

        elif question_type == '填空题':
            metrics[question_type][difficulty]['answer_cosine'].append(
                calculate_cosine_similarity_bert(original_answer, llm_answer))
            metrics[question_type][difficulty]['answer_bleu'].append(calculate_bleu(original_answer, llm_answer))
            metrics[question_type][difficulty]['answer_rouge'].append(calculate_rouge(original_answer, llm_answer))

    # 输出结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("统计:\n")
        for qtype, type_data in question_types.items():
            f.write(f"\n{qtype}:\n")
            for difficulty, count in type_data.items():
                f.write(f"  {difficulty}: {count}\n")

        f.write("\n评估结果:\n")
        for qtype, type_metrics in metrics.items():
            f.write(f"\n{qtype}:\n")
            for difficulty, qmetrics in type_metrics.items():
                f.write(f"  {difficulty}:\n")
                if 'answer_correct' in qmetrics:
                    correct_rate = sum(qmetrics['answer_correct']) / len(qmetrics['answer_correct'])
                    f.write(f"    答案正确率: {correct_rate:.2%}\n")

                if 'answer_cosine' in qmetrics:
                    # 计算平均余弦相似度
                    if qmetrics['answer_cosine']:
                        avg_cosine = sum(qmetrics['answer_cosine']) / len(qmetrics['answer_cosine'])
                    else:
                        avg_cosine = 0

                    # 计算平均BLEU分数
                    if qmetrics['answer_bleu']:
                        avg_bleu = sum(qmetrics['answer_bleu']) / len(qmetrics['answer_bleu'])
                    else:
                        avg_bleu = 0

                    # 计算平均ROUGE-L F1分数
                    if qmetrics['answer_rouge']:
                        avg_rouge_l = sum(rouge['rouge-l']['f'] for rouge in qmetrics['answer_rouge']) / len(
                            qmetrics['answer_rouge'])
                    else:
                        avg_rouge_l = 0
                    f.write(f"    答案评估:\n")
                    f.write(f"      平均余弦相似度: {avg_cosine:.4f}\n")
                    f.write(f"      平均BLEU分数: {avg_bleu:.4f}\n")
                    f.write(f"      平均ROUGE-L F1分数: {avg_rouge_l:.4f}\n")

                # 计算平均余弦相似度
                if qmetrics['analysis_cosine']:
                    avg_cosine = sum(qmetrics['analysis_cosine']) / len(qmetrics['analysis_cosine'])
                else:
                    avg_cosine = 0

                # 计算平均BLEU分数
                if qmetrics['analysis_bleu']:
                    avg_bleu = sum(qmetrics['analysis_bleu']) / len(qmetrics['analysis_bleu'])
                else:
                    avg_bleu = 0

                # 计算平均ROUGE-L F1分数
                if qmetrics['analysis_rouge']:
                    avg_rouge_l = sum(rouge['rouge-l']['f'] for rouge in qmetrics['analysis_rouge']) / len(
                        qmetrics['analysis_rouge'])
                else:
                    avg_rouge_l = 0
                f.write(f"    解析评估:\n")
                f.write(f"      平均余弦相似度: {avg_cosine:.4f}\n")
                f.write(f"      平均BLEU分数: {avg_bleu:.4f}\n")
                f.write(f"      平均ROUGE-L F1分数: {avg_rouge_l:.4f}\n")

    # 计算不同难度下的平均分
    difficulty_averages = defaultdict(lambda: defaultdict(float))
    difficulty_counts = defaultdict(lambda: defaultdict(int))

    for qtype, type_metrics in metrics.items():
        for difficulty, qmetrics in type_metrics.items():
            if qtype in ['选择题', '多项选择']:
                if 'answer_correct' in qmetrics:
                    difficulty_averages[difficulty]['answer_accuracy'] += sum(qmetrics['answer_correct'])
                    difficulty_counts[difficulty]['answer_accuracy'] += len(qmetrics['answer_correct'])
            elif qtype == '填空题':
                if 'answer_rouge' in qmetrics:
                    difficulty_averages[difficulty]['answer_rouge'] += sum(
                        rouge['rouge-l']['f'] for rouge in qmetrics['answer_rouge'])
                    difficulty_counts[difficulty]['answer_rouge'] += len(qmetrics['answer_rouge'])
                if 'answer_bleu' in qmetrics:
                    difficulty_averages[difficulty]['answer_bleu'] += sum(qmetrics['answer_bleu'])
                    difficulty_counts[difficulty]['answer_bleu'] += len(qmetrics['answer_bleu'])
                if 'answer_cosine' in qmetrics:
                    difficulty_averages[difficulty]['answer_cosine'] += sum(qmetrics['answer_cosine'])
                    difficulty_counts[difficulty]['answer_cosine'] += len(qmetrics['answer_cosine'])

            if 'analysis_rouge' in qmetrics:
                difficulty_averages[difficulty]['analysis_rouge'] += sum(
                    rouge['rouge-l']['f'] for rouge in qmetrics['analysis_rouge'])
                difficulty_counts[difficulty]['analysis_rouge'] += len(qmetrics['analysis_rouge'])
            if 'analysis_bleu' in qmetrics:
                difficulty_averages[difficulty]['analysis_bleu'] += sum(qmetrics['analysis_bleu'])
                difficulty_counts[difficulty]['analysis_bleu'] += len(qmetrics['analysis_bleu'])
            if 'analysis_cosine' in qmetrics:
                difficulty_averages[difficulty]['analysis_cosine'] += sum(qmetrics['analysis_cosine'])
                difficulty_counts[difficulty]['analysis_cosine'] += len(qmetrics['analysis_cosine'])

    # 计算不同题型下的平均分
    type_averages = defaultdict(lambda: defaultdict(float))
    type_counts = defaultdict(lambda: defaultdict(int))

    for qtype, type_metrics in metrics.items():
        for difficulty, qmetrics in type_metrics.items():
            if qtype in ['选择题', '多项选择']:
                if 'answer_correct' in qmetrics:
                    type_averages[qtype]['answer_accuracy'] += sum(qmetrics['answer_correct'])
                    type_counts[qtype]['answer_accuracy'] += len(qmetrics['answer_correct'])
            elif qtype == '填空题':
                if 'answer_rouge' in qmetrics:
                    type_averages[qtype]['answer_rouge'] += sum(
                        rouge['rouge-l']['f'] for rouge in qmetrics['answer_rouge'])
                    type_counts[qtype]['answer_rouge'] += len(qmetrics['answer_rouge'])
                if 'answer_bleu' in qmetrics:
                    type_averages[qtype]['answer_bleu'] += sum(qmetrics['answer_bleu'])
                    type_counts[qtype]['answer_bleu'] += len(qmetrics['answer_bleu'])
                if 'answer_cosine' in qmetrics:
                    type_averages[qtype]['answer_cosine'] += sum(qmetrics['answer_cosine'])
                    type_counts[qtype]['answer_cosine'] += len(qmetrics['answer_cosine'])

            if 'analysis_rouge' in qmetrics:
                type_averages[qtype]['analysis_rouge'] += sum(
                    rouge['rouge-l']['f'] for rouge in qmetrics['analysis_rouge'])
                type_counts[qtype]['analysis_rouge'] += len(qmetrics['analysis_rouge'])
            if 'analysis_bleu' in qmetrics:
                type_averages[qtype]['analysis_bleu'] += sum(qmetrics['analysis_bleu'])
                type_counts[qtype]['analysis_bleu'] += len(qmetrics['analysis_bleu'])
            if 'analysis_cosine' in qmetrics:
                type_averages[qtype]['analysis_cosine'] += sum(qmetrics['analysis_cosine'])
                type_counts[qtype]['analysis_cosine'] += len(qmetrics['analysis_cosine'])

    # 输出统计结果
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n不同难度下的平均分:\n")
        for difficulty, averages in difficulty_averages.items():
            f.write(f"  {difficulty}:\n")
            f.write(f"    答案分数:\n")
            if 'answer_accuracy' in averages:
                f.write(
                    f"      选择题准确率: {averages['answer_accuracy'] / difficulty_counts[difficulty]['answer_accuracy']:.4f}\n")
            if 'answer_rouge' in averages:
                f.write(
                    f"      填空题ROUGE: {averages['answer_rouge'] / difficulty_counts[difficulty]['answer_rouge']:.4f}\n")
            if 'answer_bleu' in averages:
                f.write(
                    f"      填空题BLEU: {averages['answer_bleu'] / difficulty_counts[difficulty]['answer_bleu']:.4f}\n")
            if 'answer_cosine' in averages:
                f.write(
                    f"      填空题相似度: {averages['answer_cosine'] / difficulty_counts[difficulty]['answer_cosine']:.4f}\n")
            f.write(f"    解析分数:\n")
            f.write(
                f"      ROUGE: {averages['analysis_rouge'] / difficulty_counts[difficulty]['analysis_rouge']:.4f}\n")
            f.write(
                f"      BLEU: {averages['analysis_bleu'] / difficulty_counts[difficulty]['analysis_bleu']:.4f}\n")
            f.write(
                f"      相似度: {averages['analysis_cosine'] / difficulty_counts[difficulty]['analysis_cosine']:.4f}\n")

        f.write("\n不同题型下的平均分:\n")
        for qtype, averages in type_averages.items():
            f.write(f"  {qtype}:\n")
            if qtype in ['选择题', '多项选择', '填空题']:
                f.write(f"    答案分数:\n")
                if 'answer_accuracy' in averages:
                    f.write(
                        f"      准确率: {averages['answer_accuracy'] / type_counts[qtype]['answer_accuracy']:.4f}\n")
                if 'answer_rouge' in averages:
                    f.write(f"      ROUGE: {averages['answer_rouge'] / type_counts[qtype]['answer_rouge']:.4f}\n")
                if 'answer_bleu' in averages:
                    f.write(f"      BLEU: {averages['answer_bleu'] / type_counts[qtype]['answer_bleu']:.4f}\n")
                if 'answer_cosine' in averages:
                    f.write(
                        f"      相似度: {averages['answer_cosine'] / type_counts[qtype]['answer_cosine']:.4f}\n")
            f.write(f"    解析分数:\n")
            f.write(f"      ROUGE: {averages['analysis_rouge'] / type_counts[qtype]['analysis_rouge']:.4f}\n")
            f.write(f"      BLEU: {averages['analysis_bleu'] / type_counts[qtype]['analysis_bleu']:.4f}\n")
            f.write(f"      相似度: {averages['analysis_cosine'] / type_counts[qtype]['analysis_cosine']:.4f}\n")

    print(f"评估结果已保存到 {output_file}")
