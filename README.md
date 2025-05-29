# EduQS

[**🤗 Dataset**](https://huggingface.co/datasets/chaosY/EduQS) | [**Github**](https://github.com/JNU-xcj/EduQS/tree/main)

This repository provides evaluation code and dataset details for the paper:

> **EduQS: A Comprehensive Benchmark for Evaluating Multi-Modal Large Language Models on Chinese Education Question Solving**


## 🔍 Introduction

**EduQS** is a large-scale Chinese multimodal educational dataset containing **52,056 questions**, spanning **7 question types** and **5 difficulty levels** across **6 subjects**: *Mathematics, Physics, Chemistry, Biology, Geography,* and *History*.

Each question includes:
- Multimodal **question description** (text + image)
- **Solution information** (difficulty, answer, full explanation)
- **Side information** (structured knowledge points and similar questions)

These features support fine-grained evaluation of MM-LLMs on **reasoning**, **in-context learning**, and **generalization**.
<p align="center">
  <img src="assets/Biology.png" alt="Biology Example" height="300px" />
  <img src="assets/Chemistry.png" alt="Chemistry Example" height="300px" />
  <img src="assets/Math.png" alt="Math Example" height="300px" />
</p>

## 📈 Data Distribution

| Difficulty Distribution | Question Type Distribution |
|-------------------------|----------------------------|
| ![](assets/Difficulty%20analysis.png) | ![](assets/Type%20analysis.png) |

## 📦 Dataset Format

Example (`.jsonl` format):

```json
{
  "subject": "biology",
  "id": "biology_799",
  "type": "fill-in-the-blank",
  "grade": "high",
  "difficulty": "hard",
  "question_info": "题干...",
  "solution_info": "解析...",
  "answer": "标准答案...",
  "side_information": [
    "辅助知识点1...",
    "辅助知识点2..."
  ],
  "image": {
    "path": "val/images/high_biology_799.png"
  }
}
```


## 📊 Evaluation

We evaluate MM-LLMs on answer accuracy and reasoning quality using both human-annotated judgments and automated metrics such as BLEU and ROUGE.

Answering and reasoning accuracy of different MM-LLMs across various question types.  
**Bold**: best result, _Underlined_: second-best.

| Model         | SCQ   | MCQ   | FBQ   | BLEU-SCQ | BLEU-MCQ | BLEU-FBQ | BLEU-CaQ | BLEU-MPQ | BLEU-DiQ | BLEU-CoQ | ROUGE-SCQ | ROUGE-MCQ | ROUGE-FBQ | ROUGE-CaQ | ROUGE-MPQ | ROUGE-DiQ | ROUGE-CoQ | Avg    |
|---------------|-------|-------|-------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|--------|
| GPT-4o        | 57.96 | 6.31  | 33.45 | 5.94      | 2.94      | 6.56      | 4.80      | 3.49      | 3.48      | 3.67      | 29.54       | 21.32       | 30.51       | 30.82       | 27.39       | 25.52       | 23.30       | 18.65  |
| GPT-4o-mini   | 45.67 | 2.72  | 25.10 | 3.82      | 1.04      | 3.13      | 2.41      | 2.45      | 1.37      | 2.10      | 25.31       | 18.86       | 24.16       | 23.20       | 21.63       | 16.31       | 19.75       | 14.06  |
| GPT-4V        | 49.03 | 3.97  | 32.39 | 3.98      | 1.63      | 3.62      | 2.98      | 2.27      | 1.58      | 2.34      | 25.33       | 19.56       | 24.31       | 25.39       | 22.15       | 17.51       | 20.09       | 15.18  |
| Gemini-Pro    | 52.56 | 5.42  | 33.67 | 7.11      | 2.89      | 5.40      | 4.37      | 3.88      | _4.20_    | 3.31      | **32.64**   | 24.47       | 30.48       | 30.78       | 29.21       | _26.06_     | 23.00       | _18.79_ |
| Gemini-Flash  | 47.67 | 4.88  | 26.67 | 6.81      | 3.24      | _6.69_    | 5.52      | 4.80      | 3.37      | 4.04      | 31.41       | _24.53_     | 30.98       | 31.78       | _29.69_     | 24.52       | 23.19       | 18.22  |
| Qwen-Plus     | 52.91 | 5.20  | _35.10_ | 3.12    | 0.28      | 0.93      | 1.06      | 0.86      | 0.91      | 0.71      | 21.16       | 15.66       | 13.14       | 9.70        | 12.01       | 11.05       | 11.03       | 11.46  |
| Qwen-Max      | _59.64_ | _8.20_ | 33.12 | 7.00   | _3.26_    | 5.60      | 4.29      | 3.06      | 2.46      | 2.60      | 29.81       | **24.53**   | 26.86       | 26.25       | 24.11       | 20.72       | 19.03       | 17.68  |
| **Qwen2-VL**  | **66.74** | **10.63** | **38.91** | **9.63** | **3.89** | 5.92      | **10.41** | **5.38** | **7.25** | **9.86** | 26.36       | 24.30       | **35.94**   | **39.21**   | **30.89**   | **31.91**   | **33.06**   | **22.96** |
| mPLUG-Owl     | 5.67  | 2.39  | 3.46  | 0.22      | 0.15      | 0.12      | 0.09      | 0.27      | 0.13      | 0.34      | 5.54        | 5.45        | 1.98        | 2.21        | 4.41        | 1.08        | 5.72        | 2.31   |
| Intern-X      | 42.52 | 2.67  | 29.50 | _7.95_    | 1.60      | **7.40**  | _6.75_    | 3.33      | 3.19      | 5.05      | _31.80_     | 23.34       | 31.46       | 32.43       | 26.68       | 25.10       | _26.68_     | 18.09  |
| IXC-2.5       | 44.28 | 3.95  | 31.49 | 6.15      | 1.71      | 2.73      | 8.91      | _5.34_    | 2.78      | _9.63_    | 28.64       | 19.36       | _31.63_     | _36.62_     | 29.07       | 24.19       | 25.45       | 18.35  |
| Deep-VL       | 32.91 | 4.35  | 20.61 | 6.13      | 0.96      | 4.05      | 4.61      | 2.09      | 1.04      | 3.62      | 26.27       | 22.05       | 22.44       | 27.57       | 19.88       | 20.76       | 21.61       | 14.17  |
| Yi-VL         | 45.10 | 5.81  | 21.67 | 6.61      | 1.89      | 6.52      | 5.17      | 3.34      | 3.46      | 3.55      | 29.38       | 22.21       | 30.08       | 30.39       | 27.25       | 25.37       | 23.17       | 17.12  |
| CogVLM        | 5.11  | 1.45  | 2.90  | 0.17      | 0.08      | 0.10      | 0.06      | 0.32      | 0.05      | 0.25      | 5.47        | 5.45        | 1.74        | 1.60        | 4.43        | 1.16        | 4.74        | 2.06   |

## 🔬 Learning Ability Evaluation (Ablation Study)

To assess the comprehensive learning capabilities of MM-LLMs, we conduct ablation experiments by progressively adding two types of side information: **knowledge details** and **similar questions**.

We select:
- 100 questions with up to **9 knowledge annotations**
- 100 questions with up to **5 similar questions**

We evaluate the models using both **answering accuracy** and **reasoning accuracy** (BLEU and ROUGE). As shown in the figures below:

<p align="center">
  <img src="assets/Knowledge ablation.png" width="100%"/>
  <br>
  <em>Effect of increasing the number of knowledge details</em>
</p>

<p align="center">
  <img src="assets/Similar ablation.png" width="100%"/>
  <br>
  <em>Effect of increasing the number of similar questions</em>
</p>

## 🔍 Case Study

To further analyze the limitations of MM-LLMs, we present a case study of **GPT-4o** on multidisciplinary questions from the EduQS dataset, covering six core subjects: *Mathematics, Physics, Chemistry, Biology, History,* and *Geography*.

<p align="center">
  <img src="assets/Case study All.png" width="100%"/>
  <br>
  <em>Figure 7: Case studies of GPT-4o across six subjects. Mistakes include conceptual confusion, image misinterpretation, and logical reasoning errors.</em>
</p>

As illustrated in **Figure 7**, GPT-4o’s responses exhibit various types of reasoning flaws:
- 🧩 **Question misunderstanding**
- 🧠 **Conceptual confusion**
- 🖼️ **Image misinterpretation**
- 🔍 **Misleading reasoning**
