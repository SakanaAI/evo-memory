import os
import json
import argparse
import numpy as np

from choubun_metrics import (
    rouge_ja_score,
    qa_f1_ja_score
)
from longbench_metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    # LongBench
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    # ChouBun
    "wiki_qa": qa_f1_ja_score,
    "edinet_qa": qa_f1_ja_score,
    "corp_sec_qa": qa_f1_ja_score,
    "corp_sec_sum": rouge_ja_score

}

# This is the customized building prompt for chat models


def build_chat(lm, prompt):
    print('IN PROMPT')
    print(prompt)
    tokenizer = lm.tokenizer
    model_name = lm.model_name
    if "chatglm3" in model_name:
        tokenized_prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        raise NotImplementedError
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence "
            "assistant. The assistant gives helpful, detailed, and polite "
            "answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    else:
        print
        raise NotImplementedError

    if not ("chatglm3" in model_name):
        tokenized_prompt = tokenizer(prompt, truncation=False,
                                     return_tensors="pt")
    return tokenized_prompt


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true',
                        help="Evaluate on LongBench-E")
    return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(
        predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](
                prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](
                prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def get_all_scores(task, predictions, answers, all_classes):
    all_scores = []
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if task in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[task](prediction, ground_truth,
                                                    all_classes=all_classes))
        all_scores.append(score)
    return all_scores


def get_score(task, predictions, answers, all_classes):
    total_score = 0.
    all_scores = []

    # Instantiate tokenizer for ChouBun tasks
    if task in ["wiki_qa", "edinet_qa", "corp_sec_qa", "corp_sec_sum"]:
        from fugashi import Tagger
        tokenizer = Tagger('-Owakati')
    else:
        tokenizer = None

    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if task in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            cur_score = dataset2metric[task](
                prediction,
                ground_truth,
                tokenizer=tokenizer,
                all_classes=all_classes
            )
            score = max(score, cur_score)
        all_scores.append(score)
        total_score += score
    mean_score = 100 * total_score / len(predictions)
    return mean_score, all_scores


if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    if args.e:
        path = f"pred_e/{args.model}/"
    else:
        path = f"pred/{args.model}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions,
                             answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    if args.e:
        out_path = f"pred_e/{args.model}/result.json"
    else:
        out_path = f"pred/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
