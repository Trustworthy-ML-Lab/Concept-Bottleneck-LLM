import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import config as CFG
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers.utils import logging
logging.set_verbosity_error()

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")

args = parser.parse_args()

print("loading data...")
train_dataset = load_dataset(args.dataset, split='train')
if args.dataset == 'SetFit/sst2':
    val_dataset = load_dataset(args.dataset, split='validation')
print("training data len: ", len(train_dataset))
if args.dataset == 'SetFit/sst2':
    print("val data len: ", len(val_dataset))


d_list = []
for i in range(CFG.class_num[args.dataset]):
    d_list.append(train_dataset.filter(lambda e: e['label'] == i).select(range(1000//CFG.class_num[args.dataset])))
train_dataset = concatenate_datasets(d_list)
if args.dataset == 'SetFit/sst2':
    d_list = []
    for i in range(CFG.class_num[args.dataset]):
        d_list.append(val_dataset.filter(lambda e: e['label'] == i).select(range(80//CFG.class_num[args.dataset])))
    val_dataset = concatenate_datasets(d_list)

print("training labeled data len: ", len(train_dataset))
if args.dataset == 'SetFit/sst2':
    print("val labeled data len: ", len(val_dataset))

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

concept_set = CFG.concept_set[args.dataset]

instr = "You will be given a yes/no question, please answer with only yes or no."
temp = "According to the movie review: '{}', the movie has '{}'. yes or no?"
few_shot_examples = ""
if args.dataset == 'SetFit/sst2':
    s1 = "no movement , no yuks , not much of anything."
    c1 = "Flat or one-dimensional characters."
    a1 = "yes"
    s2 = "we never really feel involved with the story , as all of its ideas remain just that : abstract ideas."
    c2 = "Engaging music score."
    a2 = "no"
    s3 = "the movie exists for its soccer action and its fine acting."
    c3 = "Well-executed action sequences"
    a3 = "yes"
    s4 = "what might have been a predictably heartwarming tale is suffused with complexity."
    c4 = "Complex and multi-dimensional villains."
    a4 = "no"
    few_shot_examples += (temp.format(s1, c1) + " " + a1 + "\n")
    few_shot_examples += (temp.format(s2, c2) + " " + a2 + "\n")
    few_shot_examples += (temp.format(s3, c3) + " " + a3 + "\n")
    few_shot_examples += (temp.format(s4, c4) + " " + a4 + "\n")

print("generating train labels")
train_labels = []
for i in range(len(train_dataset)):
    print("sample ", str(i), end="\r")
    sample = train_dataset[CFG.example_name[args.dataset]][i]

    labels = []
    for j in range(len(concept_set)):
        concept = concept_set[j]
        messages = [
            {"role": "system", "content": "You are a chatbot who always solve the given problem exactly!"},
            {"role": "user", "content": instr + "\nExamples:\n" + few_shot_examples + "Questions:\n" + temp.format(sample, concept)},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        prompt_length = input_ids.shape[1]

        for k in range(10):
            outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
            answer = tokenizer.decode(outputs[0][prompt_length:]).replace('<|eot_id|>', '')
            if answer == "yes" or answer == "Yes" or answer == "no" or answer == "No":
                break
        if answer == "yes" or answer == "Yes":
            labels.append(1)
        elif answer == "no" or answer == "No":
            labels.append(0)
        else:
            labels.append(0)

    train_labels.append(labels)

print("generating val labels")
if args.dataset == 'SetFit/sst2':
    val_labels = []
    for i in range(len(val_dataset)):
        print("sample ", str(i), end="\r")
        sample = val_dataset[CFG.example_name[args.dataset]][i]

        labels = []
        for j in range(len(concept_set)):
            concept = concept_set[j]
            messages = [
                {"role": "system", "content": "You are a chatbot who always solve the given problem exactly!"},
                {"role": "user",
                 "content": instr + "\nExamples:\n" + few_shot_examples + "Questions:\n" + temp.format(sample,
                                                                                                       concept)},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            prompt_length = input_ids.shape[1]

            for k in range(10):
                outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=terminators, do_sample=True,
                                         temperature=0.6, top_p=0.9)
                answer = tokenizer.decode(outputs[0][prompt_length:]).replace('<|eot_id|>', '')
                if answer == "yes" or answer == "Yes" or answer == "no" or answer == "No":
                    break
            if answer == "yes" or answer == "Yes":
                labels.append(1)
            elif answer == "no" or answer == "No":
                labels.append(0)
            else:
                labels.append(0)

        val_labels.append(labels)

d_name = args.dataset.replace('/', '_')
prefix = "./llm_labeling/"
prefix += d_name
prefix += "/"
if not os.path.exists(prefix):
    os.makedirs(prefix)
np.save(prefix + "concept_labels_train.npy", np.asarray(train_labels))
if args.dataset == 'SetFit/sst2':
    np.save(prefix + "concept_labels_val.npy", np.asarray(val_labels))