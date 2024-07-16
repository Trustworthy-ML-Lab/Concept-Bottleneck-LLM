import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from utils import normalize, get_labels, eos_pooling

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--cbl_path", type=str, default="mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt")
parser.add_argument('--sparse', action=argparse.BooleanOptionalAction)
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.texts.items()}
        return t

    def __len__(self):
        return len(self.texts['input_ids'])


def build_loaders(texts, mode):
    dataset = ClassificationDataset(texts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    acs = args.cbl_path.split("/")[0]
    dataset = args.cbl_path.split("/")[1] if 'sst2' not in args.cbl_path.split("/")[1] else args.cbl_path.split("/")[1].replace('_', '/')
    backbone = args.cbl_path.split("/")[2]
    cbl_name = args.cbl_path.split("/")[-1]

    print("loading data...")
    test_dataset = load_dataset(dataset, split='test')
    print("test data len: ", len(test_dataset))
    print("tokenizing...")
    if 'roberta' in backbone:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif 'gpt2' in backbone:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("backbone should be roberta or gpt2")

    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True,
                            max_length=args.max_length), batched=True, batch_size=len(test_dataset))
    encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]

    print("creating loader...")
    test_loader = build_loaders(encoded_test_dataset, mode="test")


    concept_set = CFG.concept_set[dataset]
    if 'roberta' in backbone:
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
        else:
            print("preparing backbone(roberta)+CBL...")
            backbone_cbl = RobertaCBL(len(concept_set), args.dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            backbone_cbl.eval()
    elif 'gpt2' in backbone:
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
        else:
            print("preparing backbone(gpt2)+CBL...")
            backbone_cbl = GPT2CBL(len(concept_set), args.dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            backbone_cbl.eval()
    else:
        raise Exception("backbone should be roberta or gpt2")

    print("get concept features...")
    FL_test_features = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                test_features = preLM(input_ids=batch["input_ids"],
                                      attention_mask=batch["attention_mask"]).last_hidden_state
                if args.backbone == 'roberta':
                    test_features = test_features[:, 0, :]
                elif args.backbone == 'gpt2':
                    test_features = eos_pooling(test_features, batch["attention_mask"])
                else:
                    raise Exception("backbone should be roberta or gpt2")
                test_features = cbl(test_features)
            else:
                test_features = backbone_cbl(batch)
            FL_test_features.append(test_features)
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()

    prefix = "./" + acs + "/" + dataset.replace('/', '_') + "/" + backbone + "/"
    model_name = cbl_name[3:]
    train_mean = torch.load(prefix + 'train_mean' + model_name)
    train_std = torch.load(prefix + 'train_std' + model_name)

    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    label = encoded_test_dataset["label"]

    final = torch.nn.Linear(in_features=len(concept_set), out_features=CFG.class_num[dataset])
    W_g_path = prefix + "W_g"
    b_g_path = prefix + "b_g"
    if args.sparse:
        W_g_path += "_sparse"
        b_g_path += "_sparse"
    W_g_path += model_name
    b_g_path += model_name
    W_g = torch.load(W_g_path)
    b_g = torch.load(b_g_path)
    final.load_state_dict({"weight": W_g, "bias": b_g})
    with torch.torch.no_grad():
        pred = np.argmax(final(test_c).detach().numpy(), axis=-1)
    correct_indices = np.where(pred == label)[0]
    mispred_indices = np.where(pred != label)[0]


    m = test_c.unsqueeze(1) * W_g.unsqueeze(0)
    print(m.size())

    error_rate = []
    for i in correct_indices:
        error = 0
        total = 0
        value, c = m[i][label[i]].topk(5)
        for j in range(len(c)):
            if value[j] > 0.0:
                total += 1
                if get_labels(c[j], dataset) != label[i]:
                    error += 1
        if total != 0:
            error_rate.append(error/total)

    print("avg error rate:", sum(error_rate)/len(error_rate))

    with open(prefix + 'Concept_contribution' + W_g_path.split("/")[-1][3:-3] + '.txt', 'w') as f:
        for i in range(m.size(0)):
            f.write(test_dataset[CFG.example_name[dataset]][i])
            f.write('\n')
            c = m[i][label[i]].topk(5)[1]
            n = m[i][label[i]].topk(5)[0]
            for j in range(len(c)):
                if n[j] > 0.0 and i not in mispred_indices:
                    f.write(CFG.concept_set[dataset][c[j]])
                    f.write('\n')
                else:
                    f.write('\n')
            for j in range(len(c)):
                if n[j] > 0.0 and i not in mispred_indices:
                    f.write("{:.4f}".format(float(n[j])))
                    f.write('\n')
                else:
                    f.write('\n')
            if i not in mispred_indices:
                f.write(str(pred[i]))
            else:
                f.write("incorrect")
            f.write('\n')
            f.write('\n')