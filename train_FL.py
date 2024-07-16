import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from utils import normalize, eos_pooling

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--cbl_path", type=str, default="mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--saga_epoch", type=int, default=500)
parser.add_argument("--saga_batch_size", type=int, default=256)

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
    train_dataset = load_dataset(dataset, split='train')
    if dataset == 'SetFit/sst2':
        val_dataset = load_dataset(dataset, split='validation')
    test_dataset = load_dataset(dataset, split='test')
    print("training data len: ", len(train_dataset))
    if dataset == 'SetFit/sst2':
        print("val data len: ", len(val_dataset))
    print("test data len: ", len(test_dataset))
    print("tokenizing...")

    if 'roberta' in backbone:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif 'gpt2' in backbone:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("backbone should be roberta or gpt2")


    encoded_train_dataset = train_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True, batch_size=len(train_dataset))
    encoded_train_dataset = encoded_train_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['title'])
    encoded_train_dataset = encoded_train_dataset[:len(encoded_train_dataset)]

    if dataset == 'SetFit/sst2':
        encoded_val_dataset = val_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True, batch_size=len(val_dataset))
        encoded_val_dataset = encoded_val_dataset.remove_columns([CFG.example_name[dataset]])
        if dataset == 'SetFit/sst2':
            encoded_val_dataset = encoded_val_dataset.remove_columns(['label_text'])
        if dataset == 'dbpedia_14':
            encoded_val_dataset = encoded_val_dataset.remove_columns(['title'])
        encoded_val_dataset = encoded_val_dataset[:len(encoded_val_dataset)]

    encoded_test_dataset = test_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True, batch_size=len(test_dataset))
    encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, mode="valid")
    if dataset == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, mode="valid")
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
    FL_train_features = []
    if dataset == 'SetFit/sst2':
        FL_val_features = []
    FL_test_features = []
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                train_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                if args.backbone == 'roberta':
                    train_features = train_features[:, 0, :]
                elif args.backbone == 'gpt2':
                    train_features = eos_pooling(train_features, batch["attention_mask"])
                else:
                    raise Exception("backbone should be roberta or gpt2")
                train_features = cbl(train_features)
            else:
                train_features = backbone_cbl(batch)
            FL_train_features.append(train_features)

    if dataset == 'SetFit/sst2':
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                if 'no_backbone' in cbl_name:
                    val_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                    if args.backbone == 'roberta':
                        val_features = val_features[:, 0, :]
                    elif args.backbone == 'gpt2':
                        val_features = eos_pooling(val_features, batch["attention_mask"])
                    else:
                        raise Exception("backbone should be roberta or gpt2")
                    val_features = cbl(val_features)
                else:
                    val_features = backbone_cbl(batch)
                FL_val_features.append(val_features)

    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                test_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
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

    train_c = torch.cat(FL_train_features, dim=0).detach().cpu()
    if dataset == 'SetFit/sst2':
        val_c = torch.cat(FL_val_features, dim=0).detach().cpu()
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()

    train_c, train_mean, train_std = normalize(train_c, d=0)
    train_c = F.relu(train_c)

    prefix = "./" + acs + "/" + dataset.replace('/', '_') + "/" + backbone + "/"
    model_name = cbl_name[3:]
    torch.save(train_mean, prefix + 'train_mean' + model_name)
    torch.save(train_std, prefix + 'train_std' + model_name)

    if dataset == 'SetFit/sst2':
        val_c, _, _ = normalize(val_c, d=0, mean=train_mean, std=train_std)
        val_c = F.relu(val_c)

    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)


    train_y = torch.LongTensor(encoded_train_dataset["label"])
    indexed_train_ds = IndexedTensorDataset(train_c, train_y)

    if dataset == 'SetFit/sst2':
        val_y = torch.LongTensor(encoded_val_dataset["label"])
        val_ds = TensorDataset(val_c, val_y)

    test_y = torch.LongTensor(encoded_test_dataset["label"])
    test_ds = TensorDataset(test_c, test_y)

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    if dataset == 'SetFit/sst2':
        val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.saga_batch_size, shuffle=False)

    print("dim of concept features: ", train_c.shape[1])
    linear = torch.nn.Linear(train_c.shape[1], CFG.class_num[dataset])
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = 0.0007

    print("training final layer...")
    if dataset == 'SetFit/sst2':
        output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, k=10,
                               val_loader=val_loader, test_loader=test_loader, do_zero=True,
                               n_classes=CFG.class_num[dataset])
    else:
        output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, k=10,
                               test_loader=test_loader, do_zero=True,
                               n_classes=CFG.class_num[dataset])

    print("save weights with test acc:", output_proj['path'][-1]['metrics']['acc_test'])
    W_g = output_proj['path'][-1]['weight']
    b_g = output_proj['path'][-1]['bias']

    if dataset == 'SetFit/sst2':
        output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, epsilon=1, k=1,
                               val_loader=val_loader, test_loader=test_loader, do_zero=False,
                               n_classes=CFG.class_num[dataset], metadata=metadata, n_ex=train_c.shape[0])
    else:
        output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, epsilon=1, k=1,
                               test_loader=test_loader, do_zero=False,
                               n_classes=CFG.class_num[dataset], metadata=metadata, n_ex=train_c.shape[0])
    print("save the sparse weights with test acc:", output_proj['path'][0]['metrics']['acc_test'])
    W_g_sparse = output_proj['path'][0]['weight']
    b_g_sparse = output_proj['path'][0]['bias']

    torch.save(W_g, prefix + 'W_g' + model_name)
    torch.save(b_g, prefix + 'b_g' + model_name)
    torch.save(W_g_sparse, prefix + 'W_g_sparse' + model_name)
    torch.save(b_g_sparse, prefix + 'b_g_sparse' + model_name)

