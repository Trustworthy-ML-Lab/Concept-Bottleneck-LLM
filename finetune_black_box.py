import argparse
import os
import torch
from torch import nn
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import config as CFG
from modules import Roberta_Baseline, GPT2_Baseline, MLP
from utils import eos_pooling

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--backbone", type=str, default="roberta", help="roberta or gpt2")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument('--tune_mlp_only', action=argparse.BooleanOptionalAction)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--projection_dim", type=int, default=256)
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True if mode == "train" else False)
    return dataloader

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    print("loading data...")
    train_dataset = load_dataset(args.dataset, split='train')
    if args.dataset == 'SetFit/sst2':
        val_dataset = load_dataset(args.dataset, split='validation')

    print("training data len: ", len(train_dataset))
    if args.dataset == 'SetFit/sst2':
        print("val data len: ", len(val_dataset))
    print("tokenizing...")

    if args.backbone == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif args.backbone == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("backbone should be roberta or gpt2")
    
    encoded_train_dataset = train_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True,
        batch_size=len(train_dataset))
    encoded_train_dataset = encoded_train_dataset.remove_columns([CFG.example_name[args.dataset]])
    if args.dataset == 'SetFit/sst2':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['label_text'])
    if args.dataset == 'dbpedia_14':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['title'])
    encoded_train_dataset = encoded_train_dataset[:len(encoded_train_dataset)]

    if args.dataset == 'SetFit/sst2':
        encoded_val_dataset = val_dataset.map(
            lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True,
            batch_size=len(val_dataset))
        encoded_val_dataset = encoded_val_dataset.remove_columns([CFG.example_name[args.dataset]])
        if args.dataset == 'SetFit/sst2':
            encoded_val_dataset = encoded_val_dataset.remove_columns(['label_text'])
        if args.dataset == 'dbpedia_14':
            encoded_val_dataset = encoded_val_dataset.remove_columns(['title'])
        encoded_val_dataset = encoded_val_dataset[:len(encoded_val_dataset)]

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, mode="train")
    if args.dataset == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, mode="valid")

    if args.backbone == 'roberta':
        if args.tune_mlp_only:
            print("preparing MLP...")
            mlp = MLP(CFG.class_num[args.dataset], args.projection_dim, args.dropout).to(device)
            mlp.train()
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
            optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-4)
        else:
            print("preparing roberta-base...")
            LM = Roberta_Baseline(CFG.class_num[args.dataset], args.projection_dim, args.dropout).to(device)
            LM.train()
            optimizer = torch.optim.Adam(LM.parameters(), lr=1e-5)
    elif args.backbone == 'gpt2':
        if args.tune_mlp_only:
            print("preparing MLP...")
            mlp = MLP(CFG.class_num[args.dataset], args.projection_dim, args.dropout).to(device)
            mlp.train()
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
            optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-4)
        else:
            print("preparing gpt2...")
            LM = GPT2_Baseline(CFG.class_num[args.dataset], args.projection_dim, args.dropout).to(device)
            LM.train()
            optimizer = torch.optim.Adam(LM.parameters(), lr=1e-5)
    else:
        raise Exception("backbone should be roberta or gpt2")

    print("start training...")
    best_loss = float('inf')
    d_name = args.dataset.replace('/', '_')
    prefix = "./baseline_models/"
    if args.backbone == 'roberta':
        prefix += "roberta"
    elif args.backbone == 'gpt2':
        prefix += "gpt2"
    else:
        raise Exception("backbone should be roberta or gpt2")
    prefix += "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    if args.tune_mlp_only:
        epoch = CFG.finetune_mlp_epoch[args.dataset]
    else:
        epoch = CFG.finetune_epoch[args.dataset]
    for e in range(epoch):
        print("Epoch ", e+1, ":")
        if args.tune_mlp_only:
            mlp.train()
        else:
            LM.train()
        training_loss = []
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if args.tune_mlp_only:
                with torch.no_grad():
                    LM_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                    if args.backbone == 'roberta':
                        LM_features = LM_features[:, 0, :]
                    elif args.backbone == 'gpt2':
                        LM_features = eos_pooling(LM_features, batch["attention_mask"])
                    else:
                        raise Exception("backbone should be roberta or gpt2")
                pred = mlp(LM_features)
            else:
                pred = LM(batch)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss(reduction='mean')(pred, batch["label"])
            loss.backward()
            optimizer.step()
            print("batch ", str(i), " loss: ", loss.detach().cpu().numpy(), end="\r")
            training_loss.append(loss.detach().cpu().numpy())
        avg_training_loss = sum(training_loss) / len(training_loss)
        print("training loss: ", avg_training_loss)

        if args.dataset == 'SetFit/sst2':
            if args.tune_mlp_only:
                mlp.eval()
            else:
                LM.eval()
            val_loss = []
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    if args.tune_mlp_only:
                        LM_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                        if args.backbone == 'roberta':
                            LM_features = LM_features[:, 0, :]
                        elif args.backbone == 'gpt2':
                            LM_features = eos_pooling(LM_features, batch["attention_mask"])
                        else:
                            raise Exception("backbone should be roberta or gpt2")
                        pred = mlp(LM_features)
                    else:
                        pred = LM(batch)
                    loss = nn.CrossEntropyLoss(reduction='mean')(pred, batch["label"])
                    val_loss.append(loss.detach().cpu().numpy())
            avg_val_loss = sum(val_loss)/len(val_loss)
            print("val loss: ", avg_val_loss)
            if avg_val_loss < best_loss:
                print("save model")
                best_loss = avg_val_loss
                if args.tune_mlp_only:
                    torch.save(mlp.state_dict(), prefix + "mlp_finetuned_" + d_name + ".pt")
                else:
                    torch.save(LM.state_dict(), prefix + "backbone_finetuned_" + d_name + ".pt")
        else:
            print("save model")
            if args.tune_mlp_only:
                torch.save(mlp.state_dict(), prefix + "mlp_finetuned_" + d_name + ".pt")
            else:
                torch.save(LM.state_dict(), prefix + "backbone_finetuned_" + d_name + ".pt")