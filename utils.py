import torch
import torch.nn.functional as F
import config as CFG

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def eos_pooling(token_embeddings, attention_mask):
    last_index = []
    for i in range(attention_mask.size(0)):
        last_index.append(check_zero(attention_mask[i]))
    last_index = torch.tensor(last_index)
    return token_embeddings[range(len(last_index)), last_index]

def check_zero(mask):
    for i in range(len(mask)):
        if mask[i] == 0:
            return i-1
    return len(mask)-1

def decorate_dataset(example, d):
    example[CFG.example_name[d]] = f'Summarize sentence "{example[CFG.example_name[d]]}" in one word:'
    return example

def decorate_concepts(c):
    for i in range(len(c)):
        c[i] = f'Summarize sentence "{c[i]}" in one word:'
    return c

def cos_sim_cubed(cbl_features, target):
    cbl_features = cbl_features - torch.mean(cbl_features, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    cbl_features = F.normalize(cbl_features**3, dim=-1)
    target = F.normalize(target**3, dim=-1)

    sim = torch.sum(cbl_features*target, dim=-1)
    return sim.mean()

def normalize(x, d=-1, mean=None, std=None):
    if mean is not None and std is not None:
        x_mean = mean
        x_std = std
    else:
        x_mean = torch.mean(x, dim=d)
        x_std = torch.std(x, dim=d)
    if d == -1:
        x = x - x_mean.unsqueeze(-1)
        x = x / (x_std.unsqueeze(-1) + 1e-12)
    else:
        x = x - x_mean.unsqueeze(0)
        x = x / (x_std.unsqueeze(0) + 1e-12)
    return x, x_mean, x_std

def get_labels(n, d):
    if d == 'SetFit/sst2':
        return sst2_labels(n)
    if d == 'yelp_polarity':
        return yelpp_labels(n)
    if d == 'ag_news':
        return agnews_labels(n)
    if d == 'dbpedia_14':
        return dbpedia_labels(n)

    return None

def sst2_labels(n):
    if n < 104:
        return 0
    else:
        return 1

def yelpp_labels(n):
    if n < 124:
        return 0
    else:
        return 1

def agnews_labels(n):
    if n < 54:
        return 0
    elif n >= 54 and n < 108:
        return 1
    elif n >= 108 and n < 162:
        return 2
    else:
        return 3

def dbpedia_labels(n):
    if n < 34:
        return 0
    elif n >= 34 and n < 68:
        return 1
    elif n >= 68 and n < 102:
        return 2
    elif n >= 102 and n < 136:
        return 3
    elif n >= 136 and n < 170:
        return 4
    elif n >= 170 and n < 204:
        return 5
    elif n >= 204 and n < 238:
        return 6
    elif n >= 238 and n < 272:
        return 7
    elif n >= 272 and n < 306:
        return 8
    elif n >= 306 and n < 340:
        return 9
    elif n >= 340 and n < 374:
        return 10
    elif n >= 374 and n < 408:
        return 11
    elif n >= 408 and n < 442:
        return 12
    else:
        return 13