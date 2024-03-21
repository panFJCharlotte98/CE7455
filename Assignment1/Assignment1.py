import torch
import numpy as np
import random
from torchtext import data, datasets
from torchtext.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD, Adam, AdamW, Adagrad
from gensim.models import KeyedVectors
import gensim.downloader
import time
import itertools
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
import torch.nn.functional as F
from einops import rearrange
import logging
logger = logging.getLogger(__name__)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
SEED = 1234

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

def get_optimizer_grouped_parameters(model, weight_decay, do_group=False):
    if do_group:
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        decay_params, decay_names, no_decay_params, no_decay_names = [], [], [], []
        for n, p in model.named_parameters():
            if (n in decay_parameters and p.requires_grad):
                decay_params.append(p)
                decay_names.append(n)
            elif (n not in decay_parameters and p.requires_grad):
                no_decay_params.append(p)
                no_decay_names.append(n)
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay, "param_names":decay_names},
            {"params": no_decay_params, "weight_decay": 0.0, "param_names":no_decay_names},
        ]
    else:
        params, param_names = [],[]
        for n, p in model.named_parameters():
            params.append(p)
            param_names.append(n)
        optimizer_grouped_parameters = [{"params": params, "weight_decay": weight_decay, "param_names":param_names}]
    return optimizer_grouped_parameters


def load_data():
    # For tokenization
    TEXT = data.Field(tokenize='spacy',
                    tokenizer_language='en_core_web_sm',
                    include_lengths=True)
    # For multi-class classification labels
    LABEL = data.LabelField()
    # Load the TREC dataset
    train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)
    # Assuming train_data is already defined
    train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    TEXT.build_vocab(train_data, max_size=10000)
    LABEL.build_vocab(train_data)
    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
    
    return train_data, valid_data, test_data, TEXT, LABEL

class RNN(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        output_dim, 
        device,
        pack_seq, 
        output_mode=None,
        key_value_proj_dim=10,
        rnn='RNN',
        pretrained_emb=None,
        add_component=False,
        multi_output_layers=False,
        padding_idx=1
    ):

        super().__init__()
        self.device = device
        self.padding_idx = padding_idx
        self.pack_seq = pack_seq
  
        if pretrained_emb is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=False, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # print(f"self.embedding.weight.requires_grad: {self.embedding.weight.requires_grad}")
        
        if rnn == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim) 
        elif rnn == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        elif rnn == 'bi-RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, bidirectional=True)
        elif rnn == '2l-RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=2)
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_dim)
        
        self.rnn_type = rnn
        self.output_mode = output_mode
        if self.output_mode == 'attention':
            self.key_value_proj_dim = key_value_proj_dim
            self.n_heads = int(hidden_dim / self.key_value_proj_dim)
            self.inner_dim = self.n_heads * self.key_value_proj_dim
            self.q = nn.Linear(hidden_dim, self.inner_dim, bias=False)
            self.k = nn.Linear(hidden_dim, self.inner_dim, bias=False)
            self.v = nn.Linear(hidden_dim, self.inner_dim, bias=False)
            self.multihead_attn = nn.MultiheadAttention(hidden_dim, self.n_heads)
        
        self.add_component = add_component
        if add_component:
            self.textcnn = TextCNN(hidden_dim, output_dim, multi_output_layers=multi_output_layers)
        else:
            if rnn == 'bi-RNN':
                self.fc = nn.Linear(2*hidden_dim, output_dim)
            else:
                self.fc = nn.Linear(hidden_dim, output_dim)
        #print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters')
        #self.last_emb = self.embedding.weight.clone().detach().to(device)
    
    def forward(self, text, text_lengths, is_training):
        #text = [sent len, batch size]
        if self.add_component:
            if text.size()[0] < 5:
                paddings = self.padding_idx * torch.ones(5-text.size()[0], text.size()[1], dtype=text.dtype).to(text.device)
                text = torch.cat([text, paddings], dim=0)
                text_lengths = 5 * torch.ones_like(text_lengths, dtype=text_lengths.dtype).to(text_lengths.device)
        embedded = self.embedding(text)
        
        #embedded = [sent len, batch size, emb dim]
        if self.pack_seq:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu())
            #PackedSequence: data=[sum(text_lengths)], batch_sizes=[max(text_lengths)]
        if self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded)
            # output = [src length, batch size, hidden dim * n directions]
            # hidden = [n layers * n directions, batch size, hidden dim]
            # cell = [n layers * n directions, batch size, hidden dim]
            # output are always from the top hidden layer
        else:
            output, hidden = self.rnn(embedded)
            # if pack_seq: output.data = [sum(text_lengths), hid_dim]
            # else: output = [sent len, batch size, hid dim * n_directions]
            # hidden = [n_layers * n_directions, batch size, hid dim]
            if self.rnn_type == 'bi-RNN':
                hidden = rearrange(hidden, "a b c -> 1 b (c a)")
            if self.rnn_type == '2l-RNN':
                hidden = hidden[-1, None, ...]
        
        if self.output_mode is not None:
            unpacked_output, unpacked_lens = torch.nn.utils.rnn.pad_packed_sequence(output) 
            # unpacked_output: [sent_len, batch_size, hid_dim * n_directions]
             
            # aggregation of step hiddens
            if self.output_mode == 'max':
                # 1. max pooling
                hidden = torch.max(unpacked_output, dim=0, keepdim=True).values
            if self.output_mode == 'avg':
                # 2. avg pooling
                hidden = torch.mean(unpacked_output, dim=0, keepdim=True)
            if self.output_mode == 'attention':
                query = self.q(hidden)
                key = self.k(unpacked_output)
                value = self.v(unpacked_output)
                attn_output, attn_weights = self.multihead_attn(query, key, value)
                hidden = attn_output
        #assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        if self.add_component:
            unpacked_output, unpacked_lens = torch.nn.utils.rnn.pad_packed_sequence(output)
            unpacked_output = rearrange(unpacked_output, "l b h -> b l h")
            logits = self.textcnn(unpacked_output, is_training)
        else:
            logits = self.fc(hidden.squeeze(0))
        return logits

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(
    model, 
    iterator, 
    optimizer, 
    criterion, 
    regularization=None, 
    reg_only_weights=None,
    l1_lambda=0.001, 
    l2_lambda=0.01,
):
    def get_regularization_terms(model, regularization, reg_only_weights, l1_lambda, l2_lambda):
        # Only apply regularization to weights
        l1_norms, l2_norms = [], []
        if reg_only_weights:
            for name, p in model.named_parameters(): 
                if 'textcnn' not in name:
                    if 'bias' not in name:
                        l1_norms.append(torch.linalg.norm(p,1))
                        l2_norms.append(p.pow(2.0).sum())
        else:
            for name, p in model.named_parameters():
                if 'textcnn' not in name:
                    l1_norms.append(torch.linalg.norm(p,1))
                    l2_norms.append(p.pow(2.0).sum())
        if regularization == 'l1':
            return l1_lambda * sum(l1_norms)
        if regularization == 'l2':
            return l2_lambda * sum(l2_norms) / 2.
        if regularization == 'both':
            return l1_lambda * sum(l1_norms) + l2_lambda * sum(l2_norms) / 2.
        
    epoch_loss = 0
    total_correct = 0
    total_instances = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()
        
        # print(batch)
        # exit()
        text, text_lengths = batch.text

        # Assuming your model's forward method automatically handles padding, then no need to pack sequence here
        predictions = model(text, text_lengths, is_training=True)

        loss = criterion(predictions, batch.label)
        
        if (regularization is not None) and (regularization != "l2_weight_decay"):
            reg_penalty = get_regularization_terms(model, regularization, reg_only_weights, l1_lambda, l2_lambda)
            loss = loss + reg_penalty
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Compute the number of correct predictions
        _, predicted_classes = predictions.max(dim=1)
        correct_predictions = (predicted_classes == batch.label).float()  # Convert to float for summation
        total_correct += correct_predictions.sum().item()
        total_instances += batch.label.size(0)

    epoch_acc = total_correct / total_instances
    # update = not torch.equal(model.embedding.weight, model.last_emb)
    # print(update)
    # model.last_emb = model.embedding.weight.clone().detach()
    return epoch_loss / len(iterator), epoch_acc

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    total_correct = 0
    total_instances = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:
            text, text_lengths = batch.text

            # Assuming your model's forward method automatically handles padding, then no need to pack sequence here
            predictions = model(text, text_lengths, is_training=False)

            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()

            # Compute the number of correct predictions
            _, predicted_classes = predictions.max(dim=1)
            
            correct_predictions = (predicted_classes == batch.label).float()  # Convert to float for summation
            total_correct += correct_predictions.sum().item()
            total_instances += batch.label.size(0)

    epoch_acc = total_correct / total_instances
    return epoch_loss / len(iterator), epoch_acc

    
def run(**kwargs):
    set_seed()
    #-------------------------- Read in kwargs --------------------------#
    train_data, valid_data, test_data, TEXT, LABEL = kwargs["train_data"], kwargs["valid_data"], kwargs["test_data"], kwargs["TEXT"], kwargs["LABEL"]
    OPTIMIZER, lr, BATCH_SIZE, HIDDEN_DIM, PACK_SEQ = kwargs['optimizer'], kwargs['lr'], kwargs['batch_size'], kwargs['hidden_dim'], kwargs['pack_seq']
    regularization = kwargs["regularization"]
    reg_only_weights = kwargs["reg_only_weights"]
    l1_lambda = kwargs["l1_lambda"]
    l2_lambda = kwargs["l2_lambda"]
    weight_decay = kwargs["weight_decay"]
    use_pretrained_word2vec = kwargs["use_pretrained_word2vec"]
    EMBEDDING_DIM = kwargs["embed_dim"]
    device = kwargs["device"]
    output_mode = kwargs["output_mode"]
    key_value_proj_dim = kwargs["key_value_proj_dim"]
    rnn = kwargs["rnn"]
    add_component = kwargs["add_component"]
    multi_output_layers = kwargs["multi_output_layers"]
    #-------------------------- Read in kwargs --------------------------#
    N_EPOCHS = 10
    best_valid_loss = float('inf')
    VOCAB_SIZE = len(TEXT.vocab)
    OUTPUT_DIM = len(LABEL.vocab)
    padding_idx = TEXT.vocab.stoi['<pad>']
    #-------------------------- Load Pretrained Embeddings --------------------------#
    pretrained_emb = None
    if use_pretrained_word2vec:
        pretrained_emb_cache = 'pretrained_emb.cache'
        if os.path.exists(pretrained_emb_cache):
            pretrained_emb = torch.load(pretrained_emb_cache)
        else:
            w2v_cache = 'word2vec-google-news-300.wordvectors'
            if os.path.exists(w2v_cache):
                wv = KeyedVectors.load(w2v_cache)# mmap='r'
            else:
                model = gensim.downloader.load('word2vec-google-news-300')
                model.save(w2v_cache)
                wv = KeyedVectors.load(w2v_cache)#mmap='r'
            word2vec_vectors = []
            for token, idx in TEXT.vocab.stoi.items():
                if token in wv.key_to_index.keys():
                    word2vec_vectors.append(torch.FloatTensor(np.array(wv[token])))
                else:
                    word2vec_vectors.append(torch.zeros(wv.vector_size))
            TEXT.vocab.set_vectors(TEXT.vocab.stoi, word2vec_vectors, wv.vector_size)
            pretrained_emb = torch.FloatTensor(TEXT.vocab.vectors)
            torch.save(pretrained_emb, pretrained_emb_cache)
    #print(pretrained_emb is None)   
    #-------------------------- Prepare Dataloader & Models --------------------------#
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = BATCH_SIZE,
        sort_within_batch = True,
        device = device)
    criterion = nn.CrossEntropyLoss().to(device)
    # Assume model is an instance of our RNN class
    
    model = RNN(
        VOCAB_SIZE, 
        EMBEDDING_DIM, 
        HIDDEN_DIM, 
        OUTPUT_DIM, 
        device, 
        PACK_SEQ, 
        output_mode,
        key_value_proj_dim,
        rnn,
        pretrained_emb,
        add_component,
        multi_output_layers,
        padding_idx
    ).to(device)
    if regularization == "l2_weight_decay":
        optim_params = get_optimizer_grouped_parameters(model, weight_decay=weight_decay, do_group=reg_only_weights)
        optimizer = OPTIMIZER(optim_params, lr=lr)
    else:
        optimizer = OPTIMIZER(model.parameters(), lr=lr)
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, 
                                      regularization, reg_only_weights, l1_lambda, l2_lambda)#regularization, 
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        # print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('tut1-model.pt'))
    # Obtain acc result with the best model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    # print(f'Validation Loss: {valid_loss:.3f} | Validation Acc: {valid_acc*100:.2f}%')
    # print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    return valid_acc, test_acc

def main(
    task,
    save_path="",
    use_pretrained_emb=False
):
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/" + save_path
    log_file_name = './results/experiment.log'
    logging.basicConfig(filename=os.path.join('./', log_file_name),
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    train_data, valid_data, test_data, TEXT, LABEL = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_res = []
    config = ExperimentConfig(task=task, use_pretrained_emb=use_pretrained_emb)
    # grid search
    for one_param_group in tqdm(config.gen_params()):
        kwargs = {
            "train_data":train_data,
            "valid_data":valid_data,
            "test_data": test_data,
            "TEXT": TEXT,
            "LABEL": LABEL,
            "device":device,
            # basic hyperparams
            "pack_seq": config.PACK_SEQ,
            "optimizer": one_param_group[0],
            "lr":one_param_group[1],
            "batch_size": one_param_group[2],
            "hidden_dim": one_param_group[3],
            "rnn":one_param_group[4],
            # output params
            "output_mode": one_param_group[5],
            "key_value_proj_dim": one_param_group[6],
            # regularization params
            "regularization": one_param_group[7], 
            "reg_only_weights": one_param_group[8], 
            "l1_lambda": one_param_group[9], 
            "l2_lambda": one_param_group[10], 
            "weight_decay": one_param_group[11],
            "use_pretrained_word2vec":one_param_group[12],
            "embed_dim": one_param_group[13],
            "add_component": one_param_group[14],
            "multi_output_layers": one_param_group[15]
        }
        #print(kwargs)
        res = run(**kwargs)
        if task in ['1a', '1b']:
            all_res.append(one_param_group[:4]+res)
        elif task == '1c':
            all_res.append(one_param_group[7:12]+res)
        else:
            fix_params = one_param_group[:4] + one_param_group[7:9] + (one_param_group[10],) + one_param_group[12:14]
            if task.startswith('2'):
                if task == '2':
                    all_res.append(fix_params + res)
                if task == '2gs':
                    all_res.append(one_param_group[:4] + one_param_group[7:12] + res)
            elif task == '3':
                all_res.append(fix_params + one_param_group[5:7] + res)
            elif task == '4':
                all_res.append(fix_params + one_param_group[5:7] + (one_param_group[4],) + res)
            elif task == '5':
                fix_params = one_param_group[:4] + one_param_group[7:12] + one_param_group[12:14]
                all_res.append(fix_params + one_param_group[5:7] + (one_param_group[4],) + one_param_group[14:] + res)
    basic_args = ['optimizer', 'lr', 'batch_size', 'hidden_dim']
    reg_args = ["regularization", "reg_only_weights", "l1_lambda", "l2_lambda", "weight_decay"]
    res_cols = ['valid_acc', 'test_acc']
    if task in ['1a', '1b']:
        res_df = pd.DataFrame(all_res, columns=basic_args + res_cols)
        res_df = res_df.sort_values(by=['valid_acc'], ignore_index=True, ascending=False)
    elif task == '1c':
        res_df = pd.DataFrame(all_res, columns= reg_args + ['valid_acc', 'test_acc'])
        res_df = res_df.sort_values(by=['regularization', 'valid_acc'], ignore_index=True, ascending=False)
    else:
        fix_cols = basic_args + ["regularization", "reg_only_weights", "l2_lambda", "use_pretrained_word2vec", "embed_dim"]
        if task.startswith('2'):
            if task == '2':
                res_df = pd.DataFrame(all_res, columns=fix_cols + res_cols)
            if task == '2gs':
                res_df = pd.DataFrame(all_res, columns=basic_args + reg_args + res_cols)
        elif task == '3':
            res_df = pd.DataFrame(all_res, columns=fix_cols + ["output_mode", "key_value_proj_dim"] + res_cols)
        elif task == '4':
            res_df = pd.DataFrame(all_res, columns=fix_cols + ["output_mode", "key_value_proj_dim"] + ["rnn"] + res_cols)
        elif task == '5':
            fix_cols = basic_args + reg_args + ["use_pretrained_word2vec", "embed_dim"]
            res_df = pd.DataFrame(all_res, columns=fix_cols + ["output_mode", "key_value_proj_dim"] + ["rnn", "add_component", "multi_output_layers"] + res_cols)
        res_df = res_df.sort_values(by=['valid_acc'], ignore_index=True, ascending=False)

    asterics = "*" * 20
    logger.info(asterics + f"Task {task} " + asterics)
    logger.info(res_df)
    res_df.to_csv(save_path, index=False)

class ExperimentConfig():
    def __init__(self, task, use_pretrained_emb):
        self.PACK_SEQ = True
        self.use_pretrained_word2vec = [False]
        self.regularization = [None]
        self.embed_dim = [100]
        self.output_mode = [None]
        self.rnn = ['RNN']
        self.add_component = [False]
        if task == "1a":
            # default setting
            self.OPTIMIZER = [SGD]
            self.lr = [0.01]
            self.BATCH_SIZE = [64]
            self.HIDDEN_DIM = [50]
        elif task == '1b':
            # # #Wide search
            # self.OPTIMIZER = [SGD, Adagrad, Adam, AdamW]
            # self.lr = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
            # self.BATCH_SIZE = [16, 32, 64, 128]
            # self.HIDDEN_DIM = [20, 50, 100, 128, 200, 256, 512]

            # #Fine-grained search 1
            self.OPTIMIZER = [Adagrad, Adam, AdamW]
            self.lr = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2]
            self.BATCH_SIZE = [16, 32, 64, 128]
            self.HIDDEN_DIM = [60, 70, 80, 90, 100, 110, 120, 128, 140, 150, 160, 170, 180, 200, 256]  
        elif task == '1c':
            # Use best config from task '1b'
            # <class 'torch.optim.adamw.AdamW'>,0.001,32,80,0.7596330275229358,0.742
            # <class 'torch.optim.adam.Adam'>,0.002,32,90,0.7541284403669725,0.818 --> pick
            # <class 'torch.optim.adam.Adam'>,0.0002,16,200,0.7495412844036697,0.782
            # <class 'torch.optim.adam.Adam'>,0.0005,16,128,0.7486238532110092,0.82
            # <class 'torch.optim.adam.Adam'>,0.0025,64,70,0.7486238532110092,0.806
            # <class 'torch.optim.adamw.AdamW'>,0.0005,16,200,0.7477064220183486,0.786
            # <class 'torch.optim.adagrad.Adagrad'>,0.01,32,200,0.7467889908256881,0.814
            # <class 'torch.optim.adam.Adam'>,0.0005,16,170,0.7458715596330275,0.794
            # <class 'torch.optim.adamw.AdamW'>,0.002,64,110,0.744954128440367,0.834
            # <class 'torch.optim.adam.Adam'>,0.001,64,120,0.7440366972477064,0.802

            self.OPTIMIZER = [Adam]
            self.lr = [2e-3]
            self.BATCH_SIZE = [32]
            self.HIDDEN_DIM = [90]
            self.regularization = ['l1', 'l2', 'both', 'l2_weight_decay']
            self.reg_only_weights = [True, False]
            self.l1_lambda = [0.001, 0.005, 0.01, 0.05, 0.1] #0.001~0.1
            self.l2_lambda = [1e-5, 2e-5, 1e-4, 2e-4, 1e-3, 2e-3, 0.01, 0.02, 0.1, 0.2, 1, 2, 4] #0.01~10
            self.weight_decay = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        else:
            self.output_mode = [None]
            self.rnn = ['RNN']
            if task == '2':
                # # Use best config from task '1c'
                #l2,True,0.0,2e-05,0.0,0.8027522935779816,0.854
                self.OPTIMIZER = [Adam]
                self.lr = [2e-3]
                self.BATCH_SIZE = [32]
                self.HIDDEN_DIM = [90]
                self.regularization = ['l2']
                self.reg_only_weights = [True]
                self.l1_lambda = [0] 
                self.l2_lambda = [2e-05] 
                self.weight_decay = [0]
                # self.OPTIMIZER = [Adam]
                # self.lr = [2e-4]
                # self.BATCH_SIZE = [16]
                # self.HIDDEN_DIM = [200]
                # self.regularization = ['l2']
                # self.reg_only_weights = [False]
                # self.l1_lambda = [0] 
                # self.l2_lambda = [1e-5] 
                # self.weight_decay = [0]
                self.embed_dim = [300,100]
                self.use_pretrained_word2vec = [True, False]
            if task == '2gs':
                self.use_pretrained_word2vec = [True]
                self.embed_dim = [300]
                # # #Fine-grained search: basic params
                # self.OPTIMIZER = [Adam, AdamW]
                # self.lr = [1e-4, 2e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2, 5e-2]
                # self.BATCH_SIZE = [16, 32, 64, 128]
                # self.HIDDEN_DIM = [60, 70, 80, 90, 100, 110, 120, 128, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300]
                # self.regularization = [None]

                # <class 'torch.optim.adam.Adam'>,0.0002,16,190,,,0,0,0,0.8770642201834863,0.904
                # <class 'torch.optim.adam.Adam'>,0.0002,16,140,,,0,0,0,0.8761467889908257,0.902
                # <class 'torch.optim.adam.Adam'>,0.0005,16,170,,,0,0,0,0.8761467889908257,0.884
                # <class 'torch.optim.adamw.AdamW'>,0.0002,16,120,,,0,0,0,0.8761467889908257,0.888
                # <class 'torch.optim.adamw.AdamW'>,0.0005,32,190,,,0,0,0,0.8743119266055046,0.886
                # <class 'torch.optim.adam.Adam'>,0.0005,32,170,,,0,0,0,0.8733944954128441,0.9 --> best with regularization
                # <class 'torch.optim.adam.Adam'>,0.0002,16,200,,,0,0,0,0.8733944954128441,0.914
                self.OPTIMIZER = [Adam]
                self.lr = [5e-4]
                self.BATCH_SIZE = [32]
                self.HIDDEN_DIM = [170]
                self.regularization = ['l1', 'l2', 'both', 'l2_weight_decay']
                self.reg_only_weights = [True, False]
                self.l1_lambda = [0.001, 0.005, 0.01, 0.05, 0.1] #0.001~0.1
                self.l2_lambda = [1e-5, 2e-5, 1e-4, 2e-4, 1e-3, 2e-3, 0.01] #0.01~10
                self.weight_decay = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            
            if task == '3':
                self.output_mode = [None, 'max', 'avg', 'attention']
                if use_pretrained_emb:
                    # # Use best config in task 2
                    #<class 'torch.optim.adam.Adam'>,0.0002,16,190,,,0,0,0,0.8770642201834863,0.904
                    #<class 'torch.optim.adam.Adam'>,0.0005,32,170,l2,False,0.0,2e-05,0.0,0.8807339449541285,0.906 --> pick
                    #<class 'torch.optim.adam.Adam'>,0.0002,16,200,l2,False,0.0,1e-05,0.0,0.881651376146789,0.892
                    #<class 'torch.optim.adamw.AdamW'>,0.0005,32,190,l2_weight_decay,False,0.0,0.0,0.001,0.8834862385321101,0.89
                    self.key_value_proj_dim = [10, 17, 34, 85, 170]
                    self.use_pretrained_word2vec = [True]
                    self.embed_dim = [300]
                    self.OPTIMIZER = [Adam]
                    self.lr = [5e-4]
                    self.BATCH_SIZE = [32]
                    self.HIDDEN_DIM = [170]
                    self.regularization = ['l2']
                    self.reg_only_weights = [False]
                    self.l1_lambda = [0] 
                    self.l2_lambda = [2e-05] 
                    self.weight_decay = [0]
                else:
                    self.key_value_proj_dim = [9, 10, 15, 30, 45, 90]
                    self.use_pretrained_word2vec = [False]
                    self.embed_dim = [100]
                    self.OPTIMIZER = [Adam]
                    self.lr = [2e-3]
                    self.BATCH_SIZE = [32]
                    self.HIDDEN_DIM = [90]
                    self.regularization = ['l2']
                    self.reg_only_weights = [True]
                    self.l1_lambda = [0] 
                    self.l2_lambda = [2e-05] 
                    self.weight_decay = [0] 
            if task == '4':
                self.rnn = ['RNN', 'GRU', 'LSTM', 'bi-RNN', '2l-RNN']
                if use_pretrained_emb:
                    # #Use best config in task 2
                    #<class 'torch.optim.adam.Adam'>,0.0005,32,170,l2,False,0.0,2e-05,0.0,0.8807339449541285,0.906 --> pick
                    self.use_pretrained_word2vec = [True]
                    self.embed_dim = [300]
                    self.OPTIMIZER = [Adam]
                    self.lr = [5e-4]
                    self.BATCH_SIZE = [32]
                    self.HIDDEN_DIM = [170]
                    self.regularization = ['l2']
                    self.reg_only_weights = [False]
                    self.l1_lambda = [0] 
                    self.l2_lambda = [2e-05] 
                    self.weight_decay = [0]
                else:
                    # #Use best config in task 1b
                    # <class 'torch.optim.adamw.AdamW'>,0.001,32,80,0.7596330275229358,0.742
                    # <class 'torch.optim.adam.Adam'>,0.0005,16,128,0.7486238532110092,0.82
                    self.use_pretrained_word2vec = [False]
                    self.embed_dim = [100]
                    self.OPTIMIZER = [AdamW]
                    self.lr = [1e-3]
                    self.BATCH_SIZE = [32]
                    self.HIDDEN_DIM = [80]
                    self.regularization = [None]
                    # self.reg_only_weights = [True]
                    # self.l1_lambda = [0] 
                    # self.l2_lambda = [2e-05] 
                    # self.weight_decay = [0]
            
            if task == '5':
                self.add_component = [True, False]
                self.multi_output_layers = [True, False]
                self.rnn = ['RNN', 'LSTM']
                if use_pretrained_emb:
                    #<class 'torch.optim.adam.Adam'>,0.001,64,170,,,0,True,300,,,RNN,True,False,0.8871559633027523,0.908
                    #<class 'torch.optim.adam.Adam'>,0.001,16,200,,,0,True,300,,,LSTM,True,False,0.8862385321100917,0.896
                    #<class 'torch.optim.adam.Adam'>,0.001,16,128,,,0,True,300,,,GRU,False,,0.8908256880733945,0.906
                    self.use_pretrained_word2vec = [True]
                    self.embed_dim = [300]
                    self.OPTIMIZER = [Adam]
                    self.lr = [1e-3]#[2e-4, 5e-4, 1e-3]
                    self.BATCH_SIZE = [16]#[16, 32, 64]
                    self.HIDDEN_DIM = [200]#[128, 150, 170, 190, 200]
                    self.regularization = [None, 'l1', 'l2', 'both', 'l2_weight_decay']
                    self.reg_only_weights = [True, False]
                    self.l1_lambda = [0.001, 0.005, 0.01, 0.05, 0.1] #0.001~0.1
                    self.l2_lambda = [1e-5, 2e-5, 1e-4, 2e-4, 1e-3, 2e-3, 0.01] #0.01~10
                    self.weight_decay = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
                else:
                    self.OPTIMIZER = [Adam]
                    self.lr = [2e-3]
                    self.BATCH_SIZE = [32]
                    self.HIDDEN_DIM = [90]
                    self.regularization = ['l2']
                    self.reg_only_weights = [True]
                    self.l1_lambda = [0] 
                    self.l2_lambda = [2e-05] 
                    self.weight_decay = [0]


    def gen_params(self):
        param_space = []
        basic_params = [self.OPTIMIZER, self.lr, self.BATCH_SIZE, self.HIDDEN_DIM, self.rnn]
        for reg in self.regularization:
            # reg_params: regularization, reg_only_weights, l1_lambda, l2_lambda, weight_decay
            if reg is None:
                reg_params = [[None], [None], [0], [0], [0]]
            elif reg == 'l1':
                reg_params = [['l1'], self.reg_only_weights, self.l1_lambda, [0], [0]]
            elif reg == 'l2':
                reg_params = [['l2'], self.reg_only_weights, [0], self.l2_lambda, [0]]
            elif reg == 'both':
                reg_params = [['both'], self.reg_only_weights, self.l1_lambda, self.l2_lambda, [0]]
            elif reg == 'l2_weight_decay':
                reg_params = [['l2_weight_decay'], self.reg_only_weights, [0], [0], self.weight_decay]
            for emb in self.use_pretrained_word2vec:
                emb_params = [[True], [self.embed_dim[0]]] if emb else [[False], [self.embed_dim[-1]]]
                
                for oput in self.output_mode:
                    if oput == 'attention':
                        output_params = [[oput], self.key_value_proj_dim]
                    else:
                        output_params = [[oput], [None]]
                    
                    for addc in self.add_component:
                        addc_params = [[True], self.multi_output_layers] if addc else [[False], [None]]
                        
                        params = basic_params + output_params + reg_params + emb_params + addc_params
                        param_space.extend(list(itertools.product(*params)))
        
        return param_space

class TextCNN(nn.Module):
    def __init__(
        self,
        embedding_dim,
        output_dim,
        multi_output_layers=False,
        channel_num = 1,
        kernel_num = 100,
        dropout = 0.3,
        kernel_sizes = [3,4,5]
    ):
        super(TextCNN, self).__init__()

        self.convs = nn.ModuleList([nn.Conv2d(channel_num, kernel_num, (size, embedding_dim)) for size in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.output_layers = nn.Sequential(
            nn.Linear(len(kernel_sizes) * kernel_num, 128), 
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(128, output_dim)
        ) if multi_output_layers else nn.Linear(len(kernel_sizes) * kernel_num, output_dim) #fc

    def forward(
        self, 
        inputs,
        is_training=False
    ):
        x = inputs
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        if is_training:
            x = self.dropout(x)
        logits = self.output_layers(x)
        return logits

main(task='1a', save_path="task1a_pack_dafld.csv")
#main(task='1b', save_path="task1b_grid_search_3.csv", use_pretrained_emb=False)#
#main(task='2', save_path="task2.csv")
#main(task='4', save_path="task4.csv", use_pretrained_emb=True)#
#main(task='4', save_path='task4_no_pretrain_emb_2.csv', use_pretrained_emb=False)
#main(task='5', save_path='task5gs_lstm_reg.csv', use_pretrained_emb=True)