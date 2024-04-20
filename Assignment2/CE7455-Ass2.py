from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
import math
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
import numpy as np
import pandas as pd
import transformers
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 15

# #----------------Helper Funcs----------------#
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
# #---------------- Prepare Datasets and Vocabulary ----------------#
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    eng_prefixes = (
        "i am", "i m",
        "he is", "he s",
        "she is", "she s",
        "you are", "you re",
        "we are", "we re",
        "they are", "they re"
    )
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
# #-----------------------------Prepare Tensors-------------------------#
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1) # [seq_len, 1]

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
# 

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, use_attention=False, use_transformer=False):
    teacher_forcing_ratio = 0.5
    input_length = input_tensor.size(0) # input_tensor = [seq_len, batch_size=1]
    target_length = target_tensor.size(0)
    # encoder_hidden = [n_layer * n_direction=1, batch_size, hidden_size]
    encoder_hidden = encoder.initHidden()


    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # encoder_outputs = [seq_len, n_direction * hidden_size]
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size * encoder.n_direction, device=device)

    loss = 0
    
    if use_transformer:
        # ------------------- Follow Task Instruction -------------------#
        # Note: You should take the mean of all hidden representation output from the transformer encoder of each token to be a sentence representation of the encoder.
        # Also, for the transformer encoder, you must input the whole sentence instead of feeding word by word to get the next token representation.
        encoder_outputs = encoder(input_tensor) # encoder_ouputs = [seq_len, batch_size, hidden_size]
        #print(encoder_outputs.size())
        
        # encoder_hidden : sentence representation
        encoder_hidden = torch.mean(encoder_outputs, dim=0, keepdim=True) # [1, batch_size, hidden_size]
        #encoder_hidden = torch.max(encoder_outputs, dim=0, keepdim=True).values
        # print(encoder_hidden.size())
        # exit()
        encoder_outputs = encoder_outputs.squeeze(1)
    else:
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden) # encoder_output = [seq_len, batch_size, hidden_size]
            encoder_outputs[ei] = encoder_output[0, 0]
        
    decoder_input = torch.tensor([[SOS_token]], device=device)
    
    if ("LSTM" in encoder.rnn_type) and ('LSTM' not in decoder.rnn_type):
        decoder_hidden = encoder_hidden[0].view(1, 1, -1)
    else:
        decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if use_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if use_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, input_lang, output_lang, train_pairs, epochs, print_every=1000, plot_every=100, learning_rate=0.01, use_attention=False, use_transformer=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    iter = 1
    n_iters = len(train_pairs) * epochs

    for epoch in tqdm(range(epochs)):
        print("Epoch: %d/%d" % (epoch, epochs))
        for training_pair in tqdm(train_pairs):
            training_pair = tensorsFromPair(training_pair, input_lang, output_lang)

            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion, use_attention=use_attention, use_transformer=use_transformer)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            iter +=1

# # -----------------------------Predict Step-----------------------------------#
def prediction_step(encoder, decoder, input_lang, output_lang, sentence, max_length=MAX_LENGTH, use_attention=False, use_transformer=False):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size * encoder.n_direction, device=device)
        if use_transformer:
            # ------------------- Follow Task Instruction -------------------#
            # Note: You should take the mean of all hidden representation output from the transformer encoder of each token to be a sentence representation of the encoder.
            # Also, for the transformer encoder, you must input the whole sentence instead of feeding word by word to get the next token representation.
            encoder_outputs = encoder(input_tensor) # encoder_ouputs = [seq_len, batch_size, hidden_size]
            # encoder_hidden : sentence representation
            encoder_hidden = torch.mean(encoder_outputs, dim=0, keepdim=True) # [1, batch_size, hidden_size]
            #encoder_hidden = torch.max(encoder_outputs, dim=0, keepdim=True).values
            encoder_outputs = encoder_outputs.squeeze(1)
        else:
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        
        if ("LSTM" in encoder.rnn_type) and ('LSTM' not in decoder.rnn_type):
            decoder_hidden = encoder_hidden[0].view(1, 1, -1)
        else:
            decoder_hidden = encoder_hidden

        decoded_words = []
        if use_attention:
            decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            if use_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            # values, indices, most likely word
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        
        if use_attention:
            return decoded_words, decoder_attentions[:di + 1]
        else:
            return decoded_words

def test(encoder, decoder, input_lang, output_lang, testing_pairs, use_attention=False, use_transformer=False):
    rouge = ROUGEScore()
    input = []
    gt = []
    predict = []
    metric_score = {
        "rouge1_fmeasure":[],
        "rouge1_precision":[],
        "rouge1_recall":[],
        "rouge2_fmeasure":[],
        "rouge2_precision":[],
        "rouge2_recall":[]
    }
    
    for i in tqdm(range(len(testing_pairs))):
        pair = testing_pairs[i]
        if use_attention:
            output_words, _ = prediction_step(encoder, decoder, input_lang, output_lang, pair[0], use_attention=use_attention, use_transformer=use_transformer)
        else:
            output_words = prediction_step(encoder, decoder, input_lang, output_lang, pair[0], use_transformer=use_transformer)
        output_sentence = ' '.join(output_words)

        input.append(pair[0])
        gt.append(pair[1])
        predict.append(output_sentence)

        try:
            rs = rouge(output_sentence, pair[1])
        except:
            continue
        for s_name in list(metric_score.keys()):
            metric_score[s_name].append(rs[s_name])
        # metric_score["rouge1_fmeasure"].append(rs['rouge1_fmeasure'])
        # metric_score["rouge1_precision"].append(rs['rouge1_precision'])
        # metric_score["rouge1_recall"].append(rs['rouge1_recall'])
        # metric_score["rouge2_fmeasure"].append(rs['rouge2_fmeasure'])
        # metric_score["rouge2_precision"].append(rs['rouge2_precision'])
        # metric_score["rouge2_recall"].append(rs['rouge2_recall'])
    
    for s_name in list(metric_score.keys()):
        metric_score[s_name] = np.array(metric_score[s_name]).mean()

    # metric_score["rouge1_fmeasure"] = np.array(metric_score["rouge1_fmeasure"]).mean()
    # metric_score["rouge1_precision"] = np.array(metric_score["rouge1_precision"]).mean()
    # metric_score["rouge1_recall"] = np.array(metric_score["rouge1_recall"]).mean()
    # metric_score["rouge2_fmeasure"] = np.array(metric_score["rouge2_fmeasure"]).mean()
    # metric_score["rouge2_precision"] = np.array(metric_score["rouge2_precision"]).mean()
    # metric_score["rouge2_recall"] = np.array(metric_score["rouge2_recall"]).mean()

    print("=== Evaluation score - Rouge score ===")
    for s_name, s_value in metric_score.items():
        print(f"{s_name}:\t{s_value}")
    # print("Rouge1 fmeasure:\t",metric_score["rouge1_fmeasure"])
    # print("Rouge1 precision:\t",metric_score["rouge1_precision"])
    # print("Rouge1 recall:  \t",metric_score["rouge1_recall"])
    # print("Rouge2 fmeasure:\t",metric_score["rouge2_fmeasure"])
    # print("Rouge2 precision:\t",metric_score["rouge2_precision"])
    # print("Rouge2 recall:  \t",metric_score["rouge2_recall"])
    print("=====================================")
    return input,gt,predict,metric_score

# # ----------------------------Model ----------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, rnn="GRU", use_transformer=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_direction = 1

        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        
        self.use_transformer = use_transformer
        if use_transformer:
            self.rnn_type = ""
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=64)#8
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)#6
        else:
            self.rnn_type = rnn
            if rnn == "GRU":
                self.rnn = nn.GRU(hidden_size, hidden_size)
            elif rnn == "LSTM":
                self.rnn = nn.LSTM(hidden_size, hidden_size)
            elif rnn == "bi-LSTM":
                self.n_direction = 2
                self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        
    def forward(self, input, hidden=None):
        
        if self.use_transformer:
            # embedded = [seq_len, batch_size, hidden_size]
            embedded = self.embedding(input)
            output = self.transformer_encoder(embedded)
            return output
        else:
            embedded = self.embedding(input).view(1, 1, -1)
            output = embedded
            # output = [seq_len, batch_size, n_direction * hidden_size]
            #       containing the output features (h_t) from the last layer of the LSTM, for each time step t
            # hidden = [n_layer * n_direction, batch_size, hidden_size]
            #       containing the final hidden state
            output, hidden = self.rnn(output, hidden) # hidden is a tuple of (hidden, cell)
            return output, hidden

    def initHidden(self):
        if self.use_transformer:
            return torch.zeros(1, 1, self.hidden_size, device=device)
        else:
            if self.rnn_type == "GRU":
                return torch.zeros(1, 1, self.hidden_size, device=device)
            elif self.rnn_type == "LSTM":
                return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))
            elif self.rnn_type == "bi-LSTM":
                return (torch.zeros(1 * 2, 1, self.hidden_size, device=device), torch.zeros(1 * 2, 1, self.hidden_size, device=device))

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
              
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_vocab_size, rnn):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)        
        self.rnn_type = rnn
        if rnn == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size)
        elif rnn == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Your code here #
        # input = [seq_len=1, batch_size=1] e.g.,[[token_id]]
        # hidden = [n_layer * n_direction, batch_size, hidden_size]
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        
        if "LSTM" in self.rnn_type:
            # output = [seq_len, batch_size, n_direction * hidden_size]
            # hidden = [n_layer * n_direction, batch_size, hidden_size]
            output, hidden = self.rnn(output, hidden)
        else:
            output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def run(
    task_name,
    input_lang, 
    output_lang,
    train_pairs,
    test_pairs,
    encoder_rnn="GRU", 
    decoder_rnn="GRU", 
    decoder_hidden_size=None,
    use_attention=False, 
    use_transformer=False,
):
    # Default setting
    epochs = 5
    hidden_size = 512
    encoder1 = Encoder(input_lang.n_words, hidden_size, rnn=encoder_rnn, use_transformer=use_transformer).to(device)
    if use_attention:
        decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    else:
        d_hidden_size = hidden_size if decoder_hidden_size is None else decoder_hidden_size
        decoder1 = Decoder(d_hidden_size, output_lang.n_words, rnn=decoder_rnn).to(device)

    trainIters(encoder1, decoder1, input_lang, output_lang, train_pairs, epochs=epochs, print_every=5000, use_attention=use_attention, use_transformer=use_transformer)
    #input,gt,predict,score = test(encoder1, decoder1, train_pairs)
    input,gt,predict,score = test(encoder1, decoder1, input_lang, output_lang, test_pairs, use_attention=use_attention, use_transformer=use_transformer)
    del encoder1
    del decoder1
    return {task_name: [round(v,3) for k, v in score.items()]}, [k for k, _ in score.items()]

def main():
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    X = [i[0] for i in pairs]
    y = [i[1] for i in pairs]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    train_pairs = list(zip(X_train,y_train))
    test_pairs = list(zip(X_test,y_test))
    
    all_res_dict = {}
    # Task 1
    score_vals, score_names = run("GRU Encoder + GRU Decoder", input_lang, output_lang, train_pairs, test_pairs, \
                            encoder_rnn="GRU", decoder_rnn="GRU")
    all_res_dict.update(score_vals)

    # Task 2
    all_res_dict.update(run("LSTM Encoder + LSTM Decoder", input_lang, output_lang, train_pairs, test_pairs, \
                            encoder_rnn="LSTM", decoder_rnn="LSTM")[0])

    # Task 3
    all_res_dict.update(run("bi-LSTM Encoder + GRU Decoder", input_lang, output_lang, train_pairs, test_pairs, \
                            encoder_rnn="bi-LSTM", decoder_hidden_size=1024, decoder_rnn="GRU")[0])
    
    # Task 4
    all_res_dict.update(run("GRU Encoder+ Attention + GRU Decoder", input_lang, output_lang, train_pairs, test_pairs, \
                            use_attention=True)[0])
    
    # Task 5
    score_vals, score_names = run("Transformer Encoder + GRU Decoder", input_lang, output_lang, train_pairs, test_pairs, \
                            use_transformer=True)
    all_res_dict.update(score_vals)
    
    res_df = pd.DataFrame.from_dict(all_res_dict, orient='index', columns = score_names)
    res_df = res_df.reset_index()
    res_df.to_csv("Assignment2_results.csv", index=False)
 
main()