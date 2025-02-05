# %%
import re
import math
import torch
import torchtext
import import_ipynb
import torch.nn as nn
import torch.nn.functional as F
from data_processing import src_vocab
from torchtext.data.utils import get_tokenizer
DEVICE = torch.device('cuda')

# %% [markdown]
# # Xây dựng class embedding tokens

# %%
class TokenizedEmbedding(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super(TokenizedEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings= vocab_size, embedding_dim= embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.embed_dim) 

# %% [markdown]
# # Lớp Encoder 

# %%
class RnnEncoder(nn.Module):
    def __init__(self, src_vocab, embed_dim, hidden_dim, n_layers, dropout, DEVICE):
        super(RnnEncoder, self).__init__()
        
        self.embedding = TokenizedEmbedding(
            embed_dim= embed_dim,
            vocab_size= len(src_vocab)
            )
        
        self.rnn_layer = nn.GRU(
            input_size= embed_dim, 
            hidden_size= hidden_dim,
            num_layers= n_layers,
            batch_first= True,
            bidirectional= True
            )
        
        self.norm = nn.LayerNorm(normalized_shape= hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(
            in_features= hidden_dim*2,
            out_features= hidden_dim,
            device= DEVICE
            )
        self.fc2 = nn.Linear(
            in_features= n_layers*2,
            out_features= n_layers,
            device= DEVICE
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        #x là câu đầu vào [batch_size, seq_len]
        x = self.embedding(x) # |x| = [batch_size, seq_len, embed_dim]
        x, hn = self.rnn_layer(x) # |hn| = [n_layers*2, batch_size, hidden_dim]
        hn = hn.permute(1,2,0)
        hn = self.fc2(hn)
        hn = hn.permute(2,0,1)
        hn = self.norm(hn)
        hn = self.relu(hn)
        hn = self.dropout(hn)
        x = self.fc(x)# |hn| = [n_layers, batch_size, hidden_dim]
        return x, hn

# %% [markdown]
# # Lớp Decoder

# %%
class RnnDecoder(nn.Module):
    def __init__(self, tgt_vocab, hidden_dim, n_layers, dropout, SOS_token, EOS_token, PAD_token, DEVICE, teaching_force= 0.3, MAX_LEN= 20):
        super(RnnDecoder, self).__init__()
        
        self.embedding = TokenizedEmbedding(
            embed_dim= hidden_dim,
            vocab_size= len(tgt_vocab)
            )

        self.rnn_layer = nn.GRU(
            input_size= hidden_dim, 
            hidden_size= hidden_dim,
            num_layers= n_layers,
            batch_first= True
            )
        
        self.avg_pool = nn.AdaptiveAvgPool1d(n_layers)

        self.fc = nn.Linear(
            in_features= hidden_dim, 
            out_features= len(tgt_vocab)
        )        

        self.norm = nn.LayerNorm(normalized_shape= hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = DEVICE        
        self.sos_token = SOS_token
        self.eos_token = EOS_token
        self.pad_token = PAD_token
        # Chiều dài câu đầu ra mong muốn của bộ giải mã
        self.max_len = MAX_LEN
        self.tf = teaching_force
    
    def forward(self, encoder_outputs, encoder_final_hidden, target_tensor=None):
        # encoder_outputs: đầu ra của encoder cho toàn bộ chuỗi đầu vào [batch_size, seq_len, hidden_dim]
        # encoder_final_hidden: hidden state cuối cùng của encoder [n_layers, batch_size, hidden_dim]
        # target_tensor: chuỗi target thực tế (sử dụng cho teacher forcing)
        
        '''
        encoder_outputs = encoder_outputs.permute(0, 2, 1).contiguous() # [batch_size, hidden_dim, seq_len]
        contexts = self.avg_pool(encoder_outputs) # |contexts| = [batch_size, hidden_dim, n_layers]
        contexts = contexts.permute(2, 0, 1).contiguous() # |contexts| = [n_layers, batch_size, hidden_dim]
        '''
        batch_size = encoder_outputs.size(0)
        decoder_hidden = encoder_final_hidden # |decoder_input| = [n_layer, batch_size, hidden_dim]
        decoder_input = torch.empty((batch_size, 1), dtype= torch.long, device= self.device).fill_(self.sos_token)
        max_len = target_tensor.size(1)
        decoder_outputs = []
        decoder_input = self.embedding(decoder_input)
        decoder_outputs.append(decoder_input)

        for i in range(1, max_len):
            #decoder_hidden = contexts + decoder_hidden
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden) # |decoder_output| = [batch_size, 1, hidden_dim]
            decoder_outputs.append(decoder_output)

            if self.tf is not None and torch.rand(1).item() < self.tf:
                decoder_input = target_tensor[:, i].unsqueeze(1)
                decoder_input = self.embedding(decoder_input)
            else:
                decoder_input = decoder_output.detach()

        decoder_outputs = torch.cat(decoder_outputs, dim= 1)   
        decoder_outputs = self.norm(decoder_outputs)
        decoder_outputs = self.dropout(decoder_outputs)
        decoder_outputs = F.relu(decoder_outputs)
        decoder_outputs = self.fc(decoder_outputs) # [batch_size, seq_len, len(tgt_vocab)]
        
        return decoder_outputs

    def forward_step(self, input, hidden):
        output, hidden = self.rnn_layer(input, hidden)
        output = self.dropout(output)
        hidden = self.dropout(hidden)
        hidden = F.relu(hidden)
        return output, hidden
    
    def inference(self, input, contexts, max_len= 20):
        self.eval()
        with torch.no_grad():
            batch_size = input.size(0)
            decoder_input = torch.empty((batch_size, 1), dtype= torch.long, device= self.device).fill_(self.sos_token)
            decoder_hidden = contexts
            decoder_outputs = []
            decoder_input = self.embedding(decoder_input)
            decoder_outputs.append(decoder_input)

            for i in range(1, max_len):
                decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden) # |decoder_output| = [batch_size=1, 1, hidden_dim]
                # Kiểm tra phần tử kết thúc <EOS>
                pred = self.fc(decoder_output) # |decoder_output| = [batch_size=1, 1, len(tgt_vocab)]
                pred = torch.argmax(torch.log_softmax(pred, dim= -1), dim= -1)
                if pred.item() == self.eos_token:
                    break

                decoder_outputs.append(decoder_output)
                decoder_input = decoder_output.detach()

            decoder_outputs = torch.cat(decoder_outputs, dim= 1)   # |decoder_output| = [batch_size=1, predicted_seq_len, len(tgt_vocab)]
            decoder_outputs = self.norm(decoder_outputs)
            decoder_outputs = self.dropout(decoder_outputs)
            decoder_outputs = F.relu(decoder_outputs)
            decoder_outputs = self.fc(decoder_outputs) # |decoder_output| = [batch_size=1, predicted_seq_len, len(tgt_vocab)]
            inference = torch.argmax(torch.log_softmax(decoder_outputs.float(), dim= -1), dim= -1).squeeze(0) # |inference| = [batch_size, predicted_seq_len]

        return inference

# %% [markdown]
# # Model Rnn tổng quát

# %%
class RnnMachineTranslate(nn.Module):
    def __init__(self, encoder, decoder):
        super(RnnMachineTranslate, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y= None):
        # x là 1 bacth các câu đầu vào
        result_time_steps, encoder_final_hidden = self.encoder(x)
        decoder_outputs = self.decoder(result_time_steps, encoder_final_hidden, y)

        return decoder_outputs    
    
    def inference(self, input):
        self.eval()
        encoder_outputs, contexts = self.encoder(input)
        inference = self.decoder.inference(input, contexts)

        return inference 

# %% [markdown]
# # Handling Input Parttern

# %%
class TextProcessor:
    def __init__(self, contractions_dict, src_vocab, DEVICE):
        self.contractions_dict = contractions_dict
        self.src_vocab = src_vocab
        self.src_tokenizer = get_tokenizer('basic_english')
        self.device = DEVICE

    def expand_match(self, contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded = self.contractions_dict.get(match.lower())
        if expanded:
            if first_char.isupper():
                expanded = expanded[0].upper() + expanded[1:]
            return expanded
        return match

    def replace_time(self, match):
        hours = match.group('hours')
        minutes = match.group('minutes')
        components = []
        if hours:
            hours = int(hours)
            hour_word = "hour" if hours == 1 else "hours"
            components.append(f"{hours} {hour_word}")
        
        if minutes:
            minutes = int(minutes)
            if minutes < 10 and ':' in match.group():
                minutes = minutes * 10
            minute_word = "minute" if minutes == 1 else "minutes"
            components.append(f"{minutes} {minute_word}")
        return " ".join(components)

    def text_normalize(self, sentence):
        sentence = re.sub(r'\.$', '', sentence.strip())
        sentence = re.sub(r'\s*,\s*', ', ', sentence)
        sentence = re.sub(r',+', ',', sentence)
        sentence = re.sub(r'([.!?])', r' \1', sentence)

        # Replace contractions
        pattern = re.compile(r'\b(' + '|'.join(self.contractions_dict.keys()) + r')\b', re.IGNORECASE)
        sentence = pattern.sub(self.expand_match, sentence)

        # Replace time formats
        patterns = [
            r'(?P<hours>\d{1,2}):(?P<minutes>\d{1,2})',
            r'(?P<hours>\d+)\s*h(?:\s*(?P<minutes>\d+)\s*m)?'
        ]
        for part in patterns:
            sentence = re.sub(part, self.replace_time, sentence, flags=re.IGNORECASE)

        # Clean up quotes
        sentence = re.sub(r"'", '', sentence)
        sentence = sentence.strip(r'"')
        sentence = sentence.strip(r"'")
        return sentence

    def tokenize_sentence(self, sentence):
        return self.src_tokenizer(sentence)

    def get_transform(self):
        text_transform = torchtext.transforms.Sequential(
            torchtext.transforms.VocabTransform(self.src_vocab),
            torchtext.transforms.AddToken(2, True),
            torchtext.transforms.AddToken(3, False)
        )
        return text_transform

    def transform_input(self, tokenized_sentence):
        transform = self.get_transform()
        return transform(tokenized_sentence)

    def process(self, sentence):
        # Full pipeline
        normalized_sentence = self.text_normalize(sentence)
        tokenized_sentence = self.tokenize_sentence(normalized_sentence)
        transformed_sentence = self.transform_input(tokenized_sentence)
        return torch.tensor([transformed_sentence]).to(self.device)

# Example usage
CONTRACTIONS = {
    "aren't": "are not",
    "can't": "can not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "she's": "she is",
    "she'll": "she will",
    "she'd": "she would",
    "i'd": "I would",
    "i'll": "I will",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "there're": "there are",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "where's": "where is",
    "why's": "why is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "wasn't": "was not",
    "it'll": "it will",
    "tom's": "tom is",
    "one's": "one is",
    "somebody's": "somebody is",
    "someone's": "someone is",
    "that's": "that is",
    "how's": "how is",
    "should've": "should have",
    "why're": "why are",
    "might've": "might have",
    "o'clock": "of the clock",
    "here's": "here is",
    "could've": "could have",
    "must've": "must have",
    "would've": "would have",
    "that'll": "that will"
}

# %% [markdown]
# # Handling Output 

# %%
class OutputProcessor:
    def __init__(self, tgt_vocab, DEVICE):
        self.device = DEVICE
        self.tgt_vocab = tgt_vocab
    
    def clean(self, batch_seq, pad_token=1, sos_token=2, eos_token=3):
        if isinstance(batch_seq, torch.Tensor):
            batch_seq = batch_seq.cpu().numpy()
    
        # Tạo mask cho các token cần loại bỏ
        mask = (batch_seq != pad_token) & (batch_seq != sos_token) & (batch_seq != eos_token)

        # Áp dụng mask cho từng sequence trong batch và loại bỏ các mảng rỗng
        cleaned_sequences = [seq[m] for seq, m in zip(batch_seq, mask)]
        return [seq for seq in cleaned_sequences if seq.size > 0]
    
    def process(self, output):
        result = []
        output = self.clean(output)
        for i in output:
            index = i.item()
            result.append(self.tgt_vocab.get_itos()[index])
        
        # Nối các phần tử trong result thành một câu
        sentence = ' '.join(result).replace(' ,', ',').replace(' .', '.')
        return sentence



