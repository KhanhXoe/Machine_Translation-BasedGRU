# %% [markdown]
# # **Tiền xử lý dữ liệu đầu vào**

# %%
import os
import re
import torch
import torchtext
import numpy as np
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset

from torchtext import transforms as T
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from sklearn.model_selection import train_test_split

# %% [markdown]
# ## **Chuẩn hóa dữ liệu văn bản đầu vào**

# %% [markdown]
# ### Phần xác định các cụm viết tắt

# %%
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
    "she's":"she is",
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
    "we'll":"we will",
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
    "wasn't":"was not",
    "it'll":"it will",
    "tom's":"tom is", 
    "one's":"one is",
    "somebody's": "sombody is",
    "someone's": "someone is",
    "that's": "that is",
    "how's": "how is",
    "should've": "should have",
    "why're": "why are",
    "might've": "might have",
    "o'clock": "of the clock",
    "here's": 'here is',
    "could've": "could have",
    "must've": "must have",
    "would've": "would have",
    "that'll": "that will"
}

# %%


# %% [markdown]
# ### Chuẩn hóa các dấu câu, viết tắt, viết hoa

# %% [markdown]
# *Thực hiện kèm Tokenizer*

# %%
file_path = r"data.csv"

def expand_match(contraction, contractions_dict= CONTRACTIONS):
    match = contraction.group(0)
    first_char = match[0]
    expanded = contractions_dict.get(match.lower())
    if expanded:
        if first_char.isupper():
            expanded = expanded[0].upper() + expanded[1:]
        return expanded
    return match

def replace_time(match):
        hours = match.group('hours')
        minutes = match.group('minutes')
        
        components = []
        
        if hours:
            hours = int(hours)
            hour_word = "hour" if hours == 1 else "hours"
            components.append(f"{hours} {hour_word}")
        
        if minutes:
            minutes = int(minutes)
            # Chuẩn hóa số phút
            if minutes < 10 and ':' in match.group():
                minutes = minutes * 10
            minute_word = "minute" if minutes == 1 else "minutes"
            components.append(f"{minutes} {minute_word}")
        
        return " ".join(components)

def text_normalize(file_path, contractions_dict= CONTRACTIONS):
    src_data = pd.read_csv(file_path, delimiter="\t", header=None, usecols=[0, 1])
    src_data.columns = ['src', 'tgt']

    normal_data = {}
    normal_data['src'] = []
    for row in src_data['src']:
    #Xóa dấu . cuối câu
        sentence = re.sub(r'\.$', '', row.strip())
        #Xóa khoảng trắng thừa
        sentence = re.sub(r'\s*,\s*', ', ', sentence)
        #Xóa dấu phẩy thừa
        sentence = re.sub(r',+', ',', sentence)
        #Tách dấu câu còn lại để lấy ngữ cảnh
        sentence = re.sub(r'([.!?])', r' \1', sentence)

        # Thực hiện thay thế viết tắt trong text
        pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b', re.IGNORECASE)    
        sentence = pattern.sub(expand_match, sentence)
        
        patterns = [
        # h:m hoặc h:mm
        r'(?P<hours>\d{1,2}):(?P<minutes>\d{1,2})',
        # Các định dạng khác
        r'(?P<hours>\d+)\s*h(?:\s*(?P<minutes>\d+)\s*m)?'
        ]
        for part in patterns:
            sentence = re.sub(part, replace_time, sentence, flags=re.IGNORECASE)
        #Xóa dấu " thừa ra
        sentence = re.sub(r"'", '', sentence)
        sentence = sentence.strip(r'"')
        sentence = sentence.strip(r"'")

        #sentence = tokenizer(sentence)
        normal_data['src'].append(sentence)
    
    normal_data['tgt'] = []
    for row in src_data['tgt']:
    #Xóa dấu . cuối câu
        sentence = re.sub(r'\.$', '', row.strip())
        #Xóa khoảng trắng thừa
        sentence = re.sub(r'\s*,\s*', ', ', sentence)
        #Xóa dấu phẩy thừa
        sentence = re.sub(r',+', ',', sentence)
        #Tách dấu câu còn lại để lấy ngữ cảnh
        sentence = re.sub(r'([.!?])', r' \1', sentence)
        # Thực hiện thay thế trong text
        pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b', re.IGNORECASE)    
        sentence = pattern.sub(expand_match, sentence)
        
        patterns = [
        # h:m hoặc h:mm
        r'(?P<hours>\d{1,2}):(?P<minutes>\d{1,2})',
        
        # Các định dạng khác
        r'(?P<hours>\d+)\s*h(?:\s*(?P<minutes>\d+)\s*m)?',
        ]
        for part in patterns:
            sentence = re.sub(part, replace_time, sentence, flags=re.IGNORECASE)
        #Xóa dấu " thừa ra
        sentence = re.sub(r"'", '', sentence)
        sentence = sentence.strip(r'"')
        sentence = sentence.strip(r"'")
        #sentence = tokenizer(sentence)
        normal_data['tgt'].append(sentence)

    return normal_data

# %% [markdown]
# ## **Quá trình vectorize dữ liệu text**

# %% [markdown]
# ### Xây dựng từ điển (vocab) cho bản gốc và bản dịch

# %%
def yield_src_tokens(dataset, src_tokenizer):
    for row in dataset:
        yield src_tokenizer(row)

def yield_tgt_tokens(dataset, tgt_tokenizer):
    for row in dataset:
        yield tgt_tokenizer(row)

def buildVocab(yield_src_tokens, yield_tgt_tokens, src_tokenizer, tgt_tokenizer, normalized_data, src_max, tgt_max):
    src_vocab = build_vocab_from_iterator(
        yield_src_tokens(normalized_data['src'], src_tokenizer), 
        specials=['<unk>', '<pad>', '<sos>', '<eos>'], 
        max_tokens= src_max, 
        min_freq= 2)
    src_vocab.set_default_index(src_vocab['<unk>'])

    tgt_vocab = build_vocab_from_iterator(
        yield_tgt_tokens(normalized_data['tgt'], tgt_tokenizer),
        specials=['<unk>', '<pad>', '<sos>', '<eos>'],
        max_tokens= tgt_max, 
        min_freq= 2)
    tgt_vocab.set_default_index(tgt_vocab['<unk>'])

    return src_vocab, tgt_vocab

# %% [markdown]
# ### Số hóa các vector chữ cái đầu vào

# %%
def tokenize_dataset(normalized_data, src_tokenizer, tgt_tokenizer):
    tokenized_data = {}
    for column in normalized_data:
        tokenized_data[column] = []
        if column== 'src':
            for row in normalized_data[column]:
                tokenized_data[column].append(src_tokenizer(row))
        if column== 'tgt':
            for row in normalized_data[column]:
                tokenized_data[column].append(tgt_tokenizer(row))

    return tokenized_data

# %%
def getTransform(vocab):
    """
    Create the transform functions for the senquence.
    """
    text_transform = T.Sequential(
        T.VocabTransform(vocab),
        T.AddToken(2, True),
        T.AddToken(3, False)
    )
    return text_transform

def transform_dataset(tokenized_data, src_vocab, tgt_vocab):
    transformed_data = {}
    
    for column in tokenized_data:
        if column== 'src':
            transformed_data[column] = getTransform(vocab= src_vocab)(tokenized_data['src'])
        if column== 'tgt':
            transformed_data[column] = getTransform(vocab= tgt_vocab)(tokenized_data['tgt'])

    return transformed_data

# %%
src_max = 200000
tgt_max = 200000
src_tokenizer = get_tokenizer("basic_english")
tgt_tokenizer = get_tokenizer("basic_english")

normalized_data = text_normalize(file_path)
src_vocab, tgt_vocab = buildVocab(yield_src_tokens, yield_tgt_tokens, src_tokenizer, tgt_tokenizer, normalized_data, src_max, tgt_max)
tokenized_data = tokenize_dataset(normalized_data, src_tokenizer, tgt_tokenizer)

transformed_data = transform_dataset(tokenized_data, src_vocab, tgt_vocab)
transformed_data = pd.DataFrame(transformed_data)

# %%


# %% [markdown]
# ## **Quá trình xây dựng DataLoader**

# %% [markdown]
# ### Xây dựng Dataset và thực hiện load lên DataLoader phục vụ cho huấn luyện

# %%
class TranslationDataset(Dataset):
    def __init__(self, source_data, target_data):
        """
        source_data: Danh sách các câu nguồn, mỗi câu là một danh sách các chỉ số (indices).
        target_data: Danh sách các câu đích, mỗi câu là một danh sách các chỉ số (indices).
        """
        self.source_data = source_data
        self.target_data = target_data

    def __len__(self):
        # Trả về số lượng mẫu dữ liệu (số lượng cặp câu nguồn-đích)
        return len(self.source_data)

    def __getitem__(self, idx):
        # Trả về cặp câu nguồn và đích tại vị trí idx
        source_sentence = torch.tensor(self.source_data.iloc[idx], dtype=torch.long)
        target_sentence = torch.tensor(self.target_data.iloc[idx], dtype=torch.long)
        return source_sentence, target_sentence


# Tạo DataLoader, sử dụng collate_fn để xử lý padding và drop cho các câu có độ dài không đồng đều
def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    
    # Padding cho các chuỗi nguồn và đích
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value= 1)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value= 1)
    
    return src_batch, tgt_batch


# %% [markdown]
# # **Đầu ra kết quả**

# %%
#####################################################################
#####################################################################
# Phân chia các tập huấn luyện 
val_ratio = 0.2
test_ratio = 0.125

train_data, val_data = train_test_split(
    transformed_data[['src', 'tgt']],
    test_size= val_ratio,
    random_state= 42,
    shuffle= True
)

train_data, test_data = train_test_split(
    train_data[['src', 'tgt']],
    test_size= test_ratio,
    random_state= 42,
    shuffle= False
)
#####################################################################
#####################################################################

train_dataset = TranslationDataset(train_data['src'], train_data['tgt'])
valid_dataset = TranslationDataset(val_data['src'], val_data['tgt'])
test_dataset = TranslationDataset(test_data['src'], test_data['tgt'])

#####################################################################
#####################################################################

# Định nghĩa tỷ lệ trích xuất từ valid_dataset và test_dataset
portion_valid_to_train = 0.8
portion_test_to_train = 0.5

# Trích xuất một phần từ valid_dataset
num_valid_to_train = int(len(valid_dataset) * portion_valid_to_train)
remaining_valid, valid_to_train = random_split(valid_dataset, [len(valid_dataset) - num_valid_to_train, num_valid_to_train])

# Trích xuất một phần từ test_dataset
num_test_to_train = int(len(test_dataset) * portion_test_to_train)
remaining_test, test_to_train = random_split(test_dataset, [len(test_dataset) - num_test_to_train, num_test_to_train])

# Hợp nhất các phần trích xuất vào train_dataset
train_dataset = ConcatDataset([train_dataset, valid_to_train, test_to_train])
#####################################################################
#####################################################################

# Khai báo số lượng câu trong 1 train, validation và test batch
train_batch_size = 32
test_batch_size = 8

train_loader = DataLoader(train_dataset, batch_size= train_batch_size, collate_fn= collate_batch)
val_loader = DataLoader(valid_dataset, batch_size= test_batch_size, collate_fn= collate_batch)
test_loader = DataLoader(test_dataset, batch_size= test_batch_size, collate_fn= collate_batch)


