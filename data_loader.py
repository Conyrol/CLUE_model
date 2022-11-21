import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *
import numpy as np
from collections import defaultdict

from create_dataset import Data_Reader, PAD
from utils import to_gpu, change_to_classify
import pickle

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

def return_unk():
    return UNK

class MSADataset(Dataset):
    def __init__(self, config):
        print(config.mode)
        if config.mode == 'train':
            data = load_pickle('/home/sunteng/CLUE_model2/train.pkl')
        if config.mode == 'dev':
            data = load_pickle('/home/sunteng/CLUE_model2/valid.pkl')
        if config.mode == 'test_iid':
            data = load_pickle('/home/sunteng/CLUE_model2/test_iid.pkl')
        if config.mode == 'test_ood':
            data = load_pickle('/home/sunteng/CLUE_model2/test_nn_text.pkl')
        
        # dataset = Data_Reader(config)
        # self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.pretrained_emb, self.word2id = torch.load('/home/share/sunteng/CLUE_model/datasets/MOSEI/original_dataset/embedding_and_mapping.pt')
        self.raw_text = data['raw_text']
        self.audio = data['audio']
        self.vision = data['vision']
        self.regression_labels = data['regression_labels']
        self.audio_lengths = data['audio_lengths']
        self.vision_lengths = data['vision_lengths']
        # self.text_bert = data['text_bert']
        self.len = len(self.audio)

        # config.visual_size = self.data[0][0][1].shape[1]
        # config.acoustic_size = self.data[0][0][2].shape[1]
        config.visual_size = self.vision[0].shape[1]
        config.acoustic_size = self.audio[0].shape[1]
        config.bert_text_size = 768
        
        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb

        self.config = config

    def __getitem__(self, index):
        return {
            "raw_text": self.raw_text[index],
            "audio": self.audio[index],
            "vision": self.vision[index],
            "regression_labels": self.regression_labels[index],
            'audio_lengths': self.audio_lengths[index],
            'vision_lengths': self.vision_lengths[index],
            "index": index
        }

    def __len__(self):
        return self.len


def get_loader(config, shuffle = True, ban_word_list = []):
    """Load DataLoader"""

    dataset = MSADataset(config)
    config.data_len = len(dataset)

    # def getText(txt):
    #     txt = txt.lower()
    #     for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
    #         txt = txt.replace(ch, " ")   #将文本中特殊字符替换为空格
    #     return txt

    def collate_fn(batch):
        # for later use we sort the batch in descending order of length
        # batch = sorted(batch, key=lambda x: x['data'][0][0].shape[0], reverse=True)
        
        index = []
        for sample in batch :
            index.append(sample['index']) 
        index = torch.LongTensor(index)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        #labels = [sample['regression_labels'] for sample in batch]
        labels = torch.cat([torch.from_numpy(np.array([sample['regression_labels']])) for sample in batch], dim=0)
        # sentences = pad_sequence([torch.LongTensor(sample['data'][0][0]) for sample in batch], padding_value = PAD)
        visual = torch.FloatTensor([sample['vision'] for sample in batch])
        acoustic = torch.FloatTensor([sample['audio'] for sample in batch])
        # segment = [sample['data'][2] for sample in batch]

        ## BERT-based features input prep
        # SENT_LEN = sentences.size(0)
        text_max_length = -1
        for sample in batch:
            text_max_length = max(len(sample['raw_text'].split()), text_max_length)
        SENT_LEN = text_max_length
        # Create bert indices using tokenizer
        bert_details = []
        text_list = []
        for sample in batch:
            text_list.append(sample['raw_text'])
            if config.base_model != 'magbert_model': NEW_SENT_LEN = SENT_LEN + 2
            else: NEW_SENT_LEN = SENT_LEN
            encoded_bert_sent = bert_tokenizer.encode_plus(sample['raw_text'], max_length=NEW_SENT_LEN, add_special_tokens=True, pad_to_max_length=True)
            bert_details.append(encoded_bert_sent)
        
        batch_sentence_vector = []
        for sample in batch:
            words = sample['raw_text'].split()
            sentence_vector = []
            for word in words:
                word_id = dataset.word2id[word]
                sentence_vector.append(dataset.pretrained_emb[word_id])
            batch_sentence_vector.append(torch.stack(sentence_vector))

        sentences_vector = pad_sequence(batch_sentence_vector)

        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        # lengths are useful later in using RNNs
        lengths_emb = torch.LongTensor([len(sample['raw_text'].split()) for sample in batch])
        lengths = torch.LongTensor([len(sample['raw_text'].split()) for sample in batch])

        # lengths
        audio_lengths = torch.LongTensor([sample['audio_lengths'] for sample in batch])
        vision_lengths = torch.LongTensor([sample['vision_lengths'] for sample in batch])

        sample_data = {
            'index': index,

            'raw_text': text_list,
            'text': to_gpu(sentences_vector, gpu_id = config.gpu_id),
            'ban_text': to_gpu(sentences_vector, gpu_id = config.gpu_id),

            'audio': to_gpu(acoustic, gpu_id = config.gpu_id),
            'visual': to_gpu(visual, gpu_id = config.gpu_id),
            'labels_classify': to_gpu(change_to_classify(labels, config), gpu_id = config.gpu_id).squeeze(),
            'labels': to_gpu(labels, gpu_id = config.gpu_id).squeeze(),
            'lengths_emb': to_gpu(lengths_emb, gpu_id = config.gpu_id),
            'lengths': to_gpu(lengths, gpu_id = config.gpu_id),
            # 'segment': segment,

            'bert_sentences': to_gpu(bert_sentences, gpu_id = config.gpu_id),
            'bert_sentence_att_mask': to_gpu(bert_sentence_att_mask, gpu_id = config.gpu_id),
            'bert_sentence_types': to_gpu(bert_sentence_types, gpu_id = config.gpu_id),
        }

        return sample_data

    data_loader = DataLoader(
        dataset = dataset,
        batch_size = config.batch_size,
        shuffle = shuffle,
        collate_fn = collate_fn,
        drop_last = True,
    )
    print(config.mode, config.data_len)

    return data_loader, len(dataset)