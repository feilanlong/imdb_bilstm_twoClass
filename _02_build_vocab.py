# -*-coding:utf-8-*-
import os
import pickle
import re
import zipfile

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from _01_vocab import Vocab

class ImdbDataset(Dataset):
    def __init__(self, train=True, sequence_max_len=100):
        if not os.path.exists("./data/download"):
            unzip_file("./data/test.zip", "./data/download")
            unzip_file("./data/train.zip", "./data/download")
        self.sequence_max_len = sequence_max_len
        data_path = r"./data/download"
        data_path += r"/train" if train else r"/test"
        self.total_path = []  # 保存所有的文件路径
        # self.voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
        for temp_path in [r"/pos", r"/neg"]:
            cur_path = data_path + temp_path
            self.total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]

    def __getitem__(self, idx):
        file = self.total_path[idx]
        # 从txt获取评论并分词
        review = tokenlize(open(file, "r", encoding="utf-8").read())
        return review

    def __len__(self):
        return len(self.total_path)

    # def get_num_embeddings(self):
    #     return len(self.voc_model)
    #
    # def get_padding_idx(self):
    #     return self.voc_model.PAD


def tokenlize(sentence):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """

    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<br />", " ", sentence)
    # sentence = re.sub("I'm","I am",sentence)
    # sentence = re.sub("isn't","is not",sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]

    return result


def unzip_file(zip_src, dst_dir):
    """
    解压缩
    :param zip_src:
    :param dst_dir:
    :return:
    """
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        bar = tqdm(fz.namelist())
        bar.set_description("unzip  " + zip_src)
        for file in bar:
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')



def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    # reviews, labels = zip(*batch)

    return batch

def get_dataloader(train=True):
 
    imdb_dataset = ImdbDataset(train, sequence_max_len=100)

    my_dataloader = DataLoader(imdb_dataset, batch_size=200, shuffle=True, collate_fn=collate_fn)
    return my_dataloader

def build_vocab(): #这里只是构建训练集的词典
    ws = Vocab()
    dl_train = get_dataloader(True)
    for reviews in tqdm(dl_train, total=len(dl_train)):
        for sentence in reviews:
            ws.fit(sentence)

    ws.build_vocab()
    print(len(ws))
    if not os.path.exists("./models"):
        os.makedirs("./models")
        pickle.dump(ws, open("./models/vocab.pkl", "wb"))


