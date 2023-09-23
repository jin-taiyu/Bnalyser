import csv
import json
import jieba
import codecs
import os

# 定义一个类 CorpusProcessor 用于对语料进行预处理
class CorpusProcessor:
    # 初始化 此处形参为源语料文件路径
    def __init__(self, file_path):
        self.file_path = file_path
        self.corpus = []
        self.vocab = set()
        self.stop_words = set()

    def load_csv(self):
        with open(self.file_path, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                self.corpus.append(row['正文'])
        print("开始csv语料处理")

    # 加载语料 读取 存储在一个list中
    def load_corpus(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.corpus.append(line.strip())

        print("开始txt语料处理")

    # 加载停用词
    def load_stop_words(self, stop_words_file):
        with codecs.open(stop_words_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.stop_words.add(line.strip())


    # 将list存储于json文件中 corpus_file_name
    def save_corpus(self, corpus_file_name):
        # 创建路径
        if os.path.exists(corpus_file_name):
            with open(corpus_file_name, 'w', encoding='utf-8') as f:
                json.dump(self.corpus, f, ensure_ascii=False)
        else:
            os.path.join('./',corpus_file_name)
            with open(corpus_file_name, 'w', encoding='utf-8') as f:
                json.dump(self.corpus, f, ensure_ascii=False)

        print("已处理语料存储")

    def remove_stop_words(self):
        self.corpus = [[word for word in jieba.cut(doc) if word not in self.stop_words] for doc in self.corpus]

    def save_vocab(self, vocab_file_name):
        if os.path.exists(vocab_file_name):
            for text in self.corpus:
                text = "".join(text)
                words = jieba.lcut(text)
                self.vocab.update(words)
            with open(vocab_file_name, 'w', encoding='utf-8') as f:
                json.dump(list(self.vocab), f, ensure_ascii=True)
        else:
            os.path.join('./',vocab_file_name)
            for text in self.corpus:
                words = jieba.lcut(text)
                self.vocab.update(words)
            with open(vocab_file_name, 'w', encoding='utf-8') as f:
                json.dump(list(self.vocab), f, ensure_ascii=True)

        print("词汇已存储")
