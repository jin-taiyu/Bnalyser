from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

class SenAnalyse:
    def __init__(self, data, texts, results, negative, positive):
        self.texts = texts
        self.results = results
        self.negative = negative
        self.positive = positive
        self.data = data


    def load_data(self):
        data_path = os.path.join('data', 'analyse.json')
        # 读取json文件
        with open(data_path) as f:
            self.data = json.load(f)
        return self.data

    def analyse(self):
        # 导入分词器和模型
        print("--------------------情感分析模型加载中--------------------")
        tokenizer = BertTokenizer.from_pretrained(os.path.join('model', 'sentiment', 'vocab.txt'))
        model = BertForSequenceClassification.from_pretrained(os.path.join('model', 'sentiment'))

        print("准备开始情感分析...")
        for text in self.data:
            # data赋值给text
            self.texts.append(text)
        for text in self.texts:
            # 进行预测
            output = model(torch.tensor([tokenizer.encode(text)]))
            result = torch.argmax(torch.nn.functional.softmax(output.logits,dim=-1), dim=1).item()
            self.results.append(result)
            if result == 0:
                self.negative = self.negative + 1
            else:
                self.positive = self.positive + 1

            labels = ['消极', '积极']
            print(text)
            print(labels[result])
            print("\n")


    def chart(self):
        # 模拟数据，positive为积极情感数量，negative为消极情感数量
        # 绘制饼图
        labels = ['positive', 'negative']
        sizes = [self.positive, self.negative]

        fig1, ax1 = plt.subplots()
        ax1.set_title("Sentiment Analysis")

        colors = sns.color_palette('pastel')[0:len(sizes)]
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                shadow=True, textprops={'fontsize': 14}, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

        ax1.axis('equal')

        # 保存图片
        img_path = os.path.join('data', 'SenAnalyse.png')
        plt.savefig(img_path)

        print("情感分析结束")
