from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

class WordCloudGenerator:
    def __init__(self, data, texts, results):
        self.texts = texts
        self.results = results
        self.data = data

    def load_data(self):
        analyse_data_path = os.path.join('data', 'analyse.json')
        # 读取json文件
        with open(analyse_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        return self.data

    def analyse(self):
        # 导入分词器和模型
        print("--------------------分类模型加载中--------------------")
        tokenizer = BertTokenizer.from_pretrained(os.path.join('model', 'topic', 'vocab.txt'))
        model = BertForSequenceClassification.from_pretrained(os.path.join('model', 'topic'))

        for text in self.data:
            # data赋值给text
            self.texts.append(text)

        print("准备开始分类预测...")
        for text in self.texts:
            # 进行预测
            output = model(torch.tensor([tokenizer.encode(text)]))
            result = torch.argmax(torch.nn.functional.softmax(output.logits,dim=-1), dim=1).item()
            labels = ['体育',
                      '财经',
                      '房产',
                      '家居',
                      '教育',
                      '科技',
                      '时尚',
                      '时政',
                      '游戏',
                      '娱乐']

            print(text)
            print(labels[result])
            print("\n")
            self.results.append(result)

    def chart(self):
        # 统计每个类别的数量
        counts = [0] * 10
        for result in self.results:
            counts[result] += 1

        labels = ['sports', 'finance', 'real estate', 'home', 'education', 'technology', 'fashion', 'politics', 'game',
                  'entertainment']

        sns.set(style="whitegrid")  # Set the style of the plot
        plt.figure(figsize=(12, 8))  # Set the size of the figure

        # 使用更淡色的现代化配色方案
        colors = sns.color_palette('Set2')

        # Plot the bar chart
        ax = sns.barplot(x=labels, y=counts, palette=colors)

        # Add the data labels on top of the bars
        for p, label in zip(ax.patches, counts):
            ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 0.1, label, ha="center", va="bottom", fontsize=12)

        # Set the labels and title
        ax.set_xlabel('Class', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('Classification Results', fontsize=16)
        plt.xticks(rotation=45)  # Rotate the x labels for better readability

        # Save the figure with 300 DPI
        plt.savefig(os.path.join('data', 'ClassificationResults.png'), dpi=600)

    def generate_wordcloud(self, vocab_json_path):
        with open(vocab_json_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            text = ' '.join(vocab)
            wordcloud = WordCloud(max_font_size=40, background_color='white', font_path="./data/Songti.ttc").generate(text)
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")

            # Save the word cloud with 300 DPI
            plt.savefig(os.path.join('data', 'WordCloud.png'), dpi=600)

            print("文本分类结束")