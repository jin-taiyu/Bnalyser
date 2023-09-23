import sys
import os
from processing.CorpusProcessor import CorpusProcessor
from sentiment.SenAnalyse import SenAnalyse
from topic.Theme import WordCloudGenerator
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QMessageBox, QLabel, QHBoxLayout, QVBoxLayout, QFrame, QTabWidget, QDockWidget, QProgressBar



class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # 设置窗口大小并居中
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Bnalyser')
        self.setStyleSheet('background-color: #F0FFFF;')

        ##############################################################
        ##############################################################
        # 创建侧边栏
        sidebar = QDockWidget()
        sidebar.setFixedWidth(200)
        sidebar.setFeatures(QDockWidget.NoDockWidgetFeatures)
        sidebar.setStyleSheet('background-color: #F5F5DC;')

        # 创建左侧框架
        left_frame = QFrame()
        left_frame.setStyleSheet('background-color: #F5F5DC;')
        self.left_layout = QVBoxLayout()
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(0)

        ##############################################################
        ##############################################################

        ##############################################################
        ##############################################################
        # 创建按钮
        option1_button = QPushButton("第一步 选择文件")
        option2_button = QPushButton("第二步 情感分析")
        option3_button = QPushButton("第三步 文本分类")

        # 将按钮与选项卡切换关联起来
        option1_button.clicked.connect(lambda: tabs.setCurrentIndex(0))
        option2_button.clicked.connect(lambda: tabs.setCurrentIndex(1))
        option3_button.clicked.connect(lambda: tabs.setCurrentIndex(2))

        # 添加按钮到布局中
        self.left_layout.addWidget(option1_button)
        self.left_layout.addWidget(option2_button)
        self.left_layout.addWidget(option3_button)
        left_frame.setLayout(self.left_layout)

        # 设置 option 的样式
        option_style = 'background-color: #FFE4C4; margin: 5px; border: 2px solid black;'
        for i in range(self.left_layout.count()):
            option = self.left_layout.itemAt(i).widget()
            option.setFixedSize(200, 200)
            option.setStyleSheet(option_style)

        # 将左侧框架添加到侧边栏中
        sidebar.setWidget(left_frame)

        # 创建选项卡
        tabs = QTabWidget()

        # 添加选项卡1
        tab1 = QWidget()
        tab1_layout = QHBoxLayout()
        tab1.setLayout(tab1_layout)

        # 添加选项卡2
        tab2 = QWidget()
        tab2_layout = QHBoxLayout()
        tab2.setLayout(tab2_layout)

        # 添加选项卡3
        tab3 = QWidget()
        tab3_layout = QHBoxLayout()
        tab3.setLayout(tab3_layout)

        ##############################################################
        ##############################################################

        ###############################框架1###########################
        # 创建右侧框架1
        right_frame1 = QFrame()
        right_frame1.setStyleSheet('background-color: #F0FFFF;')

        # 创建按钮
        select_button = QPushButton("点击选择文件", self)
        select_button.clicked.connect(self.choose_file)

        process_button = QPushButton("开始语料处理", self)
        process_button.clicked.connect(self.process_corpus)

        # 创建进度条
        self.progress_bar_corpus = QProgressBar()

        # 添加按钮到框架中
        right_layout1 = QHBoxLayout()
        right_layout1.addWidget(select_button)
        right_layout1.addStretch()
        right_layout1.addWidget(process_button)
        right_frame1.setLayout(right_layout1)

        # 添加进度条到框架中
        right_layout1.addWidget(self.progress_bar_corpus)

        # 将框架添加到选项卡布局中
        tab1_layout.addStretch()
        tab1_layout.addWidget(right_frame1)
        tab1_layout.addStretch()
        # 设置布局对齐方式为顶部对齐
        right_layout1.setAlignment(Qt.AlignTop)

        ###############################框架2###########################
        # 创建右侧框架2
        right_frame2 = QFrame()
        right_frame2.setStyleSheet('background-color: #F0FFFF;')

        # 创建“情感分析”按钮并将其添加到水平布局中
        btn_analyse = QPushButton('情感分析', self)
        btn_analyse.clicked.connect(self.sentiment_analysis)

        # 创建进度条
        self.progress_bar_sen = QProgressBar()

        # 创建一个 QLabel 控件用于显示图像
        self.image_label_sen = QLabel()
        # 设置 QLabel 的尺寸策略为保持纵横比，缩放以适应
        self.image_label_sen.setScaledContents(True)

        # 创建垂直布局
        right_layout2 = QVBoxLayout()

        # 将按钮添加到布局中，并设置对齐方式为顶端对齐和居中对齐
        right_layout2.addWidget(btn_analyse, alignment=Qt.AlignTop | Qt.AlignHCenter)
        # 添加进度条到框架中
        right_layout2.addWidget(self.progress_bar_sen, alignment=Qt.AlignTop | Qt.AlignHCenter)
        # 将 QLabel 控件添加到垂直布局中
        right_layout2.addWidget(self.image_label_sen, alignment=Qt.AlignTop | Qt.AlignHCenter)

        # 将布局设置到框架中
        right_frame2.setLayout(right_layout2)

        # 将框架添加到选项卡布局中
        tab2_layout.addStretch()
        tab2_layout.addWidget(right_frame2)
        tab2_layout.addStretch()

        ###############################框架3###########################
        # 创建右侧框架3
        right_frame3 = QFrame()
        right_frame3.setStyleSheet('background-color: #F0FFFF;')

        # 创建“文本分类”按钮并将其添加到水平布局中
        btn_classify = QPushButton('开始文本分类', self)
        btn_classify.clicked.connect(self.text_classification)

        # 创建进度条
        self.progress_bar_cla = QProgressBar()

        # 创建一个 QLabel 控件用于显示图像
        self.image_label_wc = QLabel()
        # 设置 QLabel 的尺寸策略为保持纵横比，缩放以适应
        self.image_label_wc.setScaledContents(True)

        # 创建一个 QLabel 控件用于显示图像
        self.image_label_cla = QLabel()
        # 设置 QLabel 的尺寸策略为保持纵横比，缩放以适应
        self.image_label_cla.setScaledContents(True)

        # 创建垂直布局
        right_layout3 = QVBoxLayout()

        # 将按钮添加到布局中，并设置对齐方式为顶端对齐和居中对齐
        right_layout3.addWidget(btn_classify, alignment=Qt.AlignTop | Qt.AlignHCenter)
        # 添加进度条到框架中
        right_layout3.addWidget(self.progress_bar_cla, alignment=Qt.AlignTop | Qt.AlignHCenter)
        # 将 QLabel 控件添加到垂直布局中
        right_layout3.addWidget(self.image_label_wc, alignment=Qt.AlignTop | Qt.AlignHCenter)
        right_layout3.addWidget(self.image_label_cla, alignment=Qt.AlignTop | Qt.AlignHCenter)

        # 将布局设置到框架中
        right_frame3.setLayout(right_layout3)

        # 将框架添加到选项卡布局中
        tab3_layout.addStretch()
        tab3_layout.addWidget(right_frame3)
        tab3_layout.addStretch()


        tabs.addTab(tab1, "第一步 选择文件")
        tabs.addTab(tab2, "第二步 情感分析")
        tabs.addTab(tab3, "第三步 文本分类")

        # 创建主窗口布局
        layout = QVBoxLayout()
        layout.addWidget(tabs)

        # 创建窗口部件
        central_widget = QWidget()
        central_widget.setLayout(layout)

        # 将侧边栏和主窗口添加到窗体
        self.setCentralWidget(central_widget)
        self.addDockWidget(1, sidebar)

        self.show()


    ############################################################
    ############################################################
    ############################################################

    def choose_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Text Files (*.txt);;CSV Files (*.csv)",
                                                   options=options)
        if file_path:
            # 获取文件扩展名
            extension = file_path.split(".")[-1]
            if extension == "txt":
                # 选择的是txt文件
                self.file_path = file_path
                self.csv_yes = False
            elif extension == "csv":
                # 选择的是csv文件
                self.file_path = file_path
                self.csv_yes = True
            else:
                # 不支持的文件类型
                self.file_path = None
                # 显示错误消息框
                QMessageBox.warning(self, "错误", "不支持的文件类型！请选择txt或csv文件。")

    import os

    def process_corpus(self):

        # 实例化CorpusProcessor对象，并执行预处理操作
        corpus_processor = CorpusProcessor(self.file_path)

        # 判断是否选择文件类型，并依据文件判断是否需要对文件进行处理
        if self.csv_yes:
            corpus_processor.load_csv()
            percentage = int((1 / 5) * 100)
            self.update_progress_bar_corpus(percentage)
            stop_words_path = os.path.join('data', 'stop_words.txt')
            corpus_processor.load_stop_words(stop_words_path)
            percentage = int((2 / 5) * 100)
            self.update_progress_bar_corpus(percentage)
            analyse_path = os.path.join('data', 'analyse.json')
            corpus_processor.save_corpus(analyse_path)
            percentage = int((3 / 5) * 100)
            self.update_progress_bar_corpus(percentage)
            corpus_processor.remove_stop_words()
            percentage = int((4 / 5) * 100)
            self.update_progress_bar_corpus(percentage)
            vocab_path = os.path.join('data', 'vocab.json')
            corpus_processor.save_vocab(vocab_path)
            percentage = int((5 / 5) * 100)
            self.update_progress_bar_corpus(percentage)
        else:
            corpus_processor.load_corpus()
            percentage = int((1 / 5) * 100)
            self.update_progress_bar_corpus(percentage)
            stop_words_path = os.path.join('data', 'stop_words.txt')
            corpus_processor.load_stop_words(stop_words_path)
            percentage = int((2 / 5) * 100)
            self.update_progress_bar_corpus(percentage)
            analyse_path = os.path.join('data', 'analyse.json')
            corpus_processor.save_corpus(analyse_path)
            percentage = int((3 / 5) * 100)
            self.update_progress_bar_corpus(percentage)
            corpus_processor.remove_stop_words()
            percentage = int((4 / 5) * 100)
            self.update_progress_bar_corpus(percentage)
            vocab_path = os.path.join('data', 'vocab.json')
            corpus_processor.save_vocab(vocab_path)
            percentage = int((5 / 5) * 100)
            self.update_progress_bar_corpus(percentage)

        option_style = 'background-color: #FFA07A; margin: 5px; border: 2px solid black;'
        option = self.left_layout.itemAt(0).widget()
        option.setFixedSize(200, 200)
        option.setStyleSheet(option_style)



    def sentiment_analysis(self):

        # 实例化SenAnalyse对象，并执行情感分析操作
        sen_analyse = SenAnalyse([], [], [], 0, 0)
        percentage = int((1 / 5) * 100)
        self.update_progress_bar_sen(percentage)
        sen_analyse.load_data()
        percentage = int((2 / 5) * 100)
        self.update_progress_bar_sen(percentage)
        sen_analyse.analyse()
        percentage = int((3 / 5) * 100)
        self.update_progress_bar_sen(percentage)
        sen_analyse.chart()
        percentage = int((4 / 5) * 100)
        self.update_progress_bar_sen(percentage)


        # 加载图片并显示
        image_path = os.path.join('data', 'SenAnalyse.png')
        pixmap = QPixmap(image_path)
        pixmap_resized = pixmap.scaled(320, 240, QtCore.Qt.KeepAspectRatio)
        image = pixmap_resized.toImage()
        self.image_label_sen.setPixmap(pixmap_resized)
        percentage = int((5 / 5) * 100)
        self.update_progress_bar_sen(percentage)

        option_style = 'background-color: #FFA07A; margin: 5px; border: 2px solid black;'
        option = self.left_layout.itemAt(1).widget()
        option.setFixedSize(200, 200)
        option.setStyleSheet(option_style)

    def text_classification(self):

        # 实例化WordCloudGenerator对象，并执行文本分类操作
        word_cloud_generator = WordCloudGenerator([], [], [])
        percentage = int((1 / 5) * 100)
        self.update_progress_bar_cla(percentage)

        word_cloud_generator.load_data()
        percentage = int((2 / 5) * 100)
        self.update_progress_bar_cla(percentage)

        word_cloud_generator.analyse()
        percentage = int((3 / 5) * 100)
        self.update_progress_bar_cla(percentage)

        word_cloud_generator.chart()
        percentage = int((4 / 5) * 100)
        self.update_progress_bar_cla(percentage)

        percentage = int((5 / 5) * 100)
        self.update_progress_bar_cla(percentage)

        wordcloud_path = os.path.join('data', 'WordCloud.png')
        classification_results_path = os.path.join('data', 'ClassificationResults.png')

        word_cloud_generator.generate_wordcloud(os.path.join('data', 'vocab.json'))

        # 加载图片并显示
        pixmap = QPixmap(wordcloud_path)
        pixmap_resized = pixmap.scaled(320, 240, QtCore.Qt.KeepAspectRatio)
        image = pixmap_resized.toImage()
        self.image_label_wc.setPixmap(pixmap_resized)

        # 加载分类图片并显示
        pixmap = QPixmap(classification_results_path)
        pixmap_resized = pixmap.scaled(500, 300, QtCore.Qt.KeepAspectRatio)
        image = pixmap_resized.toImage()
        self.image_label_cla.setPixmap(pixmap_resized)

        option_style = 'background-color: #FFA07A; margin: 5px; border: 2px solid black;'
        option = self.left_layout.itemAt(2).widget()
        option.setFixedSize(200, 200)
        option.setStyleSheet(option_style)

    def update_progress_bar_corpus(self, value_corpus):
        self.progress_bar_corpus.setValue(value_corpus)

    def update_progress_bar_sen(self, value_sen):
        self.progress_bar_sen.setValue(value_sen)

    def update_progress_bar_cla(self, value_cla):
        self.progress_bar_cla.setValue(value_cla)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())
