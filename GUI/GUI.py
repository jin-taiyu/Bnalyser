import os
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QDockWidget, QFileDialog, QFrame,
                             QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QTabWidget, QVBoxLayout,
                             QWidget, QSizePolicy)
from processing.CorpusProcessor import CorpusProcessor
from sentiment.SenAnalyse import SenAnalyse
from topic.Theme import WordCloudGenerator

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Bnalyser')
        self.setStyleSheet('background-color: #E8E8E8;')

        # Sidebar
        sidebar = QDockWidget()
        sidebar.setFixedWidth(200)
        sidebar.setFeatures(QDockWidget.NoDockWidgetFeatures)
        sidebar.setStyleSheet('background-color: #D3D3D3;')

        # Left frame
        left_frame = QFrame()
        left_frame.setStyleSheet('background-color: #D3D3D3;')
        self.left_layout = QVBoxLayout()
        self.left_layout.setContentsMargins(5, 5, 5, 5)
        self.left_layout.setSpacing(10)

        # Buttons style
        option_style = 'background-color: #B0C4DE; margin: 5px; border: 1px solid #A9A9A9; border-radius: 10px;'
        option1_button = QPushButton("第一步 选择文件")
        option2_button = QPushButton("第二步 情感分析")
        option3_button = QPushButton("第三步 文本分类")
        buttons = [option1_button, option2_button, option3_button]
        for button in buttons:
            button.setFixedSize(180, 60)
            button.setStyleSheet(option_style)
            self.left_layout.addWidget(button)

        left_frame.setLayout(self.left_layout)
        sidebar.setWidget(left_frame)

        # Tabs and styles
        tabs = QTabWidget()
        tabs.setStyleSheet(
            'QTabWidget::tab-bar { alignment: center; }'
            'QTabBar::tab { background: #D3D3D3; border: 1px solid #A9A9A9; border-radius: 10px; padding: 5px; }'
            'QTabBar::tab:selected { background: #B0C4DE; }'
        )

        right_frame_style = 'background-color: #E8E8E8; border-radius: 10px; padding: 10px;'
        button_style = 'QPushButton {background-color: #D3D3D3; border-radius: 10px; box-shadow: 2px 2px 2px #888888;}'
        progress_bar_style = """
        QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #20C997;
            border-radius: 5px;
            width: 10px;  # 更细致的填充
        }
        """

        image_label_style = 'QLabel { border: 1px solid #A9A9A9; border-radius: 5px; box-shadow: 3px 3px 5px #888888; text-align: center; }'

        # Tab 1
        tab1 = QWidget()
        tab1_layout = QHBoxLayout()
        tab1.setLayout(tab1_layout)

        right_frame1 = QFrame()
        right_frame1.setStyleSheet(right_frame_style)

        select_button = QPushButton("点击选择文件", self)
        select_button.setStyleSheet(button_style)
        select_button.clicked.connect(self.choose_file)

        process_button = QPushButton("开始语料处理", self)
        process_button.setStyleSheet(button_style)
        process_button.clicked.connect(self.process_corpus)

        self.progress_bar_corpus = QProgressBar()
        self.progress_bar_corpus.setStyleSheet(progress_bar_style)

        right_layout1 = QHBoxLayout()
        right_layout1.addWidget(select_button)
        right_layout1.addStretch()
        right_layout1.addWidget(process_button)
        right_frame1.setLayout(right_layout1)

        right_layout1.addWidget(self.progress_bar_corpus)

        tab1_layout.addStretch()
        tab1_layout.addWidget(right_frame1)
        tab1_layout.addStretch()

        # Tab 2
        tab2 = QWidget()
        tab2_layout = QVBoxLayout()
        tab2.setLayout(tab2_layout)

        right_frame2 = QFrame()
        right_frame2.setStyleSheet(right_frame_style)

        btn_analyse = QPushButton('情感分析', self)
        btn_analyse.setStyleSheet(button_style)
        btn_analyse.clicked.connect(self.sentiment_analysis)

        self.progress_bar_sen = QProgressBar()
        self.progress_bar_sen.setStyleSheet(progress_bar_style)

        right_layout2_horizontal = QHBoxLayout()
        right_layout2_horizontal.addWidget(btn_analyse)
        right_layout2_horizontal.addStretch()
        right_layout2_horizontal.addWidget(self.progress_bar_sen)

        self.image_label_sen = QLabel()
        self.image_label_sen.setStyleSheet(image_label_style)
        self.image_label_sen.setScaledContents(False)
        self.image_label_sen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label_sen.setAlignment(Qt.AlignCenter)

        tab2_layout.addLayout(right_layout2_horizontal)
        tab2_layout.addWidget(self.image_label_sen)
        tab2_layout.setAlignment(Qt.AlignCenter)

        # Tab 3
        tab3 = QWidget()
        tab3_layout = QVBoxLayout()
        tab3.setLayout(tab3_layout)

        right_frame3 = QFrame()
        right_frame3.setStyleSheet(right_frame_style)

        btn_classify = QPushButton('开始文本分类', self)
        btn_classify.setStyleSheet(button_style)
        btn_classify.clicked.connect(self.text_classification)

        self.progress_bar_cla = QProgressBar()
        self.progress_bar_cla.setStyleSheet(progress_bar_style)

        right_layout3_horizontal = QHBoxLayout()
        right_layout3_horizontal.addWidget(btn_classify)
        right_layout3_horizontal.addStretch()
        right_layout3_horizontal.addWidget(self.progress_bar_cla)

        self.image_label_wc = QLabel()
        self.image_label_wc.setStyleSheet(image_label_style)
        self.image_label_wc.setScaledContents(False)
        self.image_label_wc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label_wc.setAlignment(Qt.AlignCenter)

        self.image_label_cla = QLabel()
        self.image_label_cla.setStyleSheet(image_label_style)
        self.image_label_cla.setScaledContents(False)
        self.image_label_cla.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label_cla.setAlignment(Qt.AlignCenter)

        tab3_layout.addLayout(right_layout3_horizontal)
        tab3_layout.addWidget(self.image_label_wc)
        tab3_layout.addWidget(self.image_label_cla)
        tab3_layout.setAlignment(Qt.AlignCenter)

        tabs.addTab(tab1, "第一步 选择文件")
        tabs.addTab(tab2, "第二步 情感分析")
        tabs.addTab(tab3, "第三步 文本分类")

        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_widget.setLayout(central_layout)
        central_layout.addWidget(tabs)

        self.setCentralWidget(central_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, sidebar)

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

        option_style = 'background-color: #9FB2CE; margin: 5px; border: 2px solid black;'
        option = self.left_layout.itemAt(0).widget()
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

        option_style = 'background-color: #9FB2CE; margin: 5px; border: 2px solid black;'
        option = self.left_layout.itemAt(1).widget()
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

        option_style = 'background-color: #9FB2CE; margin: 5px; border: 2px solid black;'
        option = self.left_layout.itemAt(2).widget()
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
