运行说明
1.数据统计：
运行statistic.py。

2.模型训练：
a.先运行nltk_install.py下载停用词库和分词器。
b.然后运行process.py对训练数据进行分词/去除停用词的预处理。
c.最后运行train_model.py进行模型训练，模型、label encoder和tokenizer保存在data文件夹中。

3.单次测试模型使用(输入headline, authors, link, des, date输出category)：
在single_test.py中，修改test_headline,test_authors,test_link,test_description,test_date为输入数据并运行,输出预测category结果。

4.多次测试模型使用(输入含headline, authors, link, des, date的json文件，输出含预测category的json文件)：
a.如果作为输入的json文件中将short_description改为des，则修改process.py中"data['processed_text'] = data['headline'] + ' ' + data['authors'] + ' ' + data['short_description'] + ' ' + data['link']"一句中的'short_description'为'des'。
b.在process.py中，修改input_path为作为输入的json文件地址，修改output_path并运行。
c.在predict.py中，修改input_path为上一步中修改的output_path,修改output_path为所需输出地址并运行，结果保存在output_path中。