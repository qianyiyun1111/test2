import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 读取数据
data = pd.read_json("data/News_Category.json", lines=True)
# 数据集概览
print(data.info())
# 统计每个类别的数量
category_counts = data['category'].value_counts()

# 统计作者的数量
author_counts = data['authors'].value_counts()

# 统计每个类别的日期范围
category_date_range = data.groupby('category')['date'].agg(['min', 'max'])

# 统计每个类别中的平均标题长度
category_avg_headline_length = data.groupby('category')['headline'].apply(lambda x: x.str.len().mean())

# 统计每个类别中的平均描述长度
category_avg_short_description_length = data.groupby('category')['short_description'].apply(lambda x: x.str.len().mean())

# 输出统计结果
print("Category Counts:\n", category_counts)
print("\nAuthor Counts:\n", author_counts)
print("\nDate Counts:\n", category_date_range)
print("\nAverage Headline Length by Category:\n", category_avg_headline_length)
print("\nAverage Short_Description Length by Category:\n", category_avg_short_description_length)