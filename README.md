##LR文本分类器
分别使用Bag of Words和TFIDF作为文本特征,训练逻辑斯蒂回归分类器.
###依赖
需要numpy,scipy,sklearn,jieba分词
###输入格式
把要训练的文本放入data文件夹,根据训练集和测试集分别放入train和test.每个文件夹放一类文本,文件夹名即为类名.

格式如下.
```python
data
├── test
│   ├── A_宾馆饭店
│   │   ├── bj6_seg_pos.txt
│   │   ├── 三亚市春节宾馆房价不乱涨价违者将受到严处_seg_pos.txt
│   │   └── 住宿-宾馆名录_seg_pos.txt
│   ├── B_城市概况
│   │   ├── bozhou02_seg_pos.txt
│   │   ├── yangzhou01_seg_pos.txt
│   │   └── zhaoqing04_seg_pos.txt
│   ├── C_地方文化
│   .........
│   .........
└── train
    ├── A_宾馆饭店
    │   ├── bj1.txt
    │   ├── 魏宝山景区.txt
    │   └── 龙 潭 瀑 布.txt
    │   .........
    │   .........
    └── H_休闲娱乐
        ├── banna01.txt
        └── 金牌、银牌表示推荐的娱乐场所。如果想了解娱乐场所的详细信息，请点击娱乐场所名称。.txt

```
###训练和使用
这里使用的数据是随便找的旅游文本数据,可以换成其它的,文件夹格式参考上面的.
####Bag of Words
Bag of Words训练使用demo_bow.py
```python
import os
import numpy as np
from sklearn import linear_model
from TextClassify import BagOfWords,TextClassify
#数据目录
data_dir = 'data'
## BAG OF WORDS MODEL,根据数据建立词袋模型
BOW = BagOfWords(os.path.join(data_dir, 'train'))
##模型可以保存,以后直接读取
#BOW.build_dictionary()
#BOW.save_dictionary(os.path.join(data_dir, 'dicitionary.pkl'))
BOW.load_dictionary(os.path.join(data_dir, 'dicitionary.pkl'))

## LOAD DATA,将训练数据和测试数据转为词袋特征
train_feature, train_target = BOW.transform_data(os.path.join(data_dir, 'train'))
test_feature, test_target = BOW.transform_data(os.path.join(data_dir, 'test'))

## TRAIN LR MODEL,训练逻辑斯蒂模型
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(train_feature, train_target)

## PREDICT,在测试集上进行预测
test_predict = logreg.predict(test_feature)

## ACCURACY,测试集上正确率91%
true_false = (test_predict==test_target)
accuracy = np.count_nonzero(true_false)/float(len(test_target))
print "accuracy is %f" % accuracy

## TextClassify,对新文本进行预测,测试文件是test.txt
TextClassifier = TextClassify()
pred = TextClassifier.text_classify('test.txt', BOW, logreg)
print pred[0]
```
###运行结果
```
loaded dictionary from data/dicitionary.pkl
done
transforming data in to bag of words vector
done
transforming data in to bag of words vector
done
accuracy is 0.912500
D_购物美食
```
####TFIDF
TFIDF练使用demo_tfidf.py