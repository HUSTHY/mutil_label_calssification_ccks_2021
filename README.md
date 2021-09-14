![](https://img.shields.io/badge/Python-3.7.5-blue)
![](https://img.shields.io/badge/torch-1.8.0-green)
![](https://img.shields.io/badge/transformers-4.5.1-green)

<h3 align="center">
<p>A PyTorch implementation of mutil_label text classification </p>
</h3>

```
其中损失函数采用了sigmoid+BCE
苏神提出的——multilabel_crossentropy
```

[将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359/comment-page-1#comments)


###  用法


```bash
python train_bert_mutillabel_classification.py
```


#### 结果
```
BCE     train_acc:1.0000 val_acc:0.9720------best_acc:0.9720
MLCE    train_acc:1.0000 val_acc:0.9790------best_acc:0.9790
```



