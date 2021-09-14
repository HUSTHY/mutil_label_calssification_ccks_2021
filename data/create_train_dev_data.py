import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_excel('train_processed.xlsx')
    df['label_text'] = df['答案类型'].str.cat(df['属性名'],sep='|')
    label_texts = df['label_text'].values.tolist()
    label_texts = [ele.strip('|').split("|") for ele in label_texts]
    set_labels = []
    for ele in label_texts:
        set_labels.extend(ele)
    set_labels = list(set(set_labels))
    print(set_labels)
    labels = []
    for ele in label_texts:
        temp_label = []
        for i in range(len(set_labels)):
            if set_labels[i] in ele:
                temp_label.append(str(1))
            else:
                temp_label.append(str(0))
        temp_label = ' '.join(temp_label)
        labels.append(temp_label)

    with open('labels.txt','w',encoding='utf-8') as f:
        for label in set_labels:
            f.write(label+'\n')

    set_labels = '|'.join(set_labels)

    df['labels'] = labels
    df['set_label'] = [set_labels]*len(df)
    train_df, dev_df = train_test_split(df,test_size=1000/len(df))
    
    train_writer = pd.ExcelWriter('train.xlsx')
    train_df.to_excel(train_writer,index=False)
    train_writer.save()

    dev_writer = pd.ExcelWriter('dev.xlsx')
    dev_df.to_excel(dev_writer, index=False)
    dev_writer.save()