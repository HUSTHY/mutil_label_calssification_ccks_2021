import torch.nn as nn
from transformers import BertModel
from transformers import BertPreTrainedModel
import torch
class MutilLabelClassification(BertPreTrainedModel):
    def __init__(self,config,max_len):
        super(MutilLabelClassification,self).__init__(config)
        self.max_len = max_len
        self.bert = BertModel(config=config)
        self.classifier = nn.Linear(config.hidden_size,config.num_labels)


    def forward(self,inputs):
        output = self.bert(**inputs,return_dict=True, output_hidden_states=True)
        #采用最后一层
        embedding = output.hidden_states[-1]
        embedding = self.pooling(embedding,inputs)
        output = self.classifier(embedding)
        return output



    def pooling(self,token_embeddings,input):
        output_vectors = []
        #attention_mask
        attention_mask = input['attention_mask']
        #[B,L]------>[B,L,1]------>[B,L,768],矩阵的值是0或者1
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #这里做矩阵点积，就是对元素相乘(序列中padding字符，通过乘以0给去掉了)[B,L,768]
        t = token_embeddings * input_mask_expanded
        #[B,768]
        sum_embeddings = torch.sum(t, 1)

        # [B,768],最大值为seq_len
        sum_mask = input_mask_expanded.sum(1)
        #限定每个元素的最小值是1e-9，保证分母不为0
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        #得到最后的具体embedding的每一个维度的值——元素相除
        output_vectors.append(sum_embeddings / sum_mask)

        #列拼接
        output_vector = torch.cat(output_vectors, 1)

        return  output_vector



    def encoding(self,inputs):
        self.bert.eval()
        with torch.no_grad():
            output = self.bert(**inputs, return_dict=True, output_hidden_states=True)
            embedding = output.hidden_states[-1]
            embedding = self.pooling(embedding, inputs)
        return embedding

