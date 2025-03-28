import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn.functional as F

main_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(main_path, 'output')
os.makedirs(output_path, exist_ok=True)

def fetch_data():
    #加载数据集    
    sentences = pd.read_csv(os.path.join(main_path, 'datasetSentences.txt'), sep='\t', encoding='utf-8', names=['index', 'text'], skiprows=1)
    splits = pd.read_csv(os.path.join(main_path, 'datasetSplit.txt'), encoding='utf-8', names=['index', 'set_label'], skiprows=1)
    dictionary = pd.read_csv(os.path.join(main_path, 'dictionary.txt'), sep='|', encoding='utf-8', names=['phrase', 'id'])
    sentiments = pd.read_csv(os.path.join(main_path, 'sentiment_labels.txt'), sep='|', encoding='utf-8', names=['id', 'value'], skiprows=1)
        
    sentiments['class'] = sentiments['value'].apply(lambda x: min(int(x * 5), 4))
        
    merged_data = pd.merge(sentences, splits, on='index')#处理四个txt文本的信息，合并有效内容
    merged_data = pd.merge(merged_data, dictionary, left_on='text', right_on='phrase')
    merged_data = pd.merge(merged_data, sentiments[['id', 'class']], on='id')
        
    train_data = merged_data[merged_data['set_label'] == 1]#针对句子编号划分数据集
    val_data = merged_data[merged_data['set_label'] == 3]
    test_data = merged_data[merged_data['set_label'] == 2]
        
    return train_data, val_data, test_data


class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):#获取文本编号以及情感标签
        text = self.dataset.iloc[index]['text']
        label = self.dataset.iloc[index]['class']
        encoded = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')#对文本进行编码
        
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout_rate, padding_idx):
        # 结构：词嵌入层、卷积层、全连接层、池化层和dropout层
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convolutions = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids).unsqueeze(1)
        conv_results = [F.relu(conv(embedded)).squeeze(3) for conv in self.convolutions]
        pooled_results = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conv_results]
        concatenated = self.dropout(torch.cat(pooled_results, dim=1))
        return self.fc(concatenated)

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, padding_idx):
        #结构：词嵌入层、LSTM层、全连接层和dropout层
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        packed_output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, nhead, num_layers, dropout_rate, padding_idx):
        #结构：词嵌入层、transformer编码器、全连接层和dropout层
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_ids, attention_mask):
        mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))
        
        embedded = self.embedding(input_ids)
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=~attention_mask.bool())
        pooled_output = transformer_out[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs, device):
    best_loss = float('inf')#初始化最佳损失
    
    for epoch in range(epochs):
        model.train()#训练
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(input_ids, attention_mask)
            loss = loss_fn(predictions, labels)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        model.eval()#验证
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                predictions = model(input_ids, attention_mask)
                loss = loss_fn(predictions, labels)
                total_val_loss += loss.item()
        
        print(f'轮数：{epoch+1}: 训练损失：{total_train_loss/len(train_loader):.3f}, 验证损失：{total_val_loss/len(val_loader):.3f}')
        
        if total_val_loss < best_loss:
            best_loss = total_val_loss
            torch.save(model.state_dict(), os.path.join(output_path, f'model_{model.__class__.__name__}.pt'))

def train():
    device = torch.device('cuda')
    train_data, val_data, test_data = fetch_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#分词器
    
    train_dataset = TextDataset(train_data, tokenizer)#处理数据
    val_dataset = TextDataset(val_data, tokenizer)
    test_dataset = TextDataset(test_data, tokenizer)
    
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    vocab_size = tokenizer.vocab_size
    embed_dim = 256
    num_filters = 100
    filter_sizes = [2, 3, 4]
    num_classes = 5
    dropout_rate = 0.2
    padding_idx = tokenizer.pad_token_id
    
    hidden_dim = 256
    nhead = 8
    num_layers = 2

    loss_fn = nn.CrossEntropyLoss()

    
    models = {
        'CNN': CNN(vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout_rate, padding_idx),
        'RNN': RNN(vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, padding_idx),
        'Transformer': TransformerModel(vocab_size, embed_dim, num_classes, nhead, num_layers, dropout_rate, padding_idx)
    }

    for model_name, model in models.items():
        print(f"\n开始训练{model_name}模型:")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters())
        train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=10, device=device)

if __name__ == "__main__":
    train() 