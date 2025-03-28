import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
from train import TextDataset, CNN, RNN, TransformerModel, fetch_data
import warnings

warnings.filterwarnings('ignore')

main_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(main_path, 'output')

def test_model(model, test_loader, device):
    model.eval()
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)#预测
            _, predicted = torch.max(outputs, 1)#获取预测结果
            
            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())#收集预测结果和真实标签
    
    return accuracy_score(true_labels, preds)

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, val_data, test_data = fetch_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#分词器

    test_dataset = TextDataset(test_data, tokenizer)#处理数据
    test_loader = DataLoader(test_dataset, batch_size=8)#加载数据

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

    models = {#初始化模型
        'CNN': CNN(vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout_rate, padding_idx),
        'RNN': RNN(vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, padding_idx),
        'Transformer': TransformerModel(vocab_size, embed_dim, num_classes, nhead, num_layers, dropout_rate, padding_idx)
    }

    for model_name, model in models.items():
        model_path = os.path.join(output_path, f'model_{model.__class__.__name__}.pt')
        if os.path.exists(model_path):
            model = model.to(device)
            model.load_state_dict(torch.load(model_path))
            accuracy = test_model(model, test_loader, device)
            print(f"\n{model_name}模型准确率: {accuracy:.4f}")

if __name__ == "__main__":
    test()
    

