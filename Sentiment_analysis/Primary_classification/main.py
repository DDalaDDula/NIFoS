import torch
import pandas as pd
from transformers import ElectraTokenizer
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import load_model
from train_eval import train, evaluate
from visualize import plot_train_loss_accuracy
from evaluate import print_evaluation_metrics

# GPU를 사용할 수 있다면 GPU로, 아니라면 CPU로 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 예시 데이터로 대체 (이후 실제 데이터셋으로 변경해야 합니다)
dataset = pd.read_csv("C:/Users/user/Python/sentiment_analyze/NIFoS/Sentiment_analysis/dataset/primary_classification_data.csv", encoding='UTF-8')
# ElectraTokenizer 로드
tokenizer = ElectraTokenizer.from_pretrained("koelectra-base-v3")

# 데이터셋 준비
Max_length = 128
batch_size = 32

custom_dataset = CustomDataset(dataset, tokenizer, Max_length)
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# 모델 로드
num_labels = 3
model = load_model(num_labels)

# 옵티마이저 및 로스 함수 정의
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 훈련
num_epochs = 5
train_losses, train_accuracy = train(model, data_loader, optimizer, loss_fn, num_epochs)

# 시각화
plot_train_loss_accuracy(train_losses, train_accuracy)

# 테스트 및 평가
test_accuracy_mean, true_labels, predicted_labels = evaluate(model, data_loader)
print_evaluation_metrics(test_accuracy_mean, true_labels, predicted_labels)
