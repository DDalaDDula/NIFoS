import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# 커스텀 데이터셋 클래스 설정
class CustomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
        }

# dataset split 함수 설정
def dataset_split(padding_length, sentence_list, labels, modelpath, test_size): # padding_length, 문장리스트, 라벨리스트, 모델경로, test데이터셋 분할비율
    # koELECTRA 토크나이저 불러오기
    tokenizer = ElectraTokenizer.from_pretrained(modelpath)

    # 문장을 토큰화하고 시퀀스로 변환
    sequences = [tokenizer.encode(sentence, padding=padding_length, max_length=padding_length, truncation=True) for sentence in tqdm(sentence_list)]

    # 학습 데이터와 검증 데이터로 분할
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=test_size, random_state=42
    )

    # 데이터셋과 데이터로더 생성
    train_dataset = CustomDataset(train_sequences, train_labels, tokenizer, max_length=60)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)

    val_dataset = CustomDataset(val_sequences, val_labels, tokenizer, max_length=60)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader

# 모델 로드하고 GPU 가속을 활성화 하는 함수 설정
def load_model(modelpath, num_labels, model_state_path=None, optimizer_state_path=None, checkpoint_path=None, new_dropout_rate=0.2): # 모델경로, label수, 모델 dict 경로, 옵티마이저 dict 경로, 모델 체크포인트 경로, dropout비율
    model = ElectraForSequenceClassification.from_pretrained(modelpath, num_labels=num_labels)

    # 옵티마이저와 손실 함수 설정   
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 모델의 상태 딕셔너리를 로드하는 경우
    if model_state_path:
        model_state_dict = torch.load(model_state_path)
        model.load_state_dict(model_state_dict)

    # 옵티마이저의 상태 딕셔너리를 로드하는 경우
    if optimizer_state_path:
        optimizer_state_dict = torch.load(optimizer_state_path)
        optimizer.load_state_dict(optimizer_state_dict)

    # 모델 체크포인트를 로드하는 경우
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 장치 설정 (GPU 사용을 위해)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 모든 Dropout 레이어의 비율 변경
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = new_dropout_rate

    return model, optimizer, loss_fn, device

# 학습 함수 정의 (tqdm을 사용하여 진행 상황 및 지표 시각화)
def train_fn(data_loader, model, optimizer, loss_fn, device):
    model.train()
    progress_bar = tqdm(data_loader, desc="Training")

    train_accs = []  # train_accuracy 기록을 위한 리스트
    train_losses = []  # train_loss 기록을 위한 리스트

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        loss = loss_fn(logits, labels)  # 손실 함수 적용
        loss.backward()
        optimizer.step()

        # 정확도 계산
        predicted_labels = torch.argmax(logits, dim=1)
        accuracy = (predicted_labels == labels).float().mean().item()

        train_losses.append(loss.item())
        train_accs.append(accuracy)
        progress_bar.set_postfix({'Loss': loss.item(), 'Accuracy': accuracy})

    return train_accs, train_losses

# 평가 함수 정의
def eval_fn(data_loader, model, loss_fn, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0.0

    predicted_labels_list = []  # 예측한 라벨들을 저장하기 위한 리스트
    true_labels_list = []  # 실제 라벨들을 저장하기 위한 리스트

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)  # 손실 함수 적용
            total_loss += loss.item()

            predicted_labels = torch.argmax(logits, dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

            # 예측한 라벨과 실제 라벨을 리스트에 추가
            predicted_labels_list.extend(predicted_labels.tolist())
            true_labels_list.extend(labels.tolist())

    accuracy = correct_predictions / total_predictions
    avg_loss = total_loss / len(data_loader)

    # 예측한 라벨과 실제 라벨 출력
    predicted_labels_list = np.array(predicted_labels_list)
    true_labels_list = np.array(true_labels_list)
    # print("Predicted Labels:", predicted_labels_list)
    # print("True Labels:", true_labels_list)

    # f1 score, precision, recall 계산
    f1 = f1_score(true_labels_list, predicted_labels_list, average='macro')
    precision = precision_score(true_labels_list, predicted_labels_list, average='macro')
    recall = recall_score(true_labels_list, predicted_labels_list, average='macro')

    print(f"Accuracy: {accuracy:.4f}, Avg Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return accuracy, avg_loss

# 학습 및 Early Stopping 정의
def train_with_early_stopping(train_dataloader, val_dataloader, model, optimizer, loss_fn, device, num_epochs, patience=3):

    # to store train results
    train_accs_epoch = []  # epoch 별 train_accuracy 기록을 위한 리스트
    train_losses_epoch = []  # epoch 별 train_loss 기록을 위한 리스트
    
    # to store validation results
    val_accs_epoch = []
    val_losses_epoch = []

    model.to(device)
    best_val_accuracy = 0.0
    best_model_state_dict = None
    no_improvement = 0

    for epoch in range(num_epochs):
        train_accs, train_losses = train_fn(train_dataloader, model, optimizer, loss_fn, device)
        val_accuracy, val_loss  = eval_fn(val_dataloader, model, loss_fn, device)
        # train 결과 저장
        train_losses_epoch.extend(train_losses)
        train_accs_epoch.extend(train_accs)

        # Validation 결과 저장
        val_losses_epoch.append(val_loss)
        val_accs_epoch.append(val_accuracy)
        
        # 일정한 간격으로 체크포인트 저장
        if (epoch+1) % 5 == 0 and epoch >= 0: # 
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 필요한 경우 추가 정보 저장 (예: loss, accuracy 등)
            }
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(checkpoint, f'./checkpoint/checkpoint_epoch_{epoch+1}.pth')

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {np.mean(train_losses):.4f} - Train Accuracy: {np.mean(train_accs):.4f} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

        # Early Stopping 체크
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state_dict = model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"No improvement in validation accuracy for {patience} epochs. Early stopping...")
            break

    # # 최적의 모델 상태를 dict로 저장
    os.makedirs("model_state", exist_ok=True)
    torch.save(best_model_state_dict, f"./model_state/best_model_state_epoch{epoch}.pt")
    
    return train_losses_epoch, train_accs_epoch, val_losses_epoch, val_accs_epoch

# 한국어 문장을 입력으로 받아서 예측 라벨을 출력하는 함수
def predict_label(sentence, model, modelpath, device):
    # koELECTRA 토크나이저 불러오기
    tokenizer = ElectraTokenizer.from_pretrained(modelpath)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        return predicted_label