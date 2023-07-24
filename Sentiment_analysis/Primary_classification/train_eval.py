import torch

def train(model, data_loader, optimizer, loss_fn, num_epochs):
    model.to(device)
    model.train()

    train_losses = []
    train_accuracy = []

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 정확도 계산
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        average_loss = total_loss / len(data_loader)
        train_losses.append(average_loss)
        accuracy = correct_predictions / total_predictions
        train_accuracy.append(accuracy)

    return train_losses, train_accuracy

def evaluate(model, data_loader):
    model.to(device)
    model.eval()

    test_accuracy = []
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)
            correct_predictions = (predictions == labels).sum().item()
            total_predictions = labels.size(0)

            test_accuracy.append(correct_predictions / total_predictions)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    test_accuracy_mean = sum(test_accuracy) / len(test_accuracy)
    return test_accuracy_mean, true_labels, predicted_labels
