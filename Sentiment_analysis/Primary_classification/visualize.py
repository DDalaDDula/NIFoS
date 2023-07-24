import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_train_loss_accuracy(train_losses, train_accuracy):
    epochs_list = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs_list, y=train_losses, mode='lines+markers', name='Train Loss'))
    fig.add_trace(go.Scatter(x=epochs_list, y=train_accuracy, mode='lines+markers', name='Train Accuracy'))
    fig.update_layout(title='Train Loss and Accuracy', xaxis_title='Epoch', yaxis_title='Value')
    fig.show()