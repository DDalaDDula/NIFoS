from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import ElectraTokenizer

def sent_length_vis(sentence_list,tokenizerpath,ratio): # 문장 리스트, 토크나이저(모델) 경로, 원하는 비율(비율 만큼의 문장을 커버하는 padding_length를 구하기 위함)
    # koELECTRA 토크나이저 불러오기
    tokenizer = ElectraTokenizer.from_pretrained(tokenizerpath)
    length = [len(tokenizer.encode(sentence)) for sentence in tqdm(sentence_list)]
    length.sort()
    # 비율을 기반으로 padding_length를 계산
    padding_length = length[int(len(length) * ratio)+1]

    # 히스토그램 생성 (전체 데이터와 padding_length 이상의 데이터를 구분하여 그림)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=[leng for leng in length if leng <= padding_length], nbinsx=200, name='Max Length 이하'))
    fig.add_trace(go.Histogram(x=[leng for leng in length if leng > padding_length], nbinsx=200, name='Max Length 이상'))

    # 최댓값과 최솟값 계산
    max_value = max(length)
    min_value = min(length)

    # 최댓값과 최솟값을 주석으로 추가하여 표시
    annotations = [
        {
            'x': max_value,
            'y': 0,
            'xref': 'x',
            'yref': 'y',
            'text': f'Max: {max_value}',
            'showarrow': True,
            'arrowhead': 4,
            'ax': 0,
            'ay': -40
        },
        {
            'x': min_value,
            'y': 0,
            'xref': 'x',
            'yref': 'y',
            'text': f'Min: {min_value}',
            'showarrow': True,
            'arrowhead': 4,
            'ax': 0,
            'ay': -40
        }
    ]

    # 그래프 레이아웃 설정
    fig.update_layout(annotations=annotations, barmode='overlay')
    fig.update_traces(opacity=0.7)  # 겹쳐진 히스토그램의 투명도 조정
    # # 그래프 생성 및 레이아웃 설정
    # fig = go.Figure(histogram)
    # fig.update_layout(annotations=annotations)

    # 그래프 출력
    fig.show()

    # 계산된 padding_length 반환
    return padding_length




def plot_training_progress(train_losses, train_accs, val_losses, val_accs):
    # 그래프 생성
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"))

    # Train Loss 그래프
    fig.add_trace(
        go.Scatter(x=list(range(len(train_losses))), y=train_losses, mode='lines', name='Train Loss'),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Iterations", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)

    # Train Accuracy 그래프
    fig.add_trace(
        go.Scatter(x=list(range(len(train_accs))), y=train_accs, mode='lines', name='Train Accuracy'),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Iterations", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    # Validation Loss 그래프
    fig.add_trace(
        go.Scatter(x=list(range(len(val_losses))), y=val_losses, mode='lines', name='Validation Loss'),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Iterations", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)

    # Validation Accuracy 그래프
    fig.add_trace(
        go.Scatter(x=list(range(len(val_accs))), y=val_accs, mode='lines', name='Validation Accuracy'),
        row=2, col=2
    )
    fig.update_xaxes(title_text="Iterations", row=2, col=2)
    fig.update_yaxes(title_text="Accuracy", row=2, col=2)

    fig.update_layout(title="Training Progress")
    fig.show()