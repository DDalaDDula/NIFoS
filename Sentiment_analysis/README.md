 전이학습을 위한 모델 성능 평과과정
 KoBERT (Korean Bidirectional Encoder Representations from Transformers) : 기존 구글 BERT base multilingual cased의 한국어 성능 한계를 극복하기 위해 SKT Brain에서 개발한 모델
 Huggingface.co 기반으로 사용할 수 있게 Wrapping 작업을 수행한 KoBERT 사용.
 사용 데이터 목록 참고(https://littlefoxdiary.tistory.com/42):
 1. 네이버 영화 리뷰 : https://github.com/e9t/nsmc
 2. 3i4k : https://github.com/warnikchow/3i4k?fbclid=IwAR17XFnJr_y6lp49z7MwwWOJsyRql5Zwvc0KQxgE7x6DtmV7qELK0ymE__A
 3. 감정 분류를 위한 대화 음성 데이터셋 : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263
    - 라벨링 리스트 = [happiness, angry, disgust, fear, neutral, sadness, surprise]
 4. 감성 및 발화스타일 동시 고려 음성합성 데이터 : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71349
 5. 감성 대화 말뭉치 : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86
 6. Toxic Comment Data : https://github.com/songys/Toxic_comment_data