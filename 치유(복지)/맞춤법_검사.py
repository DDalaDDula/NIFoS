#- 패키지 임포트
import requests
import json
import pandas as pd
from tqdm.notebook import tqdm
import re


'''맞춤법 검사 함수'''
def ko_grammar(sent):
    global correct_list ; correct_list= []
    #for i in tqdm(range(len(a))):
    for i in tqdm(range(len(sent))):
        try:                                                                               
            response = requests.post('http://164.125.7.61/speller/results', data={'text1': sent[i]})  # 맞춤법 검사 요청 (requests)    
            data = response.text.split('data = [', 1)[-1].rsplit('];', 1)[0] # 응답에서 필요한 내용 추출 (html 파싱)
            data = json.loads(data) # 파이썬 딕셔너리 형식으로 변환
            orgStr = [err['orgStr'] for err in data['errInfo']] #오류가 담긴 errinfo에서 고쳐야 되는 단어 리스트로 추출 
            correct = [err['candWord'] for err in data['errInfo']] #고쳐진 단어도 리스트로 추출
            splited = sent[i] 

            '''여러 개로 고쳐졌을 때 맨 처음 단어만 선택'''
            for j in range(len(correct)):
                if '|'in correct[j]: #여러 개로 고쳐질 경우 문자열이 '|'로 구분되어 있음
                    com = re.compile('\|')
                    many = com.search(correct[j])
                    correct[j]= correct [j][0:(many.span()[0])]
                '''틀린 문장 고치기'''   
                splited = splited.replace(orgStr[j], correct[j])

        except:      
            splited = sent[i]

        correct_list.append(splited)