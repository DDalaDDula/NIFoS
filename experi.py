def solution(operations):
    operations = [i.split(' ')[1] if 'I' in i else i for i in operations]
    answer = []
    for i in operations:
        if 'D -' in i:
            try:
                answer = answer[1:]
            except:
                answer = []
        elif 'D' in i:
            try:
                answer.pop()
            except:
                answer = []
        else:
            answer.append(int(i))
            answer.sort()
    return [answer[-1], answer[0]] if answer else [0,0]
print(solution(["I 2","I 4","D -1", "I 1", "D 1"]))