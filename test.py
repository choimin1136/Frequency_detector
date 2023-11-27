import io

# 리스트 데이터를 생성합니다.
data = ['안녕하세요', '파이썬', '텍스트 파일']

# 리스트 데이터를 저장합니다.
with io.open('data.txt', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(item + '\n')