import pandas as pd

# 엑셀 파일 읽기
df = pd.read_excel('test.xlsx')

# 특정 열을 쉼표로 구분된 문자열로 변환
values = df['SALVATORE FERRAGAMO'].astype(str).tolist()
result = ', '.join(values)

print(result)