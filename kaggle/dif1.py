import pandas as pd

df_old = pd.read_csv('C:/data/kaggle/csv/sub1.csv')
df_new = pd.read_csv('C:/data/kaggle/csv/sub2.csv')

df_old['ver']  = 'old'
df_new['ver']  = 'new'

# 두 데이터프레임을 하나로 합칩니다.
df_concatted = pd.concat([df_old, df_new], ignore_index=True)
# 모든 컬럼의 내용이 중복되는 데이터는 삭제합니다.
changes = df_concatted.drop_duplicates(df_concatted.columns[:-1], keep='last')
changes.to_csv('C:/data/kaggle/csv/answer_concat3.csv',index=False)

