from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs =['너무 재밌어요','참 최고에요', ' 참 잘 만든 영화에요','추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요','별로에요','생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밋네요','철수가 잘 생기긴 했어요']

# postive 1 negative 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x,padding='pre')  #post
print(pad_x)
print(pad_x.shape) #(13,5)
truncated= pad_sequences(pad_x, maxlen=4)
print(truncated)
print(np.unique(pad_x))
print(len(np.unique(pad_x)))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense,LSTM, Flatten, Conv1D

model = Sequential()
##원핫인코딩의 문제점 데이터가 너무 커짐!따라서 임베딩을 사용
#model.add(Embedding(input_dim=20,output_dim=11,input_length=5))  # 3dim  #input _dim 단어사전의 갯수 같거나 커야한다 실제 단어사전의 갯수와
model.add(Embedding(28,11))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
model.fit(pad_x,labels,epochs=100)

acc = model.evaluate(truncated,labels)[1]
print(acc) #1.0
# embedding (Embedding)        (None, 5, 11)             220
# _________________________________________________________________
# flatten (Flatten)            (None, 55)                0
# ________________________________________________________________
# dense (Dense)                (None, 1)                 56
# =================================================================

