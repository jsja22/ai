#conv2D(10,(2,2), input_shape=(5,5,1))
#=>   10은그냥 다음레이어로 가는 노드  (2,2)씩자름            (5,5짜리 데이터 1은 흑백)

#Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu')
#       FILTERs,kernal_ size                       (N,28,28,1)여기서 N이 batch_shape(batch_shape,row,col,channel or filter)
#첫번째 인자 : 컨볼루션 필터의 수 입니다. #필터는 가중치를 의미
#두번째 인자 : 컨볼루션 커널의 (행, 열) 입니다.
#padding : 경계 처리 방법을 정의합니다.
#‘valid’ : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
##‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.
#input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다.
#(행, 열, 채널 수)  로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
#activation : 활성화 함수 설정합니다.
#‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
#‘relu’ : rectifier 함수, 은닉층에 주로 쓰입니다.
#‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
#‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2),strides=1,
                 padding='same', input_shape =(10,10,1))) #흑백  #stride는 건너뛰는것 디폴트는 1임.

model.add(MaxPooling2D(pool_size=(2,3)))  #(None, 5, 5, 10)   중요한거만 추출 pool_size 디폴트는 2 pool_size =3 이면 3x3 끝에 1개는 날라감. (2,3)이런식으로도 가능하다.

#padding 했기때문에  (None, 10, 10, 10)로 나옴
#(다음 layer로 넘어갈때 (9,9,10)으로 넘어감) #50번연산
model.add(Conv2D(9,(2,2),padding='valid'))
#패딩 안했기때문에 (None, 9, 9, 9) 
#model.add(Conv2D(9,(2,3))) #얘도 가능하다.
#model.add(Conv2D(8,2))
#filter, kernel_size 자동 인식 (2x2)를 2로해도됨.
#convolution은 몇번을 하던 결과를 보고  판단. 추출을 너무 많이 하는것도 안좋음.
model.add(Flatten())
model.add(Dense(1)) #dense 차원을 생각해줘야함. FLatten을 거치고 평평하게 펼친과정을 통해 2차원으로 변환하면서 output이 2차원으로 나옴.

model.summary()
# convolutional layer의 차원은 Param # 아래의 숫자를 의미한다.

#- 콘볼루션 레이어의 차원을 변경하려면 filter와 kernel_size의 인자 값을 변경해줘야 한다.

##########
#cnn은 특성추출
##########