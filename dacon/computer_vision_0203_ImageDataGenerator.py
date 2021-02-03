#####################ImageDataGenerator: 실시간 이미지 증가 (augmentation)##################
# 학습 도중에 이미지에 임의 변형 및 정규화 적용
# 변형된 이미지를 배치 단위로 불러올 수 있는 generator 생성.
# generator를 생성할 때 flow(data, labels), flow_from_directory(directory) 두 가지 함수를 사용합니다.
# fit_generator, evaluate_generator 함수를 이용하여 generator로 이미지를 불러와서 모델을 학습시킬 수 있습니다


#################각각의 파라미터들 정의####################
# rotation_range: 이미지 회전 범위 (degrees)
# width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 (원본 가로, 세로 길이에 대한 비율 값)
# rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
# shear_range: 임의 전단 변환 (shearing transformation) 범위
# zoom_range: 임의 확대/축소 범위
# horizontal_flip: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
# fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식

# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode=`nearest`)


####flow##########
##Generator를 이용해서 학습하기 전, 먼저 변형된 이미지에 이상한 점이 없는지 확인해야 합니다. 케라스는 이를 돕기 위해 flow라는 함수를 제공합니다. 여기서 rescale 인자는 빼고 진행합니다—
## 255배 어두운 이미지는 아무래도 눈으로 확인하기 힘들 테니까 말이죠.