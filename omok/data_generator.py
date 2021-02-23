import numpy as np
import os
from glob import glob
from tqdm import tqdm


game_rule = 'Renju' # Freestyle, Fastgame, Standard, Renju
base_path = 'C:/data/omok'
output_path = os.path.join('Renju_con', os.path.basename(base_path)) #->현재경로에 dataset폴더 추가 되고 그 안에 base_path의 끝에 있는 폴더가 생성됨
#os.path.join->경로를 병합하여 새 경로 생성
#폴더, 파일을 보여준다. 파일이 없거나.. 형식에 맞지 않으면 아무것도 안나온다.
#os.path.basename(base_path)

#output_path = os.path.join('base_path','dataset') #-> /bate_path/dataset 
os.makedirs(output_path, exist_ok=True)  #os.mkdir와 os.makedirs
print(os.listdir('.'))
# #makedirs는 './a/b/c' 처럼 원하는 만큼 디렉토리를 생성
# #exist_ok라는 파라미터를 True로 하면 해당 디렉토리가 기존에 존재하면 에러발생 없이 넘어가고, 없을 경우에만 생성

# #glob ->glob는 파일들의 리스트를 뽑을 때 사용하는데, 파일의 경로명을 이용해서 입맛대로 요리할 수 있답

# #renju_path =os.path.join(base_path, '%s*/*.psq' % 'Renju'
# #file_list = glob(os.path.join(base_path, '%s*/*.psq' % (game_rule, )))
free_path = os.path.join(base_path, '%s*/*.psq' % (game_rule, ))
free_list = glob(free_path)


board_num =0
#tqdm -> 상태진행률
for i, file_path in enumerate(tqdm(free_list)):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines() #한 라인씩 끊어서 리스트형태로 출력

        # 현재 x,y
        # 현재 x,y
        #print(lines)
    # w, h = lines[0].split(' ')[1].strip(',').split('x')
    # w, h = int(w), int(h)
    # print(w,",",h)
    width = 15
    height = 15
    lines = lines[1:]
    
    inputs, outputs = [], []
    board = np.zeros([height, width], dtype=np.int8) #바둑판 0으로 만들어주기

    for j, line in enumerate(lines):
        if ',' not in line:
            break
        x, y, unknown = np.array(line.split(','), np.int8)
        #print(x,",",y)
        if j % 2 == 0:
            player = 1
        else:
            player = 2  #ME, COMPUTER player set
        
        input = board.copy().astype(np.int8)  #업데이트 된 board를 copy
        input[(input != player) & (input != 0)] = -1
        input[(input == player) & (input != 0)] = 1

        output = np.zeros([height, width], dtype=np.int8)
        output[y-1, x-1] = 1  #dataset의 시작이 1,1 이기에 0,0으로 변경해줘야함
        #output 은 다음에 놓아야 할곳에 1을 두는 y값임
        #print(y-1,",",x-1)  # 바둑판의 왼쪽 위 끝이 0,0으로 봐야함(ursina)
        #print(output)  
        
        # data augmentation
        # ● 위아래 반전 : np.flipud()
        # ● 좌우 반전 : np.fliplr()
        # ● 상하좌우 전체 반전 : np.flip()
        for k in range(4):
            input_rot = np.rot90(input, k=k)
            output_rot = np.rot90(output, k=k)

            inputs.append(input_rot)
            outputs.append(output_rot)

            inputs.append(np.fliplr(input_rot))
            outputs.append(np.fliplr(output_rot))

            inputs.append(np.flipud(input_rot))
            outputs.append(np.flipud(output_rot))

        #x,y 값에 현재 player로 지정
        board[y-1, x-1] = player
        #print(board[y-1,x-1])  #1,2 번갈아 바뀜
        #print(board)
    board_num+=1
    print(board_num)
        #print(board.shape)

   
    #save dataset
    #np.savez_compressed() : 여러개의 배열을 1개의 압축된 *.npz 포맷 파일로 저장하기 ->큰 NumPy와 배열을 보낼 수있는 가장 좋은 방법, 속도도 개빠름!!!
    np.savez_compressed(os.path.join(output_path, '%s.npz' % (str(i).zfill(5))), inputs=inputs, outputs=outputs)   #str(i).zfill(5)문자열앞에 0으로 채워서 5자리까지 나타낼수있음
    
