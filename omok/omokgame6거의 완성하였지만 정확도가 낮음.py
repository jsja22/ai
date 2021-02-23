import pygame, sys
from pygame.locals import *
from tensorflow.keras.models import load_model
import numpy as np
from time import sleep

pygame.init()
### 인공지능턴에 다음 수 예측할때 사용할 모델###
model = load_model('C:/data/omok/modelcheckpoint/last_model.h5')

omokpan = pygame.image.load('C:/data/omok/image/board.png')
white_stone = pygame.image.load('C:/data/omok/image/white.png')
black_stone = pygame.image.load('C:/data/omok/image/black.png')

omokpan_size = omokpan.get_rect().size # 사이즈 가져오기
omokpan_w = omokpan_size[0] # 가로 크기
omokpan_h = omokpan_size[1] # 세로 크기

window_w = 801
window_h = 501

print(omokpan_w,omokpan_h) #501,501
omokpan_x_position = (501 / 2) - (omokpan_w / 2) 
omokpan_y_position = (501 / 2) - (omokpan_h / 2)   

fps = 60 # frames per second setting
fps_clock = pygame.time.Clock()

dis_surf = pygame.display.set_mode((window_w, window_h))
pygame.display.set_caption('오목')
font = pygame.font.SysFont("consolas",20)

##########색 정의127
bg_color = (20,130,255) 
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 50, 255)
red = (255, 0, 0)
green = (0, 255, 0)
slate_blue = (106,90,205)
deep_pink =	(255,20,147)
#########사이즈 정의

board_size = 15
grid_size = 30
black = 1
white = 2


class Board(object):
    def __init__(self):
        self.w = window_w
        self.h = window_h
        self.board_image = omokpan
        self.white_image = pygame.transform.scale(white_stone, (grid_size, grid_size))
        self.black_image = pygame.transform.scale(black_stone, (grid_size, grid_size))
        
    def init_board(self,stone_ary,stone_list):
        self.stone_ary = stone_ary
        self.stone_list = stone_list
        for y in range(board_size):
            for x in range(board_size):
                self.stone_ary[y][x] = 0   ##보드를 0으로 초기화
                self.stone_ary[y][x] = 0 
    
    def draw_board(self):
        dis_surf.fill(bg_color)
        dis_surf.blit(self.board_image,(0,0))

    def draw_w_stone(self, x_stone_position,y_stone_position):
        player = 2
        x = x_stone_position * grid_size + 25
        y = y_stone_position * grid_size + 25 
        
        self.w_stone_image = white_stone
        img = self.white_image
        dis_surf.blit(img,(x,y))
        

    def draw_b_stone(self,x_stone_position,y_stone_position):
        player = 1
        x = x_stone_position * grid_size + 25
        y = y_stone_position * grid_size + 25
        
        self.b_stone_image = black_stone
        img = self.black_image
        dis_surf.blit(img,(x,y))
    
    def draw_text(self):
        font = pygame.font.SysFont("consolas",30)
        textSurface  = font.render('Omok GAME',True, green)
        textRectObj = textSurface.get_rect()
        textRectObj.center = (650, 100) 
        dis_surf.blit(textSurface, textRectObj)
        pygame.draw.rect(dis_surf, deep_pink, (550, 150, 200, 100)) 

    def draw_reset(self):
        font = pygame.font.SysFont("consolas",30)
        textSurface  = font.render('RESET',True, white)
        textRectObj = textSurface.get_rect()
        textRectObj.center = (650, 400) 
        dis_surf.blit(textSurface, textRectObj)

    def draw_exit(self):
        font = pygame.font.SysFont("consolas",30)
        textSurface  = font.render('EXIT',True, white)
        textRectObj = textSurface.get_rect()
        textRectObj.center = (650, 470) 
        dis_surf.blit(textSurface, textRectObj)

    def quit_game(self):
        pygame.quit()
        sys.exit()

class Game_Rule(object):
    #차례 바꿔주는 함수
    def turn(self,player):
        self.player = player
        if self.player == 1:
            player = 2
            return player
        elif self.player == 2:
            player = 1
            return player


    def winner_check(self,player,stone_ary,x,y):
        self.x = x
        self.y = y
        self.player = player
        print("last point :",self.x,self.y)
        print("player:",self.player)
        self.stone_ary = stone_ary
        print("######################")
        print(stone_ary)
        print("######################")

        #턴마다 이긴사람이 있는지 체크하고 마지막에 놓인 돌기준으로 player를 받아서 가로 세로 대각 의 총 여덟방향으로 5개 연속된 돌이 있는지 체크해줌 
        if x<11 and stone_ary[y][x] == player and stone_ary[y][x+1] ==player and stone_ary[y][x+2] ==player and  stone_ary[y][x+3] == player and stone_ary[y][x+4] ==player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x>=4 and stone_ary[y][x] == player and stone_ary[y][x-1] == player and stone_ary[y][x-2] == player and stone_ary[y][x-3] == player and stone_ary[y][x-4]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y<11 and stone_ary[y][x] == player and stone_ary[y+1][x] == player and stone_ary[y+2][x] == player and stone_ary[y+3][x] == player and stone_ary[y+4][x]== player :
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y>=4 and stone_ary[y][x] == player and stone_ary[y-1][x] == player and stone_ary[y-2][x] == player and stone_ary[y-3][x] == player and stone_ary[y-4][x]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x>=4 and y>=4 and stone_ary[y][x] == player and stone_ary[y-1][x-1] == player and stone_ary[y-2][x-2] == player and stone_ary[y-3][x-3] == player and stone_ary[y-4][x-4]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x<11 and y<11 and stone_ary[y][x] == player and stone_ary[y+1][x+1] == player and stone_ary[y+2][x+2] == player and stone_ary[y+3][x+3] == player and stone_ary[y+4][x+4]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y>=4 and x<11 and stone_ary[y][x] == player and stone_ary[y-1][x+1] == player and stone_ary[y-2][x+2] == player and stone_ary[y-3][x+3] == player and stone_ary[y-4][x+4]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y<11 and x>=4 and stone_ary[y][x] == player and stone_ary[y+1][x-1] == player and stone_ary[y+2][x-2] == player and stone_ary[y+3][x-3] == player and stone_ary[y+4][x-4]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        '''   
        ################################################
        elif x<12 and stone_ary[y][x-1] == player and stone_ary[y][x] ==player and stone_ary[y][x+1] ==player and  stone_ary[y][x+2] == player and stone_ary[y][x+3] ==player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x<13 and stone_ary[y][x-2] == player and stone_ary[y][x-1] ==player and stone_ary[y][x] ==player and  stone_ary[y][x+1] == player and stone_ary[y][x+2] ==player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x<14 and stone_ary[y][x-3] == player and stone_ary[y][x-2] ==player and stone_ary[y][x-1] ==player and  stone_ary[y][x] == player and stone_ary[y][x+1] ==player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x<15 and stone_ary[y][x-4] == player and stone_ary[y][x-3] ==player and stone_ary[y][x-2] ==player and  stone_ary[y][x-1] == player and stone_ary[y][x] ==player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        #############################################
        elif x>=3 and stone_ary[y][x+1] == player and stone_ary[y][x] == player and stone_ary[y][x-1] == player and stone_ary[y][x-2] == player and stone_ary[y][x-3]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x>=2 and stone_ary[y][x+2] == player and stone_ary[y][x+1] == player and stone_ary[y][x] == player and stone_ary[y][x-1] == player and stone_ary[y][x-2]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x>=1 and stone_ary[y][x+3] == player and stone_ary[y][x+2] == player and stone_ary[y][x+1] == player and stone_ary[y][x] == player and stone_ary[y][x-1]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x>=0 and stone_ary[y][x+4] == player and stone_ary[y][x+3] == player and stone_ary[y][x+2] == player and stone_ary[y][x+1] == player and stone_ary[y][x]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        ###########################################
        elif y<12 and stone_ary[y-1][x] == player and stone_ary[y][x] == player and stone_ary[y+1][x] == player and stone_ary[y+2][x] == player and stone_ary[y+3][x]== player :
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y<13 and stone_ary[y-2][x] == player and stone_ary[y-1][x] == player and stone_ary[y][x] == player and stone_ary[y+1][x] == player and stone_ary[y+2][x]== player :
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y<14 and stone_ary[y-3][x] == player and stone_ary[y-2][x] == player and stone_ary[y-1][x] == player and stone_ary[y][x] == player and stone_ary[y+1][x]== player :
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y<15 and stone_ary[y-4][x] == player and stone_ary[y-3][x] == player and stone_ary[y-2][x] == player and stone_ary[y-1][x] == player and stone_ary[y][x]== player :
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        ##########################################
        elif y>=3 and stone_ary[y+1][x] == player and stone_ary[y][x] == player and stone_ary[y-1][x] == player and stone_ary[y-2][x] == player and stone_ary[y-3][x]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y>=2 and stone_ary[y+2][x] == player and stone_ary[y+1][x] == player and stone_ary[y][x] == player and stone_ary[y-1][x] == player and stone_ary[y-2][x]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y>=1 and stone_ary[y+3][x] == player and stone_ary[y+2][x] == player and stone_ary[y+1][x] == player and stone_ary[y][x] == player and stone_ary[y-1][x]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y>=0 and stone_ary[y+4][x] == player and stone_ary[y+3][x] == player and stone_ary[y+2][x] == player and stone_ary[y+1][x] == player and stone_ary[y][x]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        #########################################################
        elif x>=3 and y>=3 and stone_ary[y+1][x+1] == player and stone_ary[y][x] == player and stone_ary[y-1][x-1] == player and stone_ary[y-2][x-2] == player and stone_ary[y-3][x-3]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x>=2 and y>=2 and stone_ary[y+2][x+2] == player and stone_ary[y+1][x+1] == player and stone_ary[y][x] == player and stone_ary[y-1][x-1] == player and stone_ary[y-2][x-2]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x>=1 and y>=1 and stone_ary[y+3][x+3] == player and stone_ary[y+2][x+2] == player and stone_ary[y+1][x+1] == player and stone_ary[y][x] == player and stone_ary[y-1][x-1]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x>=0 and y>=0 and stone_ary[y+4][x+4] == player and stone_ary[y+3][x+3] == player and stone_ary[y+2][x+2] == player and stone_ary[y+1][x+1] == player and stone_ary[y][x]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        #########################################################
        elif x<12 and y<12 and stone_ary[y-1][x-1] == player and stone_ary[y][x] == player and stone_ary[y+1][x+1] == player and stone_ary[y+2][x+2] == player and stone_ary[y+3][x+3]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x<13 and y<13 and stone_ary[y-2][x-2] == player and stone_ary[y-1][x-1] == player and stone_ary[y][x] == player and stone_ary[y+1][x+1] == player and stone_ary[y+2][x+2]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x<14 and y<14 and stone_ary[y-3][x-3] == player and stone_ary[y-2][x-2] == player and stone_ary[y-1][x-1] == player and stone_ary[y][x] == player and stone_ary[y+1][x+1]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif x<15 and y<15 and stone_ary[y-4][x-4] == player and stone_ary[y-3][x-3] == player and stone_ary[y-2][x-2] == player and stone_ary[y-1][x-1] == player and stone_ary[y][x]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        ####################################################
        elif y>=3 and x<12 and stone_ary[y+1][x-1] == player and stone_ary[y][x] == player and stone_ary[y-1][x+1] == player and stone_ary[y-2][x+2] == player and stone_ary[y-3][x+3]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y>=2 and x<13 and stone_ary[y+2][x-2] == player and stone_ary[y+1][x-1] == player and stone_ary[y][x] == player and stone_ary[y-1][x+1] == player and stone_ary[y-2][x+2]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y>=1 and x<14 and stone_ary[y+3][x-3] == player and stone_ary[y+2][x-2] == player and stone_ary[y+1][x-1] == player and stone_ary[y][x] == player and stone_ary[y-1][x+1]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y>=0 and x<15 and stone_ary[y+4][x-4] == player and stone_ary[y+3][x-3] == player and stone_ary[y+2][x-2] == player and stone_ary[y+1][x-1] == player and stone_ary[y][x]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        ######################################################
        elif y<12 and x>=3 and stone_ary[y-1][x+1] == player and stone_ary[y][x] == player and stone_ary[y+1][x-1] == player and stone_ary[y+2][x-2] == player and stone_ary[y+3][x-3]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y<13 and x>=2 and stone_ary[y-2][x+2] == player and stone_ary[y-1][x+1] == player and stone_ary[y][x] == player and stone_ary[y+1][x-1] == player and stone_ary[y+2][x-2]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y<14 and x>=1 and stone_ary[y-3][x+3] == player and stone_ary[y-2][x+2] == player and stone_ary[y-1][x+1] == player and stone_ary[y][x] == player and stone_ary[y+1][x-1]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        elif y<15 and x>=0 and stone_ary[y-4][x+4] == player and stone_ary[y-3][x+3] == player and stone_ary[y-2][x+2] == player and stone_ary[y-1][x+1] == player and stone_ary[y][x]== player:
            print("player:",player,"  님이 승리하셨습니다")
            self.game_end(player)
        '''

    def printText(self,msg, color, pos):
        font2 = pygame.font.SysFont("consolas",30)
        textSurface  = font2.render(msg,True, (255,255,255))
        textRectObj = textSurface.get_rect()
        textRectObj.center = (650, 200) 
        dis_surf.blit(textSurface, textRectObj)
        

    def game_end(self,player):
        
        self.player = player
        if player == 1:
            self.printText('YOU win!!','BLACK',(50,50))
        elif player == -1:
            self.printText('AI win!!','WHITE',(50,50))
    
    # def stone_check(self,x_stone_position,y_stone_position,stone_ary):
    #     self.x = x_stone_position
    #     self.y = y_stone_position
    #     self.stone_ary = stone_ary
    #     if (stone_ary[self.x][self.y] == 1 or stone_ary[self.x][self.y] == -1)
        
def main():
    board = Board()
    game_rule = Game_Rule()
    LEFT = 1
    player = 1 #1이 white 2가 black
    button = None
    stone_list = np.zeros((15,15),dtype=np.int)
    stone_ary = np.zeros((15,15),dtype=np.int)

  
    while True:
        fps_clock.tick(fps)
        board.draw_board()
        board.draw_text()
        board.draw_reset()
        board.draw_exit()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
                print(event.pos[0],event.pos[1])
                if 600<event.pos[0]<700 and 385<event.pos[1]<410 :
                    print("reset")
                    stone_list = np.zeros((15,15),dtype=np.int)
                    stone_ary = np.zeros((15,15),dtype=np.int)
                    break
                if 600<event.pos[0]<700 and 450<event.pos[1]<490 :
                    board.quit_game()
                    break
                player=1
                x_stone_position = event.pos[0] // grid_size - 1
                y_stone_position = event.pos[1] // grid_size - 1
                print("내가 놓은 위치는:",x_stone_position, y_stone_position)  #왼쪽 가장 위가 0,0
                sleep(0.1)
                #game_rule.turn(player_num)
                stone_list[y_stone_position][x_stone_position] = player
                stone_ary[y_stone_position][x_stone_position] = player  #수를 둔 좌표에 15x15배열상태에 플레이어별로 저장
                print(stone_ary)
                game_rule.winner_check(player,stone_ary,x_stone_position,y_stone_position)
                player = -1  #player update
                stone_ary2 = np.expand_dims(stone_ary, axis=(0, -1)).astype(np.float32)
                pred = model.predict(stone_ary2).squeeze()
                pred = pred.reshape((15, 15))
                pred_y, pred_x = np.unravel_index(np.argmax(pred), pred.shape)
                print("인공지능이 예측한 다음수는 ",pred_x, pred_y,"입니다")

                stone_list[pred_y][pred_x] = player
                stone_ary[pred_y][pred_x] = player
                game_rule.winner_check(player,stone_ary,pred_x,pred_y)
                print("======oneclick end=======")
                
                    
            for y_stone_position in range(board_size):
                for x_stone_position in range(board_size):
                    player_num = stone_list[y_stone_position][x_stone_position]
                    if player_num == 1 : 
                        board.draw_b_stone(x_stone_position,y_stone_position)
                    
            for pred_y in range(board_size):
                for pred_x in range(board_size):
                    player_num = stone_list[pred_y][pred_x]
                    if player_num == -1:
                        board.draw_w_stone(pred_x,pred_y)
            
            pygame.display.update()
            fps_clock.tick(fps)

if __name__ == "__main__":
    main()

