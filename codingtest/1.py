#카카오 광고삽입

from datetime import datetime

#문자열을 먼저 sec로 변경하자

def convert_to_seconds(time):
    time = map(int, time.split(":"))
    result = 0 
    for t, sec in zip(time, [3600,60,1]):   
        result += t*sec #이렇게 하면 time이랑 [3600,60,1]에서 2x3600, 3x60 55x1 로 만들어줄수있음
    return result

play_time = "02:03:55"
play_sec = convert_to_seconds(play_time)
print(play_sec)

#
