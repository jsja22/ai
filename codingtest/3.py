def solution(name):
    make_name = [ min(ord(i)-ord("A"),ord("Z")-ord(i)+1) for i in name]
    idx = 0
    answer = 0
    
    while True:
        answer += make_name[idx]
        make_name[idx] = 0
        if sum(make_name) ==0:
            break
        left, right =1,1
        while make_name[idx-left] ==0:
            left +=1
        while make_name[idx+right] ==0:
            right +=1
        print('make_name[idx]:',make_name[idx], 'left :', left)
        print('make_name[idx]:',make_name[idx], 'right :', right)
        answer += left if left<right else right
        idx += -left if left < right else right
        print(answer)
        print(idx)
    return answer
        
name = "JEROEN"

print(solution(name)) ##JAO -> [9,0,12]

