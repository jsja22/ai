def solution(answers):
    student_ans1,student_ans2,student_ans3= [],[],[]
    student_ans1.append([1,2,3,4,5]*2000)
    student_ans2.append([2,1,2,3,2,4,2,5]*1250)
    student_ans3.append([3,3,1,1,2,2,4,4,5,5]*1000)
    
    ans1_cnt ,ans2_cnt,ans3_cnt= 0,0,0
    
    for i in range(len(answers[0])):
        if(student_ans1[0][i]==answers[0][i]):
            ans1_cnt +=1
        elif(student_ans2[0][i]==answers[0][i]):
            ans2_cnt +=2
        elif(student_ans3[0][i]==answers[0][i]):
            ans3_cnt +=3
    
        if 
answers = []
answers.append([2,2,4,5,4]*2000)

print(solution(answers))