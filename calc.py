import numpy as np

start = 1
step = 1/29
array = [0]
array_three = []

for i in range(0,28):
    start = start - step
    array.append(start)
    # print(start)

array_two = [4.223,4.205,4.154,4.106,4.078,4.078,4.078,4.055,3.987,3.832,3.748,3.759,3.888,4.045,
 4.286,4.684,5.075,5.442,5.901,6.484,6.967,7.446,7.662,7.482,6.889,6.32,6.085,6.169,6.528]

for i in range(len(array_two)):
    array_three.append(array_two[(len(array_two)-1)-i])
    # print(array_three[i])

array_answer = []

for i in range(len(array_two)):
    array_three[i] = (array_three[i])/10
    array_answer.append(np.power((1-array_three[i]),(2/3)))
    print(array_answer[i])

# print(array_answer)
# print("")
# print(len(array_answer))
# print(len(array_two))