from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = reuters.load_data( num_words=30000, test_split=0.2)

print(x_train[0])
print(y_train[0])

print("=============")
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

# (8982,) (2246,)
# (8982,) (2246,)

print(len(x_train[0]),len(x_train[11])) #87,59 

print("news max len: ", max(len(l)for l in x_train)) #news max len:  2376
print("news avg len: ",sum(map(len,x_train))/len(x_train))
# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()
#y 

unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("y :", dict(zip(unique_elements,counts_elements)))
print("===========================")
# plt.hist(y_train, bins=50)
# plt.show()
#x

word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index))
print("=================================")

#key, value change !!
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

#after key, value change
print(index_to_word)
print(index_to_word[1])
print(index_to_word[30979])
print(len(index_to_word))  #30979

#x_train[0]
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))


