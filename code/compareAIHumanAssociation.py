import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


labels = [str(i) for i in range(1,25)]

# negative
# similar_count = [4,3,0,1,5,2,3,1,5,0,2,1,0,0,4,3,3,1,3,2,0,2,1,1]
# diff_count = [1,7,2,8,5,6,4,9,11,3,3,4,4,6,5,4,8,9,4,6,5,4,3,5]
# noClear_count = [9,9,6,4,6,11,6,7,5,6,8,4,5,9,6,2,8,9,9,3,8,4,9,8]
# # answerAI = [2,2,2,2, 2,2,2,2, 2,2,0,2, 2,2,2,2, 0,2,0,2, 0,2,0,2]
# answerAI = [2,2,2,2, 2,2,2,2, 2,2,2,2, 2,2,2,2, 2,2,2,2, 2,2,2,2]

# # no-autocorrelation
# similar_count = [7,5,4,2,4,0,5,2,7,2,4,2,5,1,5,1,1,3,3,1,2,3,2,0]
# diff_count = [1,3,6,2,2,5,5,2,3,8,4,5,5,3,7,3,4,8,6,4,4,4,9,5]
# noClear_count = [5,5,3,7,5,8,8,4,8,7,6,7,4,6,4,5,5,4,12,9,5,6,7,6]
# # answerAI = [0,2,1,2, 1,2,1,0, 1,2,1,2, 0,2,1,2, 1,2,1,2, 1,2,1,2]
# answerAI = [2,0,1,1, 0,1,2,1, 0,1,1,1, 2,2,2,2, 2,2,2,2, 2,2,0,2]

# # small positive
# similar_count = [11,15,11,14,10,14,16,14,12,18,12,9,8,7,8,5,10,9,7,11,5,11,8,10]
# diff_count = [3,1,0,0,0,1,1,0,1,2,2,1,4,0,1,0,3,1,2,1,3,3,2,2]
# noClear_count = [1,1,1,3,3,0,2,0,3,3,1,7,3,5,2,1,2,3,3,2,3,5,3,3]
# # answerAI = [1,0,1,0, 1,0,1,1, 1,0,1,1, 1,2,1,2, 1,2,1,0, 1,2,1,2]
# answerAI = [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,0, 2,0,2,1, 1,1,0,1]

# large positive
similar_count = [13,14,6,12,15,15,15,14,12,14,8,12,21,16,10,14,11,15,12,10,8,11,13,10]
diff_count = [0,1,0,1,1,0,0,0,0,0,0,3,1,2,0,0,0,1,3,0,0,0,1,0]
noClear_count = [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0]
# answerAI = [1,1,1,0, 1,1,1,1, 1,1,1,0, 1,1,1,1, 1,0,1,1, 1,0,1,1]
answerAI = [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,0,1, 0,1,1,1, 1,1,1,1]


# men_std = [2, 3, 4, 1, 2]
# women_std = [3, 5, 2, 3, 3]
width = 0.35       # the width of the bars: can also be len(x) sequence

# fig, ax = plt.subplots(figsize=(4, 1))

fig = plt.figure(figsize=(5, 1))
ax = fig.add_subplot(1, 5, (1, 4))
# fig, ax = plt.subplots()

similar_count = np.array(similar_count)
diff_count = np.array(diff_count)
noClear_count = np.array(noClear_count)


ax.bar(labels, similar_count, width,  label='Near similar values')
ax.bar(labels, diff_count, width,  bottom=similar_count, label='Near different values')
ax.bar(labels, noClear_count, width,  bottom=similar_count + diff_count, label='Without clear associate')

yList = []
for i in range(len(answerAI)):
    ai = answerAI[i]
    if ai == 1:
        y = similar_count[i] / 2 
    elif ai == 2:
        y = similar_count[i] + diff_count[i] / 2
    else:
        y = similar_count[i] + diff_count[i] + max(noClear_count[i] / 2,0.5)
    yList.append(y)

ax.scatter(labels, yList, s = 5,c = 'black')

ax.set_xlabel('Index of maps')
ax.set_ylabel('Number of responses')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title('Maps in the large-positive group')
# ax.set_title('Maps in the negative group')
ax.legend(framealpha = 0.5,fontsize = 'small',loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()