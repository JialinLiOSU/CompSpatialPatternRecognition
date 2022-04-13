import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


labels = [str(i) for i in range(1,25)]

# negative
# yes_count = [1,3,2,1,3,2,4,1,1,0,1,1,0,0,5,1,1,1,3,3,1,2,0,0]
# no_count = [13,16,6,12,14,17,9,16,20,9,12,8,9,15,10,8,17,18,13,8,12,8,13,14]
# answerAI = [0 for i in range(len(yes_count))]

# no-autocorrelation
yes_count = [5,5,2,2,6,0,4,3,3,2,3,3,9,2,5,2,4,2,2,2,2,2,2,2]
no_count = [8,8,11,9,6,13,14,5,15,15,11,11,5,8,11,7,6,13,19,12,9,11,16,9]
answerAI = [0 for i in range(len(yes_count))]

# small positive
# yes_count = [8,15,9,14,10,12,13,14,12,21,10,11,8,9,7,3,8,8,7,11,5,12,7,12]
# no_count = [7,2,3,4,3,3,6,0,4,2,5,6,7,3,4,3,7,5,5,3,6,7,6,3]
# answerAI = [0,0,0,0, 0,1,0,1, 0,0,1,1, 0,0,1,0, 0,0,0,0, 0,0,0,1]

# large positive
# yes_count = [14,14,6,13,16,14,14,14,12,13,7,13,20,17,10,13,11,16,13,10,8,11,12,10]
# no_count = [0,1,0,0,0,1,2,0,0,2,1,2,1,1,0,0,0,1,2,0,1,0,2,0]
# answerAI = [1,0,1,1, 1,1,1,1, 1,1,1,1, 1,0,1,1, 1,0,1,1, 1,0,1,1]


# men_std = [2, 3, 4, 1, 2]
# women_std = [3, 5, 2, 3, 3]
width = 0.35       # the width of the bars: can also be len(x) sequence

fig = plt.figure(figsize=(5, 1))
ax = fig.add_subplot(1, 5, (1, 4))
# fig, ax = plt.subplots()

ax.bar(labels, yes_count, width,  label='Concentrated')
ax.bar(labels, no_count, width,  bottom=yes_count, label='Not concentrated')

yList = []
for i in range(len(yes_count)):
    ai = answerAI[i]
    if ai == 1:
        y = yes_count[i] / 2 
    else:
        y = yes_count[i] + max(no_count[i] / 2,0.5)
    yList.append(y)

ax.scatter(labels, yList, s = 5,c = 'black')

ax.set_xlabel('Index of maps')
ax.set_ylabel('Number of responses')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title('Maps in the no-association group')
ax.legend(framealpha = 0.5,fontsize = 'small', loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()