import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import xlabel
x = [ 13, 14, 9, 20, 2, 19, 6, 7, 16, 5, 1, 3, 5, 25, 17, 2, 18, 10, 0,   8,   6,   5,  17,  26,  14,  14,   5,   0, 11,  11,  10,   2,   3, 17,  23,   5,  26,  10,  12,   3,  22,  10, 0,  10,   5,  21,   0,   3,   9,   0]
plt.bar(range(0,50), x)
plt.title('Image represented as a histogram')
plt.xlabel('clusters')
plt.ylabel('frequency')
plt.show()