import numpy as np
import matplotlib.pyplot as plt

src = np.loadtxt(fname="Audio.csv", dtype="float", delimiter=",")
plt.imshow(src)
plt.show()
#height = np.arange(len(src))
#plt.bar(height, src)
#plt.show()

