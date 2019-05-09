from CollectiveMotion import CollectiveMotionClass as CMClass
from CollectiveMotionFunctions import fisher
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nc = 20

cm = CMClass()
cm.set_nc(20)

blocks     = 2**0
block_size = 2**9
n          = blocks * block_size
timestep   = 0.05

ra = 0.8
rb = 0.2
re = 0.5
r0 = 1.0
b  = 5.0
J  = 0.1

def update():
    global pos, vel, K

    n0    =  np.random.randn(n,3)
    n0    *= 1.0 / np.linalg.norm(n0, axis=1).reshape(-1,1)
    n0    = n0.astype(np.float32)

    pos, vel = cm.step(pos, vel, n, timestep,
                    ra, rb, re, r0, b, J,
                    blocks, block_size)

    pos -= np.mean(pos, axis=0)

result = []

N = 10
for J in np.linspace(0.675, 0.825, num=20):
    for j in range(1):
        pos, vel = cm.initialise(n = n)

        for k in range(150):
            update()

        result.append((J,fisher(J,nc,pos,vel)))

result = np.array(result)
print result
df = pd.DataFrame(result, columns=['J', 'F'])
dfa = df.groupby('J').mean().reset_index()
dfa.plot.scatter(x='J', y='F')
plt.show()
