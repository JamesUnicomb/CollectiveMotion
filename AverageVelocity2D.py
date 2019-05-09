from CollectiveMotion import CollectiveMotionClass as CMClass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cm = CMClass()

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

N = 20
for i in range(N):
    J = (0.2 * i) / N

    for nc in np.arange(2,32,2):
        cm.set_nc(nc)
        pos, vel = cm.initialise(n = n)

        for k in range(100):
            update()

        result.append((J,nc,np.square(np.mean(vel, axis=0)).sum()))

result = np.array(result)
print result

plt.imshow(result[:,2].reshape(20,-1))
plt.colorbar()
plt.show()

df = pd.DataFrame(result, columns=['J', 'nc', 'Va'])
print df
