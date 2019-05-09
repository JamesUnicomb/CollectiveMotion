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


for i, J in enumerate([0.001, 0.05, 0.1, 0.15, 0.2]):
    PHI = []
    THETA = []

    for j in range(20):
        pos, vel = cm.initialise(n = n)

        for k in range(150):
            update()

        result.append((J,np.square(np.mean(vel, axis=0)).sum()))

        theta = np.arctan2(vel[:,0], vel[:,1])
        phi = np.arccos(vel[:,2] / np.linalg.norm(vel, axis=1))

        PHI += list(np.mod(phi - np.mean(phi) + np.pi/2, np.pi))
        THETA += list(np.mod(theta - np.mean(theta) + np.pi, 2.0 * np.pi) - np.pi)

    plt.subplot(151 + i)


    plt.hist2d(x=PHI, y=THETA, 
                bins=[np.linspace(0.0, np.pi, num=25),
                      np.linspace(-np.pi, np.pi, num=50)], normed=True, vmin=0.0, vmax=3.0, cmap='jet')
    plt.colorbar()

plt.show()
