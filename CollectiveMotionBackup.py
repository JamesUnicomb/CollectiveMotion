import numpy as np
import pycuda.compiler
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.curandom
from pycuda.compiler import SourceModule
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent

blocks     = 2**0
block_size = 2**8
n          = blocks * block_size
timestep   = 0.05
eps        = 1e-6

ra = 0.8
rb = 0.2
re = 0.5
r0 = 1.0
b  = 5.0
J  = 0.2
nc = 20

code   = open('CollectiveMotion.cu', 'r').read()
code   = ('#define nc %d \n' % (nc)) + code 
mod    = SourceModule(code)
func   = mod.get_function('update')

def initialise(n):
    # initialise random normal(0,1) starting position and 
    # chi-square(1) mass.
    X0    =  np.random.randn(n,3)
    X0    *= 1.0 / (np.linalg.norm(X0, axis=1).reshape(-1,1) + eps)
    rho   =  10.0 * np.power(np.random.uniform(0.0, n, size=(n,1)), 1.0 / 3.0)
    pos   = (rho * X0).astype(np.float32)

    # initialise random normal(0,1) starting velocity. 
    V0    =  np.random.randn(n,3)
    V0    *= 1.0 / (np.linalg.norm(V0, axis=1).reshape(-1,1) + eps)
    rho   =  np.power(np.random.uniform(0.0, 1.0, size=(n,1)), 1.0 / 3.0)
    vel   = (rho * V0).astype(np.float32)

    return pos, vel

pos_, vel_ = initialise(n = n)

# initalise update vectors
pos = np.zeros((n,3)).astype(np.float32)
vel = np.zeros((n,3)).astype(np.float32)

# initialise the opengl app
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.showMaximized()

sp2 = gl.GLScatterPlotItem(pos=pos_, 
                           #color=np.concatenate([vel_, np.ones((n,1))], axis=1),
                           )
w.addItem(sp2)

def update():
    global pos, vel, pos_, vel_, T

    n0    =  np.random.randn(n,3)
    n0    *= 1.0 / np.linalg.norm(n0, axis=1).reshape(-1,1)
    n0    = n0.astype(np.float32)

    func(drv.Out(pos), 
         drv.Out(vel), 
         drv.In(pos_), 
         drv.In(vel_), 
         drv.In(n0),
         np.int32(n), 
         np.float32(timestep), 
         np.float32(ra), 
         np.float32(rb), 
         np.float32(re), 
         np.float32(r0), 
         np.float32(b), 
         np.float32(J),
         grid=(blocks,1), block=(block_size,1,1))

    pos_ = pos
    vel_ = vel

    print pos[0], vel[0]

    sp2.setData(pos=pos_, 
                #color=np.concatenate([vel_, np.ones((n,1))], axis=1),
                )


# initialise the qt timer
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(30)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

q = ax.quiver(pos_[:,0], pos_[:,1], pos_[:,2],
                timestep * vel_[:,0], timestep * vel_[:,1], timestep * vel_[:,2])

plt.show()

