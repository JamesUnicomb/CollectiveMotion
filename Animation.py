from CollectiveMotion import CollectiveMotionClass as CMClass
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent

cm = CMClass()

full = False

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

pos, vel = cm.initialise(n = n)

# initialise the opengl app
app = QtGui.QApplication([])
w = gl.GLViewWidget()
if full:
    w.showMaximized()
else:
    w.show()

sp2 = gl.GLScatterPlotItem(pos=pos, 
                           color=np.concatenate([vel, np.ones((n,1))], axis=1),
                           )
w.addItem(sp2)

K = 0
def update():
    global pos, vel, K

    n0    =  np.random.randn(n,3)
    n0    *= 1.0 / np.linalg.norm(n0, axis=1).reshape(-1,1)
    n0    = n0.astype(np.float32)

    pos, vel = cm.step(pos, vel, n, timestep,
                       ra, rb, re, r0, b, J,
                       blocks, block_size)

    pos -= np.mean(pos, axis=0)

    sp2.setData(pos=pos, 
                color=np.concatenate([vel, np.ones((n,1))], axis=1),
                )
    K += 1

    if K % 50 == 0:
        pos, vel = cm.initialise(n = n)


# initialise the qt timer
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(30)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()







