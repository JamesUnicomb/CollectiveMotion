EPS = 1e-6
nc  = 20

import numpy as np
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule

class CollectiveMotionClass:
    def __init__(self,
                 nc  = 20,
                 EPS = 1e-6):

        code      = open('CollectiveMotion.cu', 'r').read()
        code      = ('#define nc %d \n' % (nc)) + code 
        mod       = SourceModule(code)
        self.func = mod.get_function('update')

    def set_nc(self, 
               nc):
        code      = open('CollectiveMotion.cu', 'r').read()
        code      = ('#define nc %d \n' % (nc)) + code 
        mod       = SourceModule(code)
        self.func = mod.get_function('update')

    def initialise(self, 
                   n):
        # initialise random normal(0,1) starting position and 
        # chi-square(1) mass.
        X0    =  np.random.randn(n,3)
        X0    *= 1.0 / (np.linalg.norm(X0, axis=1).reshape(-1,1) + EPS)
        Xrho  =  2.0 * np.power(np.random.uniform(0.0, n, size=(n,1)), 1.0 / 3.0)
        pos   = (Xrho * X0).astype(np.float32)

        # initialise random normal(0,1) starting velocity. 
        V0    =  np.random.randn(n,3)
        V0    *= 1.0 / (np.linalg.norm(V0, axis=1).reshape(-1,1) + EPS)
        Vrho  =  np.power(np.random.uniform(0.0, 1.0, size=(n,1)), 1.0 / 3.0)
        vel   = (Vrho * V0).astype(np.float32)

        return pos, vel

    def step(self,
             pos,
             vel,
             n, 
             timestep,
             ra, 
             rb, 
             re, 
             r0, 
             b, 
             J,
             blocks,
             block_size):
        pos_ = np.zeros((n,3)).astype(np.float32)
        vel_ = np.zeros((n,3)).astype(np.float32)

        n0    =  np.random.randn(n,3)
        n0    *= 1.0 / np.linalg.norm(n0, axis=1).reshape(-1,1)
        n0    = n0.astype(np.float32)

        self.func(drv.Out(pos_), 
                  drv.Out(vel_), 
                  drv.In(pos), 
                  drv.In(vel), 
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

        return pos_, vel_

