import numpy as np
from scipy.spatial.distance import cdist

def probability(J, nc, pos, vel):
    amins = np.argsort(cdist(pos, pos) + 1e3 * np.eye(len(pos)), axis=1)[:,:nc]
    esum  =  np.exp(J / 2 * np.sum(np.dot(vel, np.swapaxes(vel[amins], 1,2)), axis=1))
    return esum / np.sum(esum)


def entropy(J, nc, pos, vel):
    prob = probability(J, nc, pos, vel)
    return -np.sum(prob * np.log(prob))

def fisher(J, nc, pos, vel, h=0.0025):
    prob = probability(J, nc, pos, vel)
    dprob = (probability(J + h, nc, pos, vel) - probability(J - h, nc, pos, vel)) / (2.0 * h)
    return np.sum(np.square(dprob))