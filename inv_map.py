# %% import
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import bsplyne as bs
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon, LineString, Point
import time

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()
cam = px.Camera([100, 6.95, -5.35, 0])

# %% Set up bspline
a = 0.925
Xi = np.array([[0.5, 0.75, 1],
               [0.5*a, 0.75*a, 1*a],
               [0, 0, 0]])
Yi = np.array([[0, 0, 0],
               [0.5*a, 0.75*a, 1*a],
               [0.5, 0.75, 1]])

ctrlPts = np.array([Xi, Yi])
degree = [2, 2]
kv = np.array([0, 0, 0, 1, 1, 1])
knotVect = [kv, kv]

n = 5
newr = np.linspace(0, 1, n+2)[1:-1]
n = 10
newt = np.linspace(0, 1, n+2)[1:-1]
m = BSplinePatch(ctrlPts, degree, knotVect)
m.KnotInsertion([newt, newr])

xi_u = np.unique(m.knotVect[0])
eta_u = np.unique(m.knotVect[1])

ne_xi = np.unique(xi_u).shape[0] - 1
ne_eta = np.unique(eta_u).shape[0] - 1

phik, _, _, _ = m.ShapeFunctions(xi_u, eta_u)

P = m.Get_P()

xk = phik @ P[:, 0]
yk = phik @ P[:, 1]

u, v = cam.P(xk, yk)

u = u.reshape((12,7))
v = v.reshape((12,7))


px.PlotMeshImage(f, m, cam)
# plt.plot(v, u, 'r*')

# %% truc qui fonctionne

nbg_xi = 300
nbg_eta = 300
pxi = 1.0 / nbg_xi
peta = 1.0 / nbg_eta
xi_g = np.linspace(pxi, 1-pxi, nbg_xi)
eta_g = np.linspace(peta, 1-peta, nbg_eta)
phi, _, _, _ = m.ShapeFunctions(xi_g, eta_g)
P = m.Get_P()
pgx = phi @ P[:, 0] 
pgy = phi @ P[:, 1]
u1, v1 = cam.P(pgx, pgy)
px.PlotMeshImage(f, m, cam)
#plt.scatter(v1, u1, c='g', label='integration point')
#plt.legend()


u2 = u1.astype(int)
v2 = np.ceil(v1)
plt.scatter(v2, u2)



