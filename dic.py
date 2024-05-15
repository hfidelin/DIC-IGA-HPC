# %% import
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import matplotlib.pyplot as plt
import time
import bsplyne as bs

f = px.Image('Images/zoom-0053_1.tif').Load()
g = px.Image('Images/zoom-0070_1.tif').Load()
cam = px.Camera([100, 6.95, -5.35, 0])
#cam = px.Camera([1, 0, 0, 0])
# %% Set up bspline
a = 0.925
b = 0.79
Xi = np.array([[0.5, b, 2*b],
               [0.5*a, b*a, 2*b*a],
               [0, 0, 0]])
Yi = np.array([[0, 0, 0],
               [0.5*a, b*a, 2*b*a],
               [0.5, b, 2*b]])

ctrlPts = np.array([Xi, Yi])
degree = [2, 2]
kv = np.array([0, 0, 0, 1, 1, 1])
knotVect = [kv, kv]

npt = 5
newr = np.linspace(0, 1, npt+2)[1:-1]

n = 10
newt = np.linspace(0, 1, npt+2)[1:-1]

n = np.c_[ctrlPts[0].ravel(),
          ctrlPts[1].ravel()]

e = {0 : np.arange(n.shape[0])}

m = BSplinePatch(e, n, degree, knotVect)
m.Connectivity()

m.KnotInsertion([newt, newr])


m.Plot()

# %%

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])


m.DICIntegrationPixel(f, cam)

xi, eta = m.InverseBSplineMapping(m.pgx, m.pgy)
# plt.axis('equal')
U, res = px.Correlate(f, g, m, cam, U0=U)

# u, v = cam.P(m.pgx, m.pgy)
# px.PlotMeshImage(f, m, cam)
# plt.scatter(v, u)


m.Plot(U, alpha=0.5)
m.Plot(U=3*U)

# %% DIC AVEC BSPLINE DÉGÉNÉRÉE


cam = px.Camera([100, 6.50, -2.35, 0])

a = 0.925
b = 0.79
Xi = np.array([[0, b, 2*b],
               [0, b*a, 2*b*a],
               [0, 0, 0]])
Yi = np.array([[0, 0, 0],
               [0, b*a, 2*b*a],
               [0, b, 2*b]])

ctrlPts = np.array([Xi, Yi])

n = np.c_[ctrlPts[0].ravel(),
          ctrlPts[1].ravel()]

e = {0 : np.arange(n.shape[0])}

degree = [2, 2]
kv = np.array([0, 0, 0, 1, 1, 1])
knotVect = [kv, kv]

m = BSplinePatch(e, n, degree, knotVect)

npt = 5
newr = np.linspace(0, 1, npt+2)[1:-1]

n = 10
newt = np.linspace(0, 1, npt+2)[1:-1]

m.KnotInsertion([newt, newr])
# m.Plot()
m.Connectivity()
m.RemoveDoubleNodes()
m.Connectivity()

#m.DICIntegrationPixel(f, cam)
m.DICIntegration()

# px.PlotMeshImage(f, m, cam)
U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
L = m.Laplacian()
U, res = px.Correlate(f, g, m, cam, U0=U, l0=10, L=L)


m.Plot(U, alpha=0.5)
m.Plot(U=3*U)

# px.PlotMeshImage(g, m, cam)
# ug, vg = cam.P(m.pgx, m.pgy)
# f.Plot()
# plt.plot(vg, ug, 'b.')










