# %% import
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import matplotlib.pyplot as plt
import time

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()
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

n = 5
newr = np.linspace(0, 1, n+2)[1:-1]

n = 10
newt = np.linspace(0, 1, n+2)[1:-1]
m = BSplinePatch(ctrlPts, degree, knotVect)
m.KnotInsertion([newt, newr])


# e = m.Init_elem_2D()
# %%
areas = m.compute_elem_area(cam)
m.GaussIntegration([1,1])

# %%
m.Connectivity()
m.DICIntegrationPixel(f, m, cam)

u, v = cam.P(m.pgx, m.pgy)
px.PlotMeshImage(f, m, cam)
plt.scatter(v, u)


U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res = px.Correlate(f, g, m, cam, U0=U)

m.Plot(U, alpha=0.5)
m.Plot(U=3*U)
