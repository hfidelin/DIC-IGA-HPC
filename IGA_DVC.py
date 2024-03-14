# %% Import
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import pickle
import matplotlib.pyplot as plt
import time
import vtk

# %%
spline, ctrl_pts = pickle.load(open("mesh/mesh_magma", "rb"))

#spline, ctrl_pts = bs.new_cylinder([0,0,1], [0, 0, 1], 1, 3)

degrees = spline.getDegrees()
knots = spline.getKnots()
m = BSplinePatch(ctrl_pts, degrees, knots)
# m.KnotInsertion([2, 0, 1])


# f.Plot()
cam = px.CameraVol([1, 0, 0, 0, 0, 0, 0])
refname = 'dataset_0_binning2.npy'
# defname = 'data_set_binning2_def.tiff'
f = px.Volume(refname).Load()
# f.BuildInterp()
P = m.Get_P()

# %%

init = m.DVCIntegrationPixel(f, cam)
m.Connectivity()



# %%
g = f.Copy()

g.pix *= 0
"""
u, v, w = cam.P(m.pgx, m.pgy, m.pgz)
u, v, w = cam.P(m.pgx, m.pgy, m.pgz)
u = u.astype(int)
v = v.astype(int)
w = w.astype(int)
"""
xi = init[0]
eta = init[1]
zeta = init[2]
N = spline.DN(np.array([xi, eta, zeta]), k=[0, 0, 0])

u, v, w = cam.P(N @ P[:, 0], N @ P[:, 1], N @ P[:, 2])

u = u.astype(int)
v = v.astype(int)
w = w.astype(int)

g.pix[u, v, w] = 1
g.Plot()
# %%
g = px.Volume(defname).Load()
f.BuildInterp()
g.BuildInterp()
U0 = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1], direct=False)

# %%

