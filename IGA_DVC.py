# %% Import
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import pickle
import time
import bsplyne as bs
import scipy.sparse as sps
# %%
spline, ctrl_pts = pickle.load(open("mesh/mesh_magma", "rb"))
#spline, ctrl_pts = bs.new_cylinder([0,0,1], [0, 0, 1], 1, 1)

degrees = spline.getDegrees()
knots = spline.getKnots()
m = BSplinePatch(ctrl_pts, degrees, knots)

cam = px.CameraVol([1, 0, 0, 0, 0, 0, 0])
refname = 'dataset_0_binning2.npy'
defname = 'dataset_1_binning2.npy'
f = px.Volume(refname).Load()
g = px.Volume(defname).Load()
f.BuildInterp()
g.BuildInterp()
P = m.Get_P()

# %%

m.Connectivity()
m.DVCIntegrationPixelElem(f, cam)
# m.DVCIntegrationPixel(f, cam)

u, v, w = cam.P(m.pgx, m.pgy, m.pgz)
u = np.round(u).astype('uint16')
v = np.round(v).astype('uint16')
w = np.round(w).astype('uint16')
g = f.Copy()
g.pix *= 0
g.pix[u, v, w] = 1
g.Plot()


# %%
U0 = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1], direct=False)
