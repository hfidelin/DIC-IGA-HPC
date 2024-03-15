# %% Import
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import pickle
import time
import bsplyne as bs
# %%
spline, ctrl_pts = pickle.load(open("mesh/mesh_magma", "rb"))
#spline, ctrl_pts = bs.new_cylinder([0,0,1], [0, 0, 1], 1, 1)

degrees = spline.getDegrees()
knots = spline.getKnots()
m = BSplinePatch(ctrl_pts, degrees, knots)

# f.Plot()
cam = px.CameraVol([1, 0, 0, 0, 0, 0, 0])
refname = 'dataset_0_binning2.npy'
# defname = 'data_set_binning2_def.tiff'
f = px.Volume(refname).Load()
# f.BuildInterp()
P = m.Get_P()

# %%

m.Connectivity()
m.DVCIntegrationPixelElem(f, m, cam)

# %%

m.GaussIntegration([1,1,1])

XI, ETA, ZETA = m.Get_param_3D(cam)

# Get spline object for basis function
spline = m.Get_spline()


# Basis function at evaluation points
# phi = spline.DN(np.array([xi, eta, zeta]), k=[0, 0, 0])
print("évaluation des pt évaluation")
phi = spline.DN(np.array([XI, ETA, ZETA]), k=[0, 0, 0])
print("projection dans l'espace phy")



# initiating control points
P = m.Get_P()
# xi, eta, zeta = self.Get_param_3D(cam)
# Going from parametric space to physical space
x = phi @ P[:, 0]
y = phi @ P[:, 1]
z = phi @ P[:, 2]

del phi
# Going from physical space to parametric space
print("projection espace pix")
up, vp, wp = cam.P(x, y, z)
ur = np.round(up).astype('uint16')
vr = np.round(vp).astype('uint16')
wr = np.round(wp).astype('uint16')
# %% 
g = f.Copy()
g.pix *= 0

g.pix[ur, vr, wr] = 1
g.Plot()
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

g.pix[ur, vr, wr] = 1
g.pix[u, v, w] = 1
g.Plot()

# %%

