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
spline = m.Get_spline()
P = m.Get_P()
# %%

xi = np.linspace(0,1,700)
eta = xi
param = np.meshgrid(xi, eta, indexing='ij')
print("1")
N = spline.DN([xi, eta], k=[0,0])

u, v = cam.P(N@ P[:, 0], N@P[:,1])
del N
up = np.round(u)
vp = np.round(v)
print("2")
# Placing evalution points in image space in the center of pixels
ur = np.round(up).astype('uint16')
vr = np.round(vp).astype('uint16')

Nx = f.pix.shape[0]
Ny = f.pix.shape[1]

# idpix = - Nx * vr + ur
idpix = np.ravel_multi_index((ur, vr), (Nx, Ny))
_, rep = np.unique(idpix, return_index=True)
        
u = ur[rep]
v = vr[rep]

xi_init = param[0].ravel()[rep]
eta_init = param[1].ravel()[rep]
N = spline.DN(np.array([xi_init, eta_init]), k=[0,0])
u_init, v_init = cam.P(N@ P[:, 0], N@P[:,1])

# Going from pixel space to the physical space by inversing camera model
xg, yg = cam.Pinv(u.astype(float), v.astype(float))

# Going from physical space to parametric space by inversing mapping
xi, eta = m.InverseBSplineMapping(xg, yg, init=[xi_init, eta_init])
# %%

# plt.scatter(xi, eta, c='k', label=r'$(\xi_g, \eta_g)$')
# plt.axis('equal')
# plt.legend()

N = spline.DN(np.array([xi, eta]), k=[0,0])
u, v = cam.P(N@ P[:, 0], N@P[:,1])
px.PlotMeshImage(f, m, cam)
plt.scatter(v, u, c='b', label="Interrogation Points")
# plt.scatter(v_init, u_init, c='g', label="Initialized Points")
plt.legend()

# %%
m.Connectivity()
U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])


m.DICIntegrationPixelElem(f, cam)

xi, eta = m.InverseBSplineMapping(m.pgx, m.pgy)
plt.plot(xi, eta, 'k.')
# plt.axis('equal')
U, res = px.Correlate(f, g, m, cam, U0=U)

u, v = cam.P(m.pgx, m.pgy)
px.PlotMeshImage(f, m, cam)
plt.scatter(v, u)


m.Plot(U, alpha=0.5)
m.Plot(U=3*U)
