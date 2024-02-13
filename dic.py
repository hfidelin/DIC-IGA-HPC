# %% import
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import matplotlib.pyplot as plt
import time

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()
cam = px.Camera([100, 6.95, -5.35, 0])

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

# %%

spline = m.Get_spline()
P = m.Get_P()

xiu = np.unique(m.knotVect[0])
etau = np.unique(m.knotVect[1])
phiu = spline.DN([xiu, etau], k=[0, 0])

ut, vt = cam.P(phiu @ P[:, 0], phiu @ P[:, 1])
#px.PlotMeshImage(f, m, cam)
# f.Plot()
#plt.scatter(vt, ut, c='b')
# f.Plot()
n = 0

N_vec = []
for i in range(xiu.shape[0]-1):
    xi0 = xiu[i]
    xi1 = xiu[i+1]
    for j in range(etau.shape[0]-1):

        eta0 = etau[j]
        eta1 = etau[j+1]

        phi = spline.DN([[xi0, xi1], [eta0, eta1]], k=[0, 0])

        u, v = cam.P(phi @ P[:, 0], phi @ P[:, 1])
        minu = min(u)
        maxu = max(u)

        minv = min(v)
        maxv = max(v)

        Y_coord = np.array([np.floor(minv), np.floor(maxv)])
        X_coord = np.array([np.floor(minu), np.floor(maxu)])

        area = (Y_coord[1] - Y_coord[0]) * (X_coord[1] - X_coord[0])
        N_pix = int(np.floor(np.sqrt(area)))
        N_vec.append(N_pix)

        n += area
        # plt.scatter(v, u)
        # plt.scatter(Y_coord, X_coord, label="boite")
        # plt.show()
        # plt.legend()

XI = []
for i in range(xiu.shape[0]-1):
    xi0 = xiu[i]
    xi1 = xiu[i+1]
    print(xi0, xi1)


#print(f"\nIl faudrait {n} points")

# %% Test bourrin


spline = m.Get_spline()
P = m.Get_P()

xi = np.linspace(0, 1, 350)
eta = np.linspace(0, 1, 350)
phi = spline.DN([xi, eta], k=[0, 0])

ub, vb = cam.P(phi @ P[:, 0], phi @ P[:, 1])

ub = np.round(ub)
vb = np.round(vb)

pixel = np.unique(np.array([ub, vb]).T, axis=0)

xg, yg = cam.PinvNL(pixel[:, 0], pixel[:, 1])

m.Plot()
plt.scatter(xg, yg)

xi_g, eta_g = m.InverseBSplineMapping(xg, yg)

# phit = spline.DN(np.array([xi_g, eta_g]), )


selec_xi = (xi_g>0) & (xi_g<1)
selec_eta = (eta_g>0) & (eta_g<1)

select = (xi_g>0) & (xi_g<1) & (eta_g>0) & (eta_g<1)

xi_gt = xi_g[select]
eta_gt = eta_g[select]


phit = spline.DN(np.array([xi_gt, eta_gt]), k=[0,0])

ut, vt = cam.P(phit @ P[:, 0], phit @ P[:, 1])


px.PlotMeshImage(f, m, cam) 
plt.scatter(vt, ut, c='y', label='test')
# %% Integration

m.Connectivity()
m.DICIntegrationPixel(m, cam)
u, v = cam.P(m.pgx, m.pgy)
px.PlotMeshImage(f, m, cam)
plt.plot(v, u, 'y.')
# %% Correlate

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res = px.Correlate(f, g, m, cam, U0=U)


m.Plot(U, alpha=0.5)
m.Plot(U=3*U)

px.PlotMeshImage(g, m, cam, U)
