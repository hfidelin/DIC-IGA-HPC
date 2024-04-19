#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:36:15 2024

@author: fidelin
"""

import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import pickle
import time
import scipy.sparse as sps
import matplotlib.pyplot as plt

# %%

spline, ctrl_pts = pickle.load(open("mesh/mesh_magma", "rb"))
#spline, ctrl_pts = bs.new_cylinder([0,0,1], [0, 0, 1], 1, 1)

degrees = spline.getDegrees()
knots = spline.getKnots()
m = BSplinePatch(ctrl_pts, degrees, knots)

cam = px.CameraVol([1, 0, 0, 0, 0, 0, 0])
refname = 'Images/dataset_0_binning2.npy'
defname = 'Images/dataset_1_binning2.npy'
f = px.Volume(refname).Load()
g = px.Volume(defname).Load()
f.BuildInterp()
g.BuildInterp()
m.KnotInsertion([1, 1, 1])
    

# %%
# m.KnotInsertion([1, 1, 1])
spline_ref = m.Get_spline()
spline_ref.saveParaview(m.N2CtrlPts(m.n), './', 'reference')

# %% Initialisation
m.Connectivity()
U0 = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1], direct=False)
#U0 = np.zeros((m.n.shape[0] * 3))
# %% Building evaluation points

m.DVCIntegration(8)

m2 = m.Copy()
m2.KnotInsertion([2, 2, 2])
m.DVCIntegrationPixel(f, cam, fname="phi_binning2_211.npz")

u, v, w = cam.P(m.pgx, m.pgy, m.pgz)
u = np.round(u).astype('uint16')
v = np.round(v).astype('uint16')
w = np.round(w).astype('uint16')
finte = f.Copy()
finte.pix *= 0
finte.pix[u, v, w] += 1
finte.Plot()
# finte.Save()

# %%
L = m.Laplacian()
U, res = px.Correlate(f, g, m, cam, U0=U0, l0=15, L=L, direct=False)
m.saveParaview('deformed', cam, U)

# %%
n = m.n
nu, rep = np.unique(n, axis=0, return_index=True)

Ux = U[:m.nbf]
Uy = U[m.nbf:-m.nbf]
Uz = U[-m.nbf:] 

Uxr = Ux[rep]
Uyr = Uy[rep]
Uzr = Uz[rep]


# fsub = f.Copy()
# fsub.SubSample(3)
# gsub = g.Copy()
# gsub.SubSample(3)
# camsub = cam.SubSampleCopy(3)

# px.PlotMeshImage3d(g, m, cam, U=U)
