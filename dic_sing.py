#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:34:07 2024

@author: fidelin
"""

import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import matplotlib.pyplot as plt
import bsplyne as bs

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()

cam = px.Camera([100, 7, -2.3, 0.1])

spline, ctrl_pts = bs.new_disk([0, 0, 0], [0, 0, 1], 1)
spline, ctrl_pts = bs.new_quarter_cylinder([0, 0,0], [0, 0, 1], 1, 1)



ctrl_pts = ctrl_pts[:-1]
degree = spline.getDegrees()
knotVect = spline.getKnots()

m = BSplinePatch(ctrl_pts, degree, knotVect)

m.Connectivity()

init, xi, eta = m.DICIntegrationPixel(f, m, cam)

xi_init = init[0]
eta_init = init[1]

P = m.Get_P()
N = spline.DN(np.array([xi_init, eta_init]), k=[0, 0])
u_init, v_init = cam.P(N @ P[:, 0], N @ P[:, 1])
u, v = cam.P(m.pgx, m.pgy)


px.PlotMeshImage(f, m, cam)
plt.scatter(v, u, c='b', label="Points convergés")
plt.scatter(v_init, u_init, c='r', label="Initialisation")
plt.legend()

# %% Visualisation de l'espace paramétrique

u, v = cam.P(m.pgx, m.pgy)
px.PlotMeshImage(f, m, cam)
plt.scatter(v, u)
# plt.grid()


# %%

xi2 = np.array([0, 1])
eta2 = np.array([0, 1])

N = spline.DN([xi2, eta2], k=[0, 0])

u2, v2 = cam.P(N @ P[:, 0], N @ P[:, 1])
px.PlotMeshImage(f, m, cam) 
plt.scatter(v2, u2, c='r')

# %%

