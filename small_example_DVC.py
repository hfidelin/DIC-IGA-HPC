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

spline, ctrl_pts = pickle.load(open("mesh/mesh_magma_small", "rb"))
#spline, ctrl_pts = bs.new_cylinder([0,0,1], [0, 0, 1], 1, 1)

degrees = spline.getDegrees()
knots = spline.getKnots()

n = np.c_[ctrl_pts[0].ravel(),
          ctrl_pts[1].ravel(),
          ctrl_pts[2].ravel()]

e = {0 : np.arange(n.shape[0])}

m = BSplinePatch(e, n, degrees, knots)

cam = px.CameraVol([1, 0, 0, 0, 0, 0, 0])
refname = 'Images/dataset_0_binning4.npy'
defname = 'Images/dataset_1_binning4.npy'
f = px.Volume(refname).Load()
g = px.Volume(defname).Load()
f.BuildInterp()
g.BuildInterp()
m.KnotInsertion([1, 1, 1])
spline = m.spline
m.Connectivity()
m.RemoveDoubleNodes()
# m.RemoveDoubleNodesKnots()
m.Connectivity()         

# %%

#m.DVCIntegrationPixelElem(f, cam)

m.DVCIntegrationPixel(f, cam)
# %%


for key in m.elem.keys():
    elem = m.elem[key]
    # print(elem.xig.shape, elem.etag.shape, elem.zetag.shape)
    # print(elem.pgx.shape, elem.pgy.shape, elem.pgz.shape)
    # print('\n')
    
    phi = spline.DN(np.array([elem.xig, elem.etag, elem.zetag]))    
    pgu, pgv, pgw = cam.P(elem.pgx, elem.pgy, elem.pgz)
    feval = f.Interp(pgu, pgv, pgw)
    fdxr, fdyr, fdzr = f.InterpGrad(pgu, pgv, pgw)
    Jxx, Jxy, Jxz, Jyx, Jyy, Jyz, Jzx, Jzy, Jzz = cam.dPdX(elem.pgx, 
                                                           elem.pgy, 
                                                           elem.pgz)
    
    #print(fdxr.shape, Jxx.shape)
    
    phiJdfx = sps.diags(fdxr * Jxx + fdyr * Jyx + fdzr * Jzx) @ phi
    phiJdfy = sps.diags(fdxr * Jxy + fdyr * Jyy + fdzr * Jzy) @ phi
    phiJdfz = sps.diags(fdxr * Jxz + fdyr * Jyz + fdzr * Jzz) @ phi
    phiJdf = sps.hstack((phiJdfx, phiJdfy, phiJdfz), 'csc')    

    
    wphiJdf = sps.diags(elem.wdetJ) @ phiJdf
    print(sps.diags(elem.wdetJ).shape, phiJdf.shape, '\n')
    # print(wphiJdf.shape)
    # dyn = np.max(feval) - np.min(feval)
    # mean0 = np.mean(feval)
    # std0  = np.std(feval)
    # feval     -= mean0



# %%

for key in m.elem.keys():
    elem = m.elem[key]
    
    phi = spline.DN(np.array([elem.xig, elem.etag, elem.zetag]))
    U = np.zeros(phi.shape[1])
        

    x =  elem.pgx + phi @ U
    y =  elem.pgy + phi @ U
    z =  elem.pgz + phi @ U
         
    
    pgu, pgv, pgw = cam.P(x, y, z)
    res = g.Interp(pgu, pgv, pgw)
    feval = f.Interp(pgu, pgv, pgw)
    # res = g.Interp(x,y,z) 
    res -= np.mean(res)
    std1 = np.std(res)
    # res = feval - self.std0 / std1 * res
    # b = self.wphidf.T @ res
    print(wphiJdf.T.shape, res.shape)
    # b = wphiJdf.T @ res



# %%
# m.KnotInsertion([1, 1, 1])
spline_ref = m.Get_spline()
spline_ref.saveParaview(m.N2CtrlPts(m.n), './', 'reference')

# %% Initialisation
m.Connectivity()

U0 = px.MultiscaleInit(f, g, m, cam, scales=[2, 1], direct=False)
#U0 = np.zeros((m.n.shape[0] * 3))
# %% Building evaluation points

# m.DVCIntegration(8)

# m2 = m.Copy()
# m2.KnotInsertion([2, 2, 2])
# m.DVCIntegrationPixel(f, cam, fname="phi_binning2_211.npz")

# u, v, w = cam.P(m.pgx, m.pgy, m.pgz)
# u = np.round(u).astype('uint16')
# v = np.round(v).astype('uint16')
# w = np.round(w).astype('uint16')
# finte = f.Copy()
# finte.pix *= 0
# finte.pix[u, v, w] += 1
# finte.Plot()
# finte.Save()

# %%
L = m.Laplacian()
U, res = px.Correlate(f, g, m, cam, U0=U0, l0=25, L=L, direct=False)
m.saveParaview('deformed', cam, U)

