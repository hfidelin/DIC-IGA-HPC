#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:07:23 2024

@author: fidelin
"""
from paral_utils import _compute_phi_pixel, _compute_phi
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import pickle
import scipy.sparse as sps
#from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# if rank == 0:
#     print(f"\nTHERE IS {size} CPU\n")
# comm.Barrier()



spline, ctrl_pts = pickle.load(open("mesh/mesh_magma_small", "rb"))
#spline, ctrl_pts = bs.new_cylinder([0,0,1], [0, 0, 1], 1, 1)

degrees = spline.getDegrees()
knots = spline.getKnots()
m = BSplinePatch(ctrl_pts, degrees, knots)

cam = px.CameraVol([1, 0, 0, 0, 0, 0, 0])
refname = 'Images/dataset_0_binning4.npy'
defname = 'Images/dataset_1_binning4.npy'
f = px.Volume(refname).Load()
g = px.Volume(defname).Load()
f.BuildInterp()
g.BuildInterp()
m.Connectivity()

phi = _compute_phi(m, f, cam)

print(phi.shape)
"""
if rank == 0:
    m.DVCIntegrationPixel(f, cam)
    phi_ref = m.phi
    
m2 = m.Copy()
m2.KnotInsertion([3, 3, 3])   
phi = _compute_phi_pixel(m, f, cam, m2=m2)
P = m.Get_P()


if phi is not None :
    if rank == 0:
        print(f"\nFINAL PHI : {phi.shape}")
        u, v, w = cam.P(phi @ P[:, 0], phi @ P[:, 1], phi @ P[:, 2])
        u_ref, v_ref, w_ref = cam.P(phi_ref @ P[:, 0], phi_ref @ P[:, 1], phi_ref @ P[:, 2])
        
        # plt.spy(phi)
        # print(u)
        u = np.round(u).astype('uint16')
        v = np.round(v).astype('uint16')
        w = np.round(w).astype('uint16')
        np.save('u.npy', u)
        np.save('v.npy', v)
        np.save('w.npy', w)
        # print(f"OBTENU {u.min(), u.max()}")
        # print(f"OBTENU {v.min(), v.max()}")
        # print(f"OBTENU {w.min(), w.max()}")
        
        u_ref = np.round(u_ref).astype('uint16')
        v_ref = np.round(v_ref).astype('uint16')
        w_ref = np.round(w_ref).astype('uint16')
        print(f"REF {u_ref.min(), u_ref.max()} \t OBTENU {u.min(), u.max()}")
        print(f"REF {v_ref.min(), v_ref.max()} \t OBTENU {v.min(), v.max()}")
        print(f"REF {w_ref.min(), w_ref.max()} \t OBTENU {w.min(), w.max()}")
        print(f"\nPIX SHAPE : {f.pix.shape}")
    # print(f"\n VERIF : {(phi != phi_ref).nnz==0}")
    
    # finte = f.Copy()
    # finte.pix *= 0
    # finte.pix[u, v, w] += 1
    # finte.Plot()
    # plt.show()
"""
# %%


# n = np.array([[2.5, 3.5, 1.2],
#               [6.4, 3.5, 23.2],
#               [2.5, 3.5, 1.2],
#               [23.4, 1.2, 2.2],
#               [12.4, 4., 23.5],
#               [12.4, 4., 23.5]])


# nu, ind = np.unique(n, axis=0, return_index=True)
# nu = nu[np.argsort(ind)]
# p = np.zeros(m.n.shape[0])
# for i in range(len(n)):
    
#     for j in range(len(nu)):
#         if (n[i] == nu[j]).all():
#             index = j
    
#     print(index)  
#     p[i] = index