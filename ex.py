#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:07:23 2024

@author: fidelin
"""
from paral_utils import DVCIntegrationPixelPara
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import pickle
import matplotlib.pyplot as plt

# from mpi4py import MPI
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
refname = 'dataset_0_binning4.npy'
defname = 'dataset_1_binning4.npy'
f = px.Volume(refname).Load()
g = px.Volume(defname).Load()
f.BuildInterp()
g.BuildInterp()
m.Connectivity()
m.KnotInsertion([1, 1, 1])
phi = DVCIntegrationPixelPara(m, f, cam)
P = m.Get_P()
if phi is not None :
    print(f"FINAL PHI : {phi.shape}")
    print(f"\nP : {P.shape}")
    u, v, w = cam.P(phi @ P[:, 0], phi @ P[:, 1], phi @ P[:, 2])
    # plt.spy(phi)
    # print(u)
    u = np.round(u).astype('uint16')
    v = np.round(v).astype('uint16')
    w = np.round(w).astype('uint16')
    print(u.min(), u.max())
    print(v.min(), v.max())
    print(w.min(), w.max())
    
    plt.scatter(v, w)
    
    # finte = f.Copy()
    # finte.pix *= 0
    # finte.pix[u, v, w] += 1
    # finte.Plot()
    plt.show()
# %%


# dic = {}

# N = 111
   
# for i in range(N):
#     dic[i] = i
    
# keys = list(dic.keys())    


# if rank == 0:
    
#     sendbuf = np.array_split(keys, size)
# else:
#     sendbuf = None
    
# local_keys = comm.scatter(sendbuf, root=0)
       
# for i in range(size):
#     comm.Barrier()
#     if rank == i:
#         print(f"Rank {rank} | {local_keys}")
# for key in local_keys:
#     print(f"Rank {rank} : {dic[key]}")