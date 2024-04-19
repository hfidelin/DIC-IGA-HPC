#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:10:33 2024

@author: fidelin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:36:15 2024

@author: fidelin
"""
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import pickle
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    print(f"\nTHERE IS {size} CPU\n")
comm.Barrier()
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


spline_ref = m.Get_spline()
if rank == 0 :
    spline_ref.saveParaview(m.N2CtrlPts(m.n), './', 'reference')
# %% Initialisation
m.Connectivity()
if rank == 0: 
    U0 = px.MultiscaleInit(f, g, m, cam, scales=[2, 1], direct=False)
#U0 = np.zeros((m.n.shape[0] * 3))
# %% Building evaluation points
m2 = m.Copy()
m2.KnotInsertion([2, 3, 2])
m.DVCIntegrationPixelPara(f, cam, m2=m2)

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
if rank == 0 : 
    L = m.Laplacian()
    U, res = px.Correlate(f, g, m, cam, U0=U0, l0=15, L=L, direct=False)
# m.saveParaview('deformed', cam, U)
# fsub = f.Copy()
# fsub.SubSample(3)
# gsub = g.Copy()
# gsub.SubSample(3)
# camsub = cam.SubSampleCopy(3)

# px.PlotMeshImage3d(g, m, cam, U=U)
