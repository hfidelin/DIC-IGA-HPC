#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:13:04 2024

@author: fidelin
"""
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
import scipy.sparse as sps
from tqdm import tqdm

def _compute_phi_pixel(m, f, cam, m2=None, P=None):
    """
    Create 
    
    Parameters
    ----------
    m : Bspline mesh
        DESCRIPTION.
    m2 : BSpline mesh, optional
        DESCRIPTION. The default is None.
    f : Volumic Image
        DESCRIPTION.
    cam : Camera model
        DESCRIPTION.
    P : Control Points, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    phi : TYPE
        DESCRIPTION.

    """
    
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if m2 is None:
        m2 = m
    
    if P is None :
        P2 = m2.Get_P()
    
    spline2 = m2.Get_spline()
    spline = m.Get_spline()
    
    keys = list(m2.e.keys())
    if rank == 0 :
        sendbuf = np.array_split(keys, size)
    else:
        sendbuf = None
        
    local_keys = comm.scatter(sendbuf, root=0)
    progress_bar = tqdm(total=len(local_keys), desc=f"Process {rank}", position=rank)
    # print(f"Rank {rank} | {local_keys}")
    for key in local_keys:
        
        e = m2.e[key]
        # print(f"\nÉlément {key} :")
        
        # Compute largest eedge of the element
        N_eval = m.compute_largest_edge(cam, e)
        
        # Inflating the number of eval point to ensure to have all pixels
        N_eval = int(N_eval)
        # print(f"N_eval = {N_eval ** 3}")
        
        # Setting the eval points to find all pxel center
        xi = np.linspace(e.xi[0], e.xi[1], N_eval)
        eta = np.linspace(e.eta[0], e.eta[1], N_eval)
        zeta = np.linspace(e.zeta[0], e.zeta[1], N_eval)
        
        # Going from parametric space to image space
        N = spline2.DN([xi, eta, zeta], k=[0, 0, 0])
        u, v, w = cam.P(N @ P2[:, 0], N @ P2[:, 1], N @ P2[:, 2])
        del N
        
        # Rounding the projected points 
        ur = np.round(u).astype('uint16')
        vr = np.round(v).astype('uint16')
        wr = np.round(w).astype('uint16')
        del u, v, w
        
        # Number the rounded points
        Nx = f.pix.shape[0]
        Ny = f.pix.shape[1]
        Nz = f.pix.shape[2]
        
        
        idpix = np.ravel_multi_index((ur, vr, wr), (Nx, Ny, Nz))
        _, rep = np.unique(idpix, return_index=True)
        del idpix, Nx, Ny, Nz
        ur = ur[rep]
        vr = vr[rep]
        wr = wr[rep]
        
        
        param = np.meshgrid(xi, eta, zeta, indexing='ij')
        xi_init = param[0].ravel()[rep]
        eta_init = param[1].ravel()[rep]
        zeta_init = param[2].ravel()[rep]
        del param
        init = [xi_init, eta_init, zeta_init]
        xg, yg, zg = cam.Pinv(ur.astype(float), vr.astype(float), wr.astype(float))
        del ur, vr, wr
        
        # Inversing Bspline Mapping
        xi, eta, zeta = m2.InverseBSplineMapping3D(xg, yg, zg, init=init, elem=e) 
        del xg, yg, zg, init

        # Evaluating shape functions on the integration points 
        phi_loc = spline.DN(np.array([xi, eta, zeta]), k=[0,0,0])
        del xi, eta, zeta
        
        # Stacking the local evaluation matrix
        if key == local_keys[0]:
            phi = phi_loc
        else :
            phi = sps.vstack((phi, phi_loc))
        progress_bar.update(1)
    
    progress_bar.close()    
    
    comm.Barrier()
    
    print(f"Rank {rank} : phi shpae : {phi.shape}")
    
    if rank == 0:
        list_phi = np.empty(size, dtype=phi.dtype)
    else:
        list_phi = None
    
    comm.Barrier()
    print(f"Rank {rank} : JUSQU'ICI TOUT VA BIEN")
    list_phi = comm.gather(phi, root=0)
    if rank == 0:
        for i in range(len(list_phi)):
            phi = list_phi[i]
            if i == 0:
                PHI = phi
            else:
                PHI = sps.vstack((PHI, phi))
        
        return PHI
    
    else:
        return None
    



def _compute_phi(m, f, cam, m2=None, P=None):
    
    comm = PETSc.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    
    
    

    if m2 is None:
        m2 = m
    
    if P is None :
        P2 = m2.Get_P()
    
    spline2 = m2.Get_spline()
    spline = m.Get_spline()
    
    keys = list(m2.e.keys())
    
    list_keys = np.array_split(keys, size)
    
    local_keys = list_keys[rank]
    progress_bar = tqdm(total=len(local_keys), desc=f"Process {rank}", position=rank)
    for key in local_keys:
        
        e = m2.e[key]
        
        # Compute largest eedge of the element
        N_eval = m.compute_largest_edge(cam, e)
        
        # Inflating the number of eval point to ensure to have all pixels
        N_eval = int(N_eval)
        # print(f"N_eval = {N_eval ** 3}")
        
        # Setting the eval points to find all pxel center
        xi = np.linspace(e.xi[0], e.xi[1], N_eval)
        eta = np.linspace(e.eta[0], e.eta[1], N_eval)
        zeta = np.linspace(e.zeta[0], e.zeta[1], N_eval)
        
        # Going from parametric space to image space
        N = spline2.DN([xi, eta, zeta], k=[0, 0, 0])
        u, v, w = cam.P(N @ P2[:, 0], N @ P2[:, 1], N @ P2[:, 2])
        del N
        
        # Rounding the projected points 
        ur = np.round(u).astype('uint16')
        vr = np.round(v).astype('uint16')
        wr = np.round(w).astype('uint16')
        del u, v, w
        
        # Number the rounded points
        Nx = f.pix.shape[0]
        Ny = f.pix.shape[1]
        Nz = f.pix.shape[2]
        
        
        idpix = np.ravel_multi_index((ur, vr, wr), (Nx, Ny, Nz))
        _, rep = np.unique(idpix, return_index=True)
        del idpix, Nx, Ny, Nz
        ur = ur[rep]
        vr = vr[rep]
        wr = wr[rep]
        
        
        param = np.meshgrid(xi, eta, zeta, indexing='ij')
        xi_init = param[0].ravel()[rep]
        eta_init = param[1].ravel()[rep]
        zeta_init = param[2].ravel()[rep]
        del param
        init = [xi_init, eta_init, zeta_init]
        xg, yg, zg = cam.Pinv(ur.astype(float), vr.astype(float), wr.astype(float))
        del ur, vr, wr
        
        # Inversing Bspline Mapping
        xi, eta, zeta = m2.InverseBSplineMapping3D(xg, yg, zg, init=init, elem=e) 
        del xg, yg, zg, init

        # Evaluating shape functions on the integration points 
        phi_loc_elem = spline.DN(np.array([xi, eta, zeta]), k=[0,0,0])
        del xi, eta, zeta
        
        # Stacking the local evaluation matrix
        if key == local_keys[0]:
            phi_loc = phi_loc_elem
        else :
            phi_loc = sps.vstack((phi_loc, phi_loc_elem))
        progress_bar.update(1)
    # print(f"Rank {rank} phi loc shape : {phi_loc.shape}")
    progress_bar.close() 
    
    
    
    sendbuf = np.array(phi_loc.shape[0], dtype='int64')
    recvbuf = (np.array(0, dtype='int64'), MPI.INT)
    MPI.COMM_WORLD.Reduce(sendbuf=sendbuf, recvbuf=recvbuf, root=0, op=MPI.SUM)
    MPI.COMM_WORLD.Bcast([recvbuf[0], MPI.INT], root=0)
  
    
    shape_i = recvbuf[0]
    shape_j = m.Get_nbf()
    
    row_phi = phi_loc.indptr
    col_phi = phi_loc.indices
    data_phi = phi_loc.data
    phi_glob = PETSc.Mat()
    # print(f"Rank {rank} : {row_phi}")
    # print(f"Rank {rank} : {col_phi}")
    # print(f"Rank {rank} : {data_phi}")
    phi_glob.setValuesLocalCSR(row_phi, col_phi, data_phi, addv=None)
    # phi_glob.createAIJWithArrays(((PETSc.DECIDE, shape_i), (PETSc.DECIDE, shape_j)),
    #                              (row_phi, col_phi, data_phi), comm=comm)
    # phi_glob.createIS(((PETSc.DECIDE, shape_i), (PETSc.DECIDE, shape_j)), comm=comm)
    # phi_glob.setISLocalMat(phi_loc)
    comm.barrier()
    if rank == 0:
        return phi_glob

