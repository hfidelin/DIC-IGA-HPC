#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:13:04 2024

@author: fidelin
"""

import numpy as np
from mpi4py import MPI
import scipy.sparse as sps


def DVCIntegrationPixelPara(m, f, cam, P=None):
    
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if P is None :
        P = m.Get_P()
    
    spline = m.Get_spline()
    
    keys = list(m.e.keys())
    if rank == 0 :
        sendbuf = np.array_split(keys, size)
    else:
        sendbuf = None
        
    local_keys = comm.scatter(sendbuf, root=0)
    
    # print(f"Rank {rank} | {local_keys}")
    for key in local_keys:
        
        e = m.e[key]
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
        N = spline.DN([xi, eta, zeta], k=[0, 0, 0])
        u, v, w = cam.P(N @ P[:, 0], N @ P[:, 1], N @ P[:, 2])
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
        xi, eta, zeta = m.InverseBSplineMapping3D(xg, yg, zg, init=init, elem=e) 
        del xg, yg, zg, init
        """
        # Keeping the point which are not on the element bordrer
        select = (xi > e.xi[0]) & (xi < e.xi[1]) &\
             (eta > e.eta[0]) & (eta < e.eta[1]) &\
             (zeta > e.zeta[0]) & (zeta < e.zeta[1])
        
        xi = xi[select]
        eta = eta[select]
        zeta = zeta[select]
        del select
        """
        # Evaluating shape functions on the integration points 
        phi_loc = spline.DN(np.array([xi, eta, zeta]), k=[0,0,0])
        del xi, eta, zeta
        
        # Stacking the local evaluation matrix
        if key == local_keys[0]:
            phi = phi_loc
        else :
            phi = sps.vstack((phi, phi_loc))
        
    
    comm.Barrier()
    
    shape_i = comm.reduce(phi.shape[0], op=MPI.SUM, root=0)
    row, col, val = sps.find(phi)
    # print(f"\nRank {rank} row : \n {row}\n")
    # print(f"\nRank {rank} col : \n {col}\n")
    # print(f"\nRank {rank} val : \n {val}\n")
    
    row_size = np.empty(size, dtype=int)
    col_size = np.empty(size, dtype=int)
    val_size = np.empty(size, dtype=int)
        
    comm.Gather(np.array([row.shape[0]]), row_size, root=0)
    comm.Gather(np.array([col.shape[0]]), col_size, root=0)
    comm.Gather(np.array([val.shape[0]]), val_size, root=0)
    
    if rank == 0:
        total_row_size = sum(row_size)
        total_col_size = sum(col_size)
        total_val_size = sum(val_size)
        
        ROW = np.empty(total_row_size, dtype=row.dtype)
        COL = np.empty(total_col_size, dtype=col.dtype)
        VAL = np.empty(total_val_size, dtype=val.dtype)
        
        counts_row = np.array(row_size)
        displacements_row = np.zeros(size, dtype=int)
        displacements_row[1:] = np.cumsum(counts_row[:-1])
        
        counts_col = np.array(col_size)
        displacements_col = np.zeros(size, dtype=int)
        displacements_col[1:] = np.cumsum(counts_col[:-1])
        
        counts_val = np.array(val_size)
        displacements_val = np.zeros(size, dtype=int)
        displacements_val[1:] = np.cumsum(counts_val[:-1])
        

    else :  
        counts_row = np.empty(0)
        displacements_row = np.empty(0)
        ROW = np.empty(0, dtype=row.dtype)
        
        counts_col = np.empty(0)
        displacements_col = np.empty(0)
        COL = np.empty(0, dtype=col.dtype)
        
        counts_val = np.empty(0)
        displacements_val = np.empty(0)
        VAL = np.empty(0, dtype=val.dtype)
        
        
    
    
    row_to_send = (row, row.size, MPI.INT)
    row_to_recv = (ROW, tuple(counts_row), tuple(displacements_row), MPI.INT)
    
    col_to_send = (col, col.size, MPI.INT)
    col_to_recv = (COL, tuple(counts_col), tuple(displacements_col), MPI.INT)
    
    val_to_send = (val, val.size, MPI.INT)
    val_to_recv = (VAL, tuple(counts_val), tuple(displacements_val), MPI.INT)
    
    # print(f'Rank {rank} : count {tuple(counts_row)}')
    # print(f'Rank {rank} : displacements_row {tuple(displacements_row)}')
    print(f'Rank {rank} : row type {row.dtype}')
    print(f'Rank {rank} : ROW dtype {ROW.dtype}')
    comm.Barrier()
    comm.Gatherv(sendbuf=row_to_send, recvbuf=row_to_recv, root=0)
    comm.Gatherv(sendbuf=col_to_send, recvbuf=col_to_recv, root=0)
    comm.Gatherv(sendbuf=val_to_send, recvbuf=val_to_recv, root=0)

    if rank == 0:
        print(5 * '-', 'GATHERED', 5 * '-')
        print(f"\nROW : \n {ROW}\n")
        print(f"\nCOL : \n {COL}\n")
        print(f"\nVAL : \n {VAL}\n")
        shape_j = m.Get_nbf()
        phi = sps.csc_matrix((VAL, (ROW, COL)), shape=(shape_i, shape_j))
        # print(f"PHI FINAL : {phi.shape}")
        return phi
    else :
        return None
    for i in range(size):
        comm.Barrier()
        if rank == i:
            print(f"Rank \t {rank} \t {phi.shape}")

