# -*- coding: utf-8 -*-
"""
Bspline patch Digital Image Correlation method

@author: A. Rouawne, H. Fidelin INSA Toulouse, 2024

pyxel

PYthon library for eXperimental mechanics using Finite ELements

"""

# %% import
import os
import numpy as np
import scipy as sp
# from .bspline_routines import bspdegelev, bspkntins, global_basisfuns, Get2dBasisFunctionsAtPts, global_basisfunsWd
import scipy.sparse as sps
import matplotlib.pyplot as plt
from pyxel.camera import Camera
from pyxel.mesher import StructuredMeshQ4
from pyxel import BSplinePatch as BSplinePatch_ref
import bsplyne as bs
from pyxel.bspline_routines import global_basisfuns


def scatter(ctrlP):
    if ctrlP.shape[0] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        ax.scatter(*ctrlP)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        plt.show()
    elif ctrlP.shape[0] == 2 :
        plt.scatter(*ctrlP)
        plt.show()
    

class BSplinePatch(object):
    def __init__(self, ctrlPts, degree, knotVect):
        """
        Nurbs surface from R^2 (xi,eta)--->R^2 (x,y) 
        ctrlPts = [X,Y] or ctrlPts = [X,Y,Z]
        """
        self.dim = ctrlPts.shape[0]
        self.ctrlPts = ctrlPts
        self.n = self.CrtlPts2N()
        self.degree = np.array(degree)
        self.knotVect = knotVect

        self.ien = 0              # NURBS Connectivity p-uplet (IENu,IENv)
        # Connectivity: Control Point number (column) of each element (line)
        self.noelem = 0
        self.tripleien = 0        # Correspondance between 2D elements and 1D elements
        self.iperM = 0            # Sparse matrix for coincident control points

        # Dictionary containing the value det(Jacobian)*Gauss_Weights*ElementMeasure/4
        self.wdetJmes = 0
        # Dictionary containing the measure of the elements in the ordre of listeltot
        self.mes = 0

        # Dictionary of basis functions evaluated at gauss points
        self.phi = np.empty(0)
        # Dictionary containing the derivative of Basis function in x direction
        self.dphidx = np.empty(0)
        # Dictionary containing the derivative of Basis function in y direction
        self.dphidy = np.empty(0)

        """ Attributes when using vectorization  """
        """ In this case, the implicit connectivity of the structured B-spline parametric space is used """
        self.npg = 0
        self.phix = np.empty(0)      # Matrix (N,0)
        self.phiy = np.empty(0)      # Matrix (0,N)
        
        self.phiz = np.empty(0)      # Matrix ?????
        
        self.dphixdx = np.empty(0)  # Matrix (dNdx,0)
        self.dphixdy = np.empty(0)  # Matrix (dNdy,0)
        
        self.dphixdz = np.empty(0)  # Matrix ?????
        
        self.dphiydx = np.empty(0)  # Matrix (0,dNdx)
        self.dphiydy = np.empty(0)  # Matrix (0,dNdy)
        
        self.dphiydz = np.empty(0)  # Matrix ????
        
        self.dphizdx = np.empty(0)  # Matrix ?????
        self.dphizdy = np.empty(0)  # Matrix ?????
        self.dphizdz = np.empty(0)  # Matrix ?????
        
        
        
        self.wdetJ = np.empty(0)        # Integration weights diagonal matrix
        self.pgx = np.empty(0)
        self.pgy = np.empty(0)
        self.pgz = np.empty(0)

        self.phiMatrix = 0
        self.n_elems = 0
        self.pix = 0
        self.piy = 0
        self.piz = 0
        self.integrationCellsCoord = 0
        """ fixed parameters for integration """
        self.nbg_xi = 0
        self.nbg_eta = 0
        self.nbg_zeta = 0
        self.Gauss_xi = 0
        self.Gauss_eta = 0
        self.Gauss_zeta = 0
        self.wgRef = 0
        self.refGaussTriangle = 0


    def Copy(self):
        m = BSplinePatch(self.ctrlPts.copy(), self.degree.copy(),
                         self.knotVect.copy())
        m.conn = self.conn.copy()
        m.ndof = self.ndof
        m.dim = self.dim
        m.npg = self.npg
        m.pgx = self.pgx.copy()
        m.pgy = self.pgy.copy()
        m.phix = self.phix.copy()
        m.phiy = self.phiy.copy()
        m.wdetJ = self.wdetJ.copy()
        return m
    
    def IsRational(self):
        return (self.ctrlPts[3] != 1).any()

    def Get_nbf_1d(self):
        """ Get the number of basis functions per parametric direction """
        return self.ctrlPts.shape[1:]

    def Get_nbf(self):
        """ Total number of basis functions """
        return np.product(self.Get_nbf_1d())

    def Get_nbf_elem_1d(self):
        return self.degree + 1

    def Get_nbf_elem(self):
        return np.product(self.degree+1)

    def Get_listeltot(self):
        """ Indices of elements """
        return np.arange(self.ien[0].shape[1]*self.ien[1].shape[1])

    def Get_P(self):
        """  Returns the total"""
        if self.dim == 2:
            P = np.c_[self.ctrlPts[0].ravel(order='F'),
                  self.ctrlPts[1].ravel(order='F')]
        
        elif self.dim == 3:
            P = np.c_[self.ctrlPts[0].ravel(order='F'),
                  self.ctrlPts[1].ravel(order='F'),
                  self.ctrlPts[2].ravel(order='F')]
        return P
  

        """
        Il manque ici les fonctions SelectNodes et Select Lines
        """

  
    def Connectivity(self):
        nn = len(self.n)
        self.ndof = nn * self.dim
        if self.dim == 2 :
            self.conn = np.c_[np.arange(nn), np.arange(nn) + nn]
                    
        elif self.dim == 3 :
            self.conn = np.c_[np.arange(nn), np.arange(nn) + nn, np.arange(nn) + 2 * nn]

    
    def CrtlPts2N(self, ctrlPts=None):
        if ctrlPts is None:
            ctrlPts = self.ctrlPts.copy()
            
        if self.dim == 2:
            n = np.c_[ctrlPts[0].ravel(order='F'),
                  ctrlPts[1].ravel(order='F')]
        elif self.dim == 3:
            n = np.c_[ctrlPts[0].ravel(order='F'),
                  ctrlPts[1].ravel(order='F'),
                  ctrlPts[2].ravel(order='F')]
        return n
    
    def N2CrtlPts(self, n=None):
        # n should be in the right order (xi, eta) meshgrid
        if n is None:
            n = self.n.copy()
        if self.dim == 2 :
            nbf = self.Get_nbf_1d()
            ctrlPts = np.array([n[:, 0].reshape(nbf, order='F'),
                                n[:, 1].reshape(nbf, order='F')])
        elif self.dim == 3:
            nbf = self.Get_nbf_1d()
            ctrlPts = np.array([n[:, 0].reshape(nbf, order='F'),
                                n[:, 1].reshape(nbf, order='F'),
                                n[:, 2].reshape(nbf, order='F')])
            
        return ctrlPts
    
    def BS2FE(self, U, n=[30, 30]):
        xi = np.linspace(self.knotVect[0][self.degree[0]],
                         self.knotVect[0][-self.degree[0]], n[0])
        eta = np.linspace(self.knotVect[1][self.degree[1]],
                          self.knotVect[1][-self.degree[1]], n[1])
        phi, b, c = self.ShapeFunctionsAtGridPoints(xi, eta)
        """
        x = phi.dot(self.n[:, 0])
        y = phi.dot(self.n[:, 1])
        roi = np.c_[np.ones(2), n].T-1
        mfe = StructuredMeshQ4(roi, 1)
        mfe.n = np.c_[x, y]
        # mfe.Plot()
        mfe.Connectivity()
        V = np.zeros(mfe.ndof)
        V[mfe.conn[:, 0]] = phi.dot(U[self.conn[:, 0]])
        V[mfe.conn[:, 1]] = phi.dot(U[self.conn[:, 1]])
        # mfe.Plot(V*30)
        return mfe, V, phi
        """

    def ShapeFunctionsAtGridPoints(self, xi, eta):
        """ xi and eta are the 1d points 
        This method computes the basis functions on the mesh-grid point 
        obtained from the 1d vector points xi and eta 
        """
        basis_xi = bs.BSplineBasis(self.degree[0], self.knotVect[0])
        basis_eta = bs.BSplineBasis(self.degree[1], self.knotVect[1])
        
        N_xi, DN_xi = global_basisfuns(self.degree[0], self.knotVect[0], xi)
        N_eta, DN_eta = global_basisfuns(self.degree[1], self.knotVect[1], eta)
        phi_xi = basis_xi.N(xi)
        phi_eta = basis_eta.N(eta)
        
        print(f"Correct : {N.shape} | Bsplyne = {phi_xi.shape}")
        
        
        
        dphi_xi = basis_xi.N(xi, k=1)
        dphi_eta = basis_eta.N(eta, k=1)
        
        # print(DN == dphi_xi)
        
        phi = sps.kron(phi_eta,  phi_xi,  'csc')
               
        
        dphidxi = sps.kron(phi_eta,  dphi_xi,  'csc')
        dphideta = sps.kron(dphi_eta,  phi_xi,  'csc')
        
           
        P = self.Get_P()
        
        dxdxi = dphidxi.dot(P[:, 0])
        dxdeta = dphideta.dot(P[:, 0])
        dydxi = dphidxi.dot(P[:, 1])
        dydeta = dphideta.dot(P[:, 1])
        detJ = dxdxi*dydeta - dydxi*dxdeta
        
        
        return phi, phi ,phi
    
        """
        
        
        dphidx = sps.diags(dydeta/detJ).dot(dphidxi) + \
            sps.diags(-dydxi/detJ).dot(dphideta)
        dphidy = sps.diags(-dxdeta/detJ).dot(dphidxi) + \
            sps.diags(dxdxi/detJ).dot(dphideta)
        # Univariate basis functions if needed
        # Nxi  = phi_xi
        # Neta = phi_eta
        N = phi
        return N, dphidx, dphidy
        """
            
        
        
# %% MAIN

if __name__ == "__main__":
    
    dim = 3
    N = 10
    
    ctrlP_2D = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])
    
    ctrlP_3D = np.array([[0, 0, 0],
                     [1, 0, 0,],
                     [0, 1, 0],
                     [0, 0, 1],
                     [0, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1],
                     [1, 1, 1]])
    
    ctrlP_3D = ctrlP_3D.T
    ctrlP_2D = ctrlP_2D.T
    
    xmin = 0 ; xmax = 1
    ymin = 0 ; ymax = 1
    zmin = 0 ; zmax = 1
    
    p = 2 ; q = 2 ; r = 2
    
    degree = [p, q, r]
    
    Xi = np.concatenate(
        (np.repeat(0, p+1), np.repeat(1, p+1)))*(xmax-xmin) + xmin
    Eta = np.concatenate(
        (np.repeat(0, q+1), np.repeat(1, q+1)))*(ymax-ymin) + ymin
    
    Zeta = np.concatenate(
        (np.repeat(0, r+1), np.repeat(1, r+1)))*(zmax-zmin) + zmin
    
    knotVect_2D = [Xi, Eta]
    knotVect_3D = [Xi, Eta, Zeta]
    
    m2D_ref = BSplinePatch_ref(ctrlPts=ctrlP_2D, degree=degree, knotVect=knotVect_2D)
   
    m2D = BSplinePatch(ctrlPts=ctrlP_2D, degree=degree, knotVect=knotVect_2D)
    m3D = BSplinePatch(ctrlPts=ctrlP_3D, degree=degree, knotVect=knotVect_3D)
   
    xi = np.linspace(knotVect_3D[0][degree[0]], knotVect_3D[0][-degree[0]], 30)
    eta = np.linspace(knotVect_3D[1][degree[1]], knotVect_3D[1][-degree[1]], 30)
    zeta = np.linspace(knotVect_3D[2][degree[2]], knotVect_3D[2][-degree[2]], 30)
    
    m2D_ref.ShapeFunctionsAtGridPoints(xi, eta)
    m2D.ShapeFunctionsAtGridPoints(xi, eta)
    #m2D.ShapeFunctionsAtGridPoints(xi, eta)
    

    
