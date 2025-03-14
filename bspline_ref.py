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
import pyxel as px

    

class BSplinePatch(object):
    def __init__(self, ctrlPts, degree, knotVect):
        """
        Nurbs surface from R^2 (xi,eta)--->R^2 (x,y) 
        ctrlPts = [X,Y] or ctrlPts = [X,Y,Z]
        """
        self.dim = ctrlPts.shape[0]
        self.ctrlPts = ctrlPts
        self.n = self.CtrlPts2N()
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
        return np.prod(self.Get_nbf_1d())

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
  
    def SelectNodes(self, n=-1):
        """
        Selection of nodes by hand in a mesh.
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        if hasattr(figManager.window, 'showMaximized'):
            figManager.window.showMaximized()
        else:
            if hasattr(figManager.window, 'maximize'):
                figManager.resize(figManager.window.maximize())
        plt.title("Select " + str(n) + " points... and press enter")
        pts1 = np.array(plt.ginput(n, timeout=0))
        plt.close()
        dx = np.kron(np.ones(pts1.shape[0]), self.n[:, [0]]) - np.kron(
            np.ones((self.n.shape[0], 1)), pts1[:, 0]
        )
        dy = np.kron(np.ones(pts1.shape[0]), self.n[:, [1]]) - np.kron(
            np.ones((self.n.shape[0], 1)), pts1[:, 1]
        )
        nset = np.argmin(np.sqrt(dx ** 2 + dy ** 2), axis=0)
        self.Plot()
        plt.plot(self.n[nset, 0], self.n[nset, 1], "ro")
        return nset

    def SelectLine(self, eps=1e-8):
        """
        Selection of the nodes along a line defined by 2 nodes.
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        if hasattr(figManager.window, 'showMaximized'):
            figManager.window.showMaximized()
        else:
            if hasattr(figManager.window, 'maximize'):
                figManager.resize(figManager.window.maximize())
        plt.title("Select 2 points of a line... and press enter")
        pts1 = np.array(plt.ginput(2, timeout=0))
        plt.close()
        n1 = np.argmin(np.linalg.norm(self.n - pts1[0, :], axis=1))
        n2 = np.argmin(np.linalg.norm(self.n - pts1[1, :], axis=1))
        v = np.diff(self.n[[n1, n2]], axis=0)[0]
        nv = np.linalg.norm(v)
        v = v / nv
        n = np.array([v[1], -v[0]])
        c = n.dot(self.n[n1, :])
        (rep,) = np.where(abs(self.n.dot(n) - c) < eps)
        c1 = v.dot(self.n[n1, :])
        c2 = v.dot(self.n[n2, :])
        nrep = self.n[rep, :]
        (rep2,) = np.where(((nrep.dot(v) - c1)
                            * (nrep.dot(v) - c2)) < nv * 1e-2)
        nset = rep[rep2]
        self.Plot()
        plt.plot(self.n[nset, 0], self.n[nset, 1], "ro")
        return nset
  
    def Connectivity(self):
        nn = len(self.n)
        self.ndof = nn * self.dim
        if self.dim == 2 :
            self.conn = np.c_[np.arange(nn), np.arange(nn) + nn]
                    
        elif self.dim == 3 :
            self.conn = np.c_[np.arange(nn), np.arange(nn) + nn, np.arange(nn) + 2 * nn]

    """
    def Connectivity(self, order = 'C'):
        nn = len(self.n)
        self.ndof = nn * self.dim
        self.conn = np.arange(self.ndof * nn)
        if order == 'C':
            self.conn.reshape(self.ndim, nn)
        
        elif order == 'N':
            self.conn.reshape(nn, ndim)
    """  

    def CtrlPts2N(self, ctrlPts=None):
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
    
    def N2CtrlPts(self, n=None):
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
        phi, _, _ = self.ShapeFunctionsAtGridPoints(xi, eta)
        
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
        
    def VTKSol(self, filename, U=None, n=[30, 30]):
        # Surface
        if U is None:
            U = np.zeros(self.ndof)
        m, V, phi = self.BS2FE(U, n)
        m.VTKSol(filename, V)
        # Control mesh
        nbf = self.Get_nbf_1d()
        roi = np.c_[np.ones(2, dtype=int), nbf].T-1
        mfe = StructuredMeshQ4(roi, 1)
        mfe.n = np.c_[self.ctrlPts[0].ravel(),
                      self.ctrlPts[1].ravel()]
        mfe.Connectivity()
        V = U.copy()
        V[self.conn[:, 0]] = U[self.conn[:, 0]].reshape(nbf, order='F').ravel()
        V[self.conn[:, 1]] = U[self.conn[:, 1]].reshape(nbf, order='F').ravel()
        mfe.VTKSol(filename+'_cp', V)

    def Plot(self, U=None, n=None, neval=[30, 30], **kwargs):
        """ Physical elements = Image of the parametric elements on Python """
        alpha = kwargs.pop("alpha", 1)
        edgecolor = kwargs.pop("edgecolor", "k")
        nbf = self.Get_nbf()
        if n is None:
            n = self.Get_P()  # control points
        if U is None:
            U = np.zeros(2*nbf)
        Pxm = n[:, 0] + U[:nbf]
        Pym = n[:, 1] + U[nbf:]

        xi = np.linspace(
            self.knotVect[0][self.degree[0]], self.knotVect[0][-self.degree[0]], neval[0])
        eta = np.linspace(
            self.knotVect[1][self.degree[1]], self.knotVect[1][-self.degree[1]], neval[1])
        # Iso parameters for the elemnts
        xiu = np.unique(self.knotVect[0])
        etau = np.unique(self.knotVect[1])

        # Basis functions
        basis_xi = bs.BSplineBasis(self.degree[0], self.knotVect[0])
        basis_eta = bs.BSplineBasis(self.degree[1], self.knotVect[1])
                
        
        phi_xi1 = basis_xi.N(xiu)
        phi_eta1 = basis_eta.N(eta)
        phi_xi2 = basis_xi.N(xi)
        phi_eta2 = basis_eta.N(etau)

        phi1 = sps.kron(phi_eta1,  phi_xi1,  'csc')
        phi2 = sps.kron(phi_eta2,  phi_xi2,  'csc')
        

        xe1 = phi1.dot(Pxm)
        ye1 = phi1.dot(Pym)
        xe2 = phi2.dot(Pxm)
        ye2 = phi2.dot(Pym)

        xe1 = xe1.reshape((xiu.size, neval[1]), order='F')
        ye1 = ye1.reshape((xiu.size, neval[1]), order='F')
        xe2 = xe2.reshape((neval[0], etau.size), order='F')
        ye2 = ye2.reshape((neval[0], etau.size), order='F')

        for i in range(xiu.size):
            # loop on xi
            # Getting one eta iso-curve
            plt.plot(xe1[i, :], ye1[i, :], color=edgecolor,
                     alpha=alpha, **kwargs)

        for i in range(etau.size):
            # loop on eta
            # Getting one xi iso-curve
            plt.plot(xe2[:, i], ye2[:, i], color=edgecolor,
                     alpha=alpha, **kwargs)
        plt.plot(Pxm, Pym, color=edgecolor,
                 alpha=alpha, marker='o', linestyle='')
        plt.axis('equal')

    def DegreeElevation(self, new_degree):
        
        spline = bs.BSpline(self.degree, self.knotVect)
        
        t = new_degree - self.degree
        self.ctrlPts = spline.orderElevation(self.ctrlPts, t)
        self.degree = new_degree
        self.knotVect = spline.getKnots()
        self.n = self.CtrlPts2N()

    def KnotInsertion(self, knots):
        
        spline = bs.BSpline(self.degree, self.knotVect)
        
        self.ctrlPts = spline.knotInsertion(self.ctrlPts, knots)
        self.knotVect = spline.getKnots()
        self.n = self.CtrlPts2N()
     
    def Stiffness(self, hooke):
        """ 
        Stiffness Matrix 
        """
        wg = sps.diags(self.wdetJ)
        Bxy = self.dphixdy+self.dphiydx
        K = hooke[0, 0]*self.dphixdx.T.dot(wg.dot(self.dphixdx)) +   \
            hooke[1, 1]*self.dphiydy.T.dot(wg.dot(self.dphiydy)) +   \
            hooke[2, 2]*Bxy.T.dot(wg.dot(Bxy)) + \
            hooke[0, 1]*self.dphixdx.T.dot(wg.dot(self.dphiydy)) +   \
            hooke[0, 2]*self.dphixdx.T.dot(wg.dot(Bxy)) +  \
            hooke[1, 2]*self.dphiydy.T.dot(wg.dot(Bxy)) +  \
            hooke[1, 0]*self.dphiydy.T.dot(wg.dot(self.dphixdx)) +   \
            hooke[2, 0]*Bxy.T.dot(wg.dot(self.dphixdx)) +  \
            hooke[2, 1]*Bxy.T.dot(wg.dot(self.dphiydy))
        return K   

    
    def Laplacian(self):
        wg = sps.diags(self.wdetJ)
        return self.dphixdx.T.dot(wg.dot(self.dphixdx)) + self.dphixdy.T.dot(wg.dot(self.dphixdy)) +\
            self.dphiydx.T.dot(wg.dot(self.dphiydx)) + \
            self.dphiydy.T.dot(wg.dot(self.dphiydy))

    def DoubleLaplacian(self):
        wg = sps.diags(self.wdetJ)
        return 2*self.dphixdxx.T.dot(wg.dot(self.dphixdyy)) +\
            2*self.dphiydxx.T.dot(wg.dot(self.dphiydyy)) +\
            self.dphixdxx.T.dot(wg.dot(self.dphixdxx)) +\
            self.dphixdyy.T.dot(wg.dot(self.dphixdyy)) +\
            self.dphiydxx.T.dot(wg.dot(self.dphiydxx)) +\
            self.dphiydyy.T.dot(wg.dot(self.dphiydyy))


    def GaussIntegration(self, npg=None, P=None):
        """ Gauss integration: build of the global differential operators """
        if npg is None:
            nbg_xi = self.degree[0]+1
            nbg_eta = self.degree[1]+1
        else:
            nbg_xi = npg[0]
            nbg_eta = npg[1]

        Gauss_xi = GaussLegendre(nbg_xi)
        Gauss_eta = GaussLegendre(nbg_eta)
        nbf = self.Get_nbf()

        e_xi = np.unique(self.knotVect[0])
        ne_xi = e_xi.shape[0]-1
        e_eta = np.unique(self.knotVect[1])
        ne_eta = e_eta.shape[0]-1
        xi_min = np.kron(e_xi[:-1], np.ones(nbg_xi))
        eta_min = np.kron(e_eta[:-1], np.ones(nbg_eta))
        xi_g = np.kron(np.ones(ne_xi), Gauss_xi[0])
        eta_g = np.kron(np.ones(ne_eta), Gauss_eta[0])

        basis_xi = bs.BSplineBasis(self.degree[0], self.knotVect[0])
        basis_eta = bs.BSplineBasis(self.degree[1], self.knotVect[1])

        """ Measures of elements """
        mes_xi = e_xi[1:] - e_xi[:-1]
        mes_eta = e_eta[1:] - e_eta[:-1]

        mes_xi = np.kron(mes_xi, np.ones(nbg_xi))
        mes_eta = np.kron(mes_eta, np.ones(nbg_eta))

        """ Going from the reference element to the parametric space  """
        xi = xi_min + 0.5*(xi_g+1) * \
            mes_xi     # Aranged gauss points in  xi direction
        # Aranged gauss points in  eta direction
        eta = eta_min + 0.5*(eta_g+1)*mes_eta

        phi_xi, dphi_xi = basis_xi.N(xi), basis_xi.N(xi, k=1)
        phi_eta, dphi_eta = basis_eta.N(eta), basis_eta.N(eta, k=1)

        phi = sps.kron(phi_eta,  phi_xi,  'csc')
        dphidxi = sps.kron(phi_eta,  dphi_xi,  'csc')
        dphideta = sps.kron(dphi_eta,  phi_xi,  'csc')
        self.npg = phi.shape[0]

        wg_xi = np.kron(np.ones(ne_xi), Gauss_xi[1])
        wg_eta = np.kron(np.ones(ne_eta), Gauss_eta[1])

        mes_xi = np.kron(np.ones(eta.shape[0]), mes_xi)
        mes_eta = np.kron(mes_eta, np.ones(xi.shape[0]))

        if P is None:
            P = self.Get_P()

        """ Jacobian elements"""
        dxdxi = dphidxi.dot(P[:, 0])
        dxdeta = dphideta.dot(P[:, 0])
        dydxi = dphidxi.dot(P[:, 1])
        dydeta = dphideta.dot(P[:, 1])
        detJ = dxdxi*dydeta - dydxi*dxdeta
        """ Spatial derivatives """
        dphidx = sps.diags(dydeta/detJ).dot(dphidxi) + \
            sps.diags(-dydxi/detJ).dot(dphideta)
        dphidy = sps.diags(-dxdeta/detJ).dot(dphidxi) + \
            sps.diags(dxdxi/detJ).dot(dphideta)
        """ Integration weights + measures + Jacobian of the transformation """
        self.wdetJ = np.kron(wg_eta, wg_xi)*np.abs(detJ)*mes_xi*mes_eta/4
        zero = sps.csr_matrix((self.npg, nbf))
        self.phi = phi
        self.phix = sps.hstack((phi, zero),  'csc')
        self.phiy = sps.hstack((zero, phi),  'csc')
        self.dphixdx = sps.hstack((dphidx, zero),  'csc')
        self.dphixdy = sps.hstack((dphidy, zero),  'csc')
        self.dphiydx = sps.hstack((zero, dphidx),  'csc')
        self.dphiydy = sps.hstack((zero, dphidy),  'csc')
        
        
    def GetApproxElementSize(self, cam=None):
        if cam is None:
            # in physical unit
            u, v = self.n[:, 0], self.n[:, 1]
            m2 = self.Copy()
            m2.GaussIntegration(npg=[1, 1], P=np.c_[u, v])
            n = np.max(np.sqrt(m2.wdetJ))
        else:
            # in pyxel unit (int)
            u, v = cam.P(self.n[:, 0], self.n[:, 1])
            m2 = self.Copy()
            m2.GaussIntegration(npg=[1, 1], P=np.c_[u, v])
            n = int(np.floor(np.max(np.sqrt(m2.wdetJ))))
        return n
    
    def DICIntegrationFast(self, n=10):
        self.DICIntegration(n)

        
    def DICIntegration(self, n=10):
        """ DIC integration: build of the global differential operators """
        if hasattr(n, 'rz'):
            # if n is a camera then n is autocomputed
            n = self.GetApproxElementSize(n)
        if type(n) == int:
            n = np.array([n, n], dtype=int)
        n = np.maximum(self.degree + 1, n)
        nbg_xi = n[0]
        nbg_eta = n[1]

        Rect_xi = np.linspace(-1, 1, nbg_xi)
        Weight_xi = 2/n[0] * np.ones(nbg_xi)
        Rect_eta = np.linspace(-1, 1, nbg_eta)
        Weight_eta = 2/n[1] * np.ones(nbg_eta)

        nbf = self.Get_nbf()

        e_xi = np.unique(self.knotVect[0])
        ne_xi = e_xi.shape[0]-1
        e_eta = np.unique(self.knotVect[1])
        ne_eta = e_eta.shape[0]-1
        xi_min = np.kron(e_xi[:-1], np.ones(nbg_xi))
        eta_min = np.kron(e_eta[:-1], np.ones(nbg_eta))
        xi_g = np.kron(np.ones(ne_xi), Rect_xi)
        eta_g = np.kron(np.ones(ne_eta), Rect_eta)

        """ Measures of elements """
        mes_xi = e_xi[1:] - e_xi[:-1]
        mes_eta = e_eta[1:] - e_eta[:-1]

        mes_xi = np.kron(mes_xi, np.ones(nbg_xi))
        mes_eta = np.kron(mes_eta, np.ones(nbg_eta))

        """ Going from the reference element to the parametric space  """
        xi = xi_min + 0.5*(xi_g+1) * \
            mes_xi     # Aranged gauss points in  xi direction
        # Aranged gauss points in  eta direction
        eta = eta_min + 0.5*(eta_g+1)*mes_eta

        basis_xi = bs.BSplineBasis(self.degree[0], self.knotVect[0])
        basis_eta = bs.BSplineBasis(self.degree[1], self.knotVect[1])

        phi_xi, dphi_xi = basis_xi.N(xi), basis_xi.N(xi, k=1)
        phi_eta, dphi_eta = basis_eta.N(eta), basis_eta.N(eta, k=1)

        phi = sps.kron(phi_eta,  phi_xi,  'csc')
        dphidxi = sps.kron(phi_eta,  dphi_xi,  'csc')
        dphideta = sps.kron(dphi_eta,  phi_xi,  'csc')
        self.npg = phi.shape[0]

        wg_xi = np.kron(np.ones(ne_xi), Weight_xi)
        wg_eta = np.kron(np.ones(ne_eta), Weight_eta)

        mes_xi = np.kron(np.ones(eta.shape[0]), mes_xi)
        mes_eta = np.kron(mes_eta, np.ones(xi.shape[0]))

        P = self.Get_P()

        """ Jacobian elements"""
        dxdxi = dphidxi.dot(P[:, 0])
        dxdeta = dphideta.dot(P[:, 0])
        dydxi = dphidxi.dot(P[:, 1])
        dydeta = dphideta.dot(P[:, 1])
        detJ = dxdxi*dydeta - dydxi*dxdeta
        """ Spatial derivatives """
        dphidx = sps.diags(dydeta/detJ).dot(dphidxi) + \
            sps.diags(-dydxi/detJ).dot(dphideta)
        dphidy = sps.diags(-dxdeta/detJ).dot(dphidxi) + \
            sps.diags(dxdxi/detJ).dot(dphideta)
        """ Integration weights + measures + Jacobian of the transformation """
        self.wdetJ = np.kron(wg_eta, wg_xi)*np.abs(detJ)*mes_xi*mes_eta/4

        zero = sps.csr_matrix((self.npg, nbf))
        self.phi = phi
        self.phix = sps.hstack((phi, zero),  'csc')
        self.phiy = sps.hstack((zero, phi),  'csc')
        self.dphixdx = sps.hstack((dphidx, zero),  'csc')
        self.dphixdy = sps.hstack((dphidy, zero),  'csc')
        self.dphiydx = sps.hstack((zero, dphidx),  'csc')
        self.dphiydy = sps.hstack((zero, dphidy),  'csc')

        self.pgx = self.phi @ P[:, 0]
        self.pgy = self.phi @ P[:, 1]
 
    def ShapeFunctionsAtGridPoints(self, xi, eta, zeta=None):
        """ xi, eta (and zeta in 3D) are the 1d points 
        This method computes the basis functions on the mesh-grid point 
        obtained from the 1d vector points xi, eta (and zeta)
        """
        param = np.array([xi, eta])
            
        basis_xi = bs.BSplineBasis(self.degree[0], self.knotVect[0])
        basis_eta = bs.BSplineBasis(self.degree[1], self.knotVect[1])
        
        phi_xi, dphi_xi = basis_xi.N(xi), basis_xi.N(xi, k=1)
        phi_eta, dphi_eta = basis_eta.N(eta), basis_eta.N(eta, k=1)
                       
        if self.dim == 2:    
            
            phi = sps.kron(phi_eta,  phi_xi, 'csc')            
            dphidxi = sps.kron(phi_eta,  dphi_xi, 'csc')
            dphideta = sps.kron(dphi_eta,  phi_xi, 'csc')
            
            
            P = self.Get_P()
            
            dxdxi = dphidxi.dot(P[:, 0])
            dxdeta = dphideta.dot(P[:, 0])
            dydxi = dphidxi.dot(P[:, 1])
            dydeta = dphideta.dot(P[:, 1])
            detJ = dxdxi*dydeta - dydxi*dxdeta
            dphidx = sps.diags(dydeta/detJ).dot(dphidxi) + \
                sps.diags(-dydxi/detJ).dot(dphideta)
            dphidy = sps.diags(-dxdeta/detJ).dot(dphidxi) + \
                sps.diags(dxdxi/detJ).dot(dphideta)
            # Univariate basis functions if needed
            # Nxi  = phi_xi
            # Neta = phi_eta
            N = phi
            return N, dphidx, dphidy
        
        elif self.dim == 3 :
            
            basis_zeta = bs.BSplineBasis(self.degree[2], self.knotVect[2])
            phi_zeta = basis_zeta.N(zeta)
            dphi_zeta = basis_zeta.N(zeta, k=1)
     
        


    def PlaneWave(self, T):
        V = np.zeros(self.ndof)
        V[self.conn[:, 0]] = np.cos(self.n[:, 1] / T * 2 * np.pi)
        return V       


def Rectangle(roi, n_elems, degree):
    xmin, ymin, xmax, ymax = roi.ravel()
    # Parametric space properties
    p = 1
    q = 1
    Xi = np.concatenate(
        (np.repeat(0, p+1), np.repeat(1, p+1)))*(xmax-xmin) + xmin
    Eta = np.concatenate(
        (np.repeat(0, q+1), np.repeat(1, q+1)))*(ymax-ymin) + ymin
    # Control points for a recangular plate
    x = np.array([[xmin, xmin],
                [xmax, xmax]])
    y = np.array([[ymin, ymax],
                [ymin, ymax]])
    ctrlPts = np.array([x, y])
    knot_vector = [Xi, Eta]
    m = BSplinePatch(ctrlPts, np.array([p, q]), knot_vector)
    # Degree elevation
    m.DegreeElevation(degree)
    # Knot refinement
    ubar = [None]*2
    ubar[0] = 1/n_elems[0] * np.arange(1, n_elems[0]) * (xmax-xmin) + xmin
    ubar[1] = 1/n_elems[1] * np.arange(1, n_elems[1]) * (ymax-ymin) + ymin
    m.KnotInsertion(ubar)
    # Building connectivity
    m.Connectivity()
    return m


def SplineFromROI(roi, dx, degree=[2, 2]):
    """Build a structured FE mesh and a pyxel.camera object from a region
    of interest (ROI) selected in an image f

    Parameters
    ----------
    roi : numpy.array
        The Region of Interest made using  f.SelectROI(), f being a pyxel.Image
    dx : numpy or python array
        dx  = [dx, dy]: average element size (can be scalar) in pixels
    typel : int
        type of element: {3: 'qua4',2: 'tri3',9: 'tri6',16: 'qua8',10: 'qua9'}

    Returns
    -------
    m : pyxel.Mesh
        The finite element mesh
    cam : pyxel.Camera
        The corresponding camera

    Example:
    -------
    f.SelectROI()  -> select the region with rectangle selector
                    and copy - paste the roi in the python terminal
    m, cam = px.MeshFromROI(roi, [20, 20], 3)
    """
    dbox = roi[1] - roi[0]
    NE = (dbox / dx).astype(int)
    NE = np.max(np.c_[NE, np.ones(2, dtype=int)], axis=1)
    m = Rectangle(roi, NE, degree)
    m.n[:, 1] *= -1
    print(m.ctrlPts.shape)
    m.ctrlPts = m.N2CrtlPts()
    p = np.array([1., 0., 0., 0.])
    cam = Camera(p)
    return m, cam





if __name__ == "__main__":
    
    f = px.Image('zoom-0053_1.tif').Load()
    # f.Plot()
    g = px.Image('zoom-0070_1.tif').Load()
    
    # %% Test ex 9    
    a = 0.925
    Xi = np.array([[0.5, 0.75, 1],
                [0.5*a, 0.75*a, 1*a],
                [0, 0, 0]])
    Yi = np.array([[0, 0, 0],
                [0.5*a, 0.75*a, 1*a],
                [0.5, 0.75, 1]])

    ctrlPts = np.array([Xi, Yi])
    degree = [2, 2]
    kv = np.array([0, 0, 0, 1, 1, 1])
    knotVect = [kv, kv]

    n = 5
    newr = np.linspace(0, 1, n+2)[1:-1]
    n = 10
    newt = np.linspace(0, 1, n+2)[1:-1]
    m = BSplinePatch(ctrlPts, degree, knotVect)
    # m.Plot()
    m.GaussIntegration()

    
    n = 500


    m.KnotInsertion([newt, newr])
    # m.DegreeElevation(np.array([3, 3]))
    m.Plot()

    cam = px.Camera([100, 6.95, -5.35, 0])
    #px.PlotMeshImage(f, m, cam)
    u, v = cam.P(m.pgx, m.pgy)
    # plt.plot(u, v, "y.")
    # plt.show()
    m.Connectivity()
    m.DICIntegration(cam)
    # %% Next    
    
    U = px.MultiscaleInit(f, g, m, cam, scales=[2, 1])
    U, res = px.Correlate(f, g, m, cam, U0=U)

    m.Plot(U, alpha=0.5)
    m.Plot(U=3*U)

    px.PlotMeshImage(g, m, cam, U)

    # %% Test 2
    
    roi = np.array([[536, 54], [849, 481]])
    m, cam = SplineFromROI(roi, dx=30, degree=[2, 2])
    px.PlotMeshImage(f, m, cam)

    m.Connectivity()
    m.Plot()

    m.DICIntegration(cam)

    px.PlotMeshImage(f, m, cam)

    U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
    U, res = px.Correlate(f, g, m, cam, U0=U)

    m.Plot(alpha=0.5)
    m.Plot(U=3*U)
