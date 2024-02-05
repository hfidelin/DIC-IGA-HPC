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
import scipy.sparse as sps
import matplotlib.pyplot as plt
from pyxel.camera import Camera
from pyxel.mesher import StructuredMeshQ4
import bsplyne as bs
import pyxel as px
import pickle
from shapely.geometry import Polygon, Point


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
        self.spline = bs.BSpline(self.degree, self.knotVect)

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
        
        
        
        self.wdetJ = np.empty(0)    # Integration weights diagonal matrix
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
        m.pgz = self.pgz.copy()
        m.phix = self.phix.copy()
        m.phiy = self.phiy.copy()
        m.phiz = self.phiy.copy()
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
            P = np.c_[self.ctrlPts[0].ravel(),
                  self.ctrlPts[1].ravel()]
        
        elif self.dim == 3:
            P = np.c_[self.ctrlPts[0].ravel(),
                  self.ctrlPts[1].ravel(),
                  self.ctrlPts[2].ravel()]
        return P
  
    
    def Get_spline(self):
        return bs.BSpline(self.degree, self.knotVect)
    
    
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
    def Connectivity2(self, order = 'C'):
        nn = len(self.n)
        self.ndof = nn * self.dim
        self.conn = np.arange(self.ndof * nn)
        if order == 'C':
            self.conn.reshape(self.dim, nn)
        
        elif order == 'N':
            self.conn.reshape(nn, self.dim)
    """

    def CtrlPts2N(self, ctrlPts=None):
        if ctrlPts is None:
            ctrlPts = self.ctrlPts.copy()
            
        if self.dim == 2:
            n = np.c_[ctrlPts[0].ravel(),
                  ctrlPts[1].ravel()]
        
        elif self.dim == 3:
            n = np.c_[ctrlPts[0].ravel(),
                  ctrlPts[1].ravel(),
                  ctrlPts[2].ravel()]
        return n
    
    def N2CtrlPts(self, n=None):
        # n should be in the right order (xi, eta) meshgrid
        if n is None:
            n = self.n.copy()
        if self.dim == 2 :
            nbf = self.Get_nbf_1d()
            ctrlPts = np.array([n[:, 0].reshape(nbf),
                                n[:, 1].reshape(nbf)])
        elif self.dim == 3:
            nbf = self.Get_nbf_1d()
            ctrlPts = np.array([n[:, 0].reshape(nbf),
                                n[:, 1].reshape(nbf),
                                n[:, 2].reshape(nbf)])
            
        return ctrlPts
    
    def BS2FE(self, U, n=[30, 30]):
        xi = np.linspace(self.knotVect[0][self.degree[0]],
                         self.knotVect[0][-self.degree[0]], n[0])
        eta = np.linspace(self.knotVect[1][self.degree[1]],
                          self.knotVect[1][-self.degree[1]], n[1])
        phi, _, _ = self.ShapeFunctions(xi, eta)
        
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
        V[self.conn[:, 0]] = U[self.conn[:, 0]].reshape(nbf).ravel()
        V[self.conn[:, 1]] = U[self.conn[:, 1]].reshape(nbf).ravel()
        mfe.VTKSol(filename+'_cp', V)

    def Plot(self, U=None, n=None, neval=None, **kwargs):
        """ Physical elements = Image of the parametric elements on Python """
        if neval == None:
            if self.dim == 2:
                neval = [30, 30]
            elif self.dim == 3:
                neval = [30, 30, 30]
        alpha = kwargs.pop("alpha", 1)
        edgecolor = kwargs.pop("edgecolor", "k")
        nbf = self.Get_nbf()
        if n is None:
            n = self.Get_P()  # control points
        if U is None:
            U = np.zeros(self.dim*nbf)
        U = U.reshape((self.dim, -1))    
            
        if self.dim == 3:
            try:
                import k3d
            except:
                raise Exception("k3d is not installed")
            spline = bs.BSpline(self.degree, self.knotVect)
            ctrl_pts = self.ctrlPts
            save = "/tmp/stl_mesh.stl"
            mesh = make_stl_mesh(spline, ctrl_pts)
            mesh.save(save)
            with open(save, 'rb') as stl:
                data = stl.read()
            
            x = self.ctrlPts[0].ravel()
            y = self.ctrlPts[1].ravel()
            z = self.ctrlPts[2].ravel()
            
            plt_points = k3d.points(positions=np.array([x, y, z]).T,
                                    point_size = 0.2,
                                    shader='3d',
                                    color=0x3f6bc5)
            
           
            
            plot = k3d.plot()
            plot += k3d.stl(data)
            plot += plt_points
            plot.display()
            
            
        elif self.dim == 2:    
            Pxm = n[:, 0] + U[0]
            Pym = n[:, 1] + U[1]
            
            xi = np.linspace(
                self.knotVect[0][self.degree[0]], self.knotVect[0][-self.degree[0]], neval[0])
            eta = np.linspace(
                self.knotVect[1][self.degree[1]], self.knotVect[1][-self.degree[1]], neval[1])
            
            # Iso parameters for the elemnts
            xiu = np.unique(self.knotVect[0])
            etau = np.unique(self.knotVect[1])

            # Basis functions
            spline = bs.BSpline(self.degree, self.knotVect)
            
            
            phi1 = spline.DN([xiu, eta], k=[0, 0])
            phi2 = spline.DN([xi, etau], k=[0, 0])
            
            #xe, ye1 = spline(self.ctrlPts)
            xe1 = phi1.dot(Pxm)
            ye1 = phi1.dot(Pym)
            xe2 = phi2.dot(Pxm)
            ye2 = phi2.dot(Pym)
            
            
            xe1 = xe1.reshape((xiu.size, neval[1]))
            ye1 = ye1.reshape((xiu.size, neval[1]))
            xe2 = xe2.reshape((neval[0], etau.size))
            ye2 = ye2.reshape((neval[0], etau.size))

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
        self.degree = np.array(new_degree)
        self.knotVect = spline.getKnots()
        self.n = self.CtrlPts2N()
        self.spline = bs.BSpline(self.degree, self.knotVect)

    def KnotInsertion(self, knots):
        
        spline = bs.BSpline(self.degree, self.knotVect)
        # self.ctrlPts =  self.ctrlPts.transpose(0,2,1)
        self.ctrlPts = spline.knotInsertion(self.ctrlPts, knots)
        self.knotVect = spline.getKnots()
        self.n = self.CtrlPts2N()
        self.spline = bs.BSpline(self.degree, self.knotVect)
     
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


    def Init_Polygon(self, cam):
        """
        Compute a polygon object from a mesh of a bspline patch.
        
        Parameter:
        ----------
        
        m : bspline mesh class from pyxel
        """
        
        # Number of evaluation point for element contour
        neval = [30, 30]
        Pxm = self.n[:, 0]
        Pym = self.n[:, 1]
        
        # parametric point for evaluation of element contour
        
        xi = np.linspace(
            self.knotVect[0][self.degree[0]], self.knotVect[0][-self.degree[0]], neval[0])
        eta = np.linspace(
            self.knotVect[1][self.degree[1]], self.knotVect[1][-self.degree[1]], neval[1])
        
        # Iso parameters for the elements
        xiu = np.unique(self.knotVect[0])
        etau = np.unique(self.knotVect[1])
        
        # Basis functions
        spline = self.Get_spline()
        phi_xiu = spline.DN([xiu, eta], k=[0, 0])
        phi_etau = spline.DN([xi, etau], k=[0, 0])
        
        # Projection in physical space using bspline mapping
        x_xiu = phi_xiu @ Pxm
        y_xiu = phi_xiu @ Pym
        x_etau = phi_etau @ Pxm
        y_etau = phi_etau @ Pym
        
        # Reshaping into element contour coordinates 
        x_xiu = x_xiu.reshape((xiu.size, neval[1]))
        y_xiu = y_xiu.reshape((xiu.size, neval[1]))
        x_etau = x_etau.reshape((neval[0], etau.size))
        y_etau = y_etau.reshape((neval[0], etau.size))
        
        
        xi_first = []
        xi_last = []
        for i in range(x_xiu.shape[1]):
            xi_first.append([x_xiu[0, i], y_xiu[0, i]])
            xi_last.append([x_xiu[-1, i], y_xiu[-1, i]])
        
        
        eta_first = []
        eta_last = []
        for i in range(x_etau.shape[0]):
            eta_first.append([x_etau[i, 0], y_etau[i, 0]])
            eta_last.append([x_etau[i, -1], y_etau[i, -1]])
           
        coord = np.vstack((xi_first, eta_last, xi_last[::-1], eta_first[::-1]))
        
        u, v = cam.P(coord[:, 0], coord[:, 1])
        coord = np.vstack((v, u))
        patch = Polygon(coord.T)
        
        return patch


    def IsInPatch(self, patch):
        """
        Compute the coordinate of the pixels contained by a bspline patch
        
        Parameter:
        ----------
        
        patch : polygon class from shapely.geometry that describe a bspline patch
        
        Return:
        -------
        
        x, y : np.array containing the coordinates of the pixels within the 
                bspline patch
        """
        
        xmin, ymin, xmax, ymax = patch.bounds
    
        # Meshgrid of bounding box
        xx, yy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    
        X = xx.flatten()
        Y = yy.flatten()
        points = []
        for x, y in zip(X, Y):
            if patch.contains(Point(x, y)):
                points.append([x, y])
    
        points = np.array(points)
    
        return points

    def InverseBSplineapping(self, x, y):
        """ 
        Inverse the BSpline mapping in order to map the coordinates
        of any physical points (x, y) to their corresponding position in
        the parametric space (xg, yg).
        
        Parameter : 
            m : bspline mesh from pyxel
            xpix : x coordinate in physical space
            ypix : y coordinate in physical space
            
        Return :
            xg, yg : x coordinate 
        """
        
        # Basis function
        
        spline = self.Get_spline()
        
        # Coordinates of controls points
        xn = self.n[:, 0]
        yn = self.n[:, 1]
        
        # Initializing  parametric integration points to zero
        xi_g = 0.5 * np.ones_like(x) #0 * x
        eta_g = 0.5 * np.ones_like(y) #0 * y
        res = 1
        # Gauss Newton loop
        for k in range(7):
            
            phi = spline.DN(np.array([xi_g, eta_g]), k=[0,0])
            N_r = spline.DN(np.array([xi_g, eta_g]), k=[1,0])
            N_s = spline.DN(np.array([xi_g, eta_g]), k=[0,1])
            
            dxdr = N_r @ xn
            dydr = N_r @ yn
            dxds = N_s @ xn
            dyds = N_s @ yn
            
            detJ = dxdr * dyds - dydr * dxds
            invJ = np.array([dyds / detJ, -dxds / detJ,
                             -dydr / detJ, dxdr / detJ]).T
            
            xp = phi @ xn
            yp = phi @ yn
            
            dxi_g = invJ[:, 0] * (x - xp) + invJ[:, 1] * (y - yp)
            deta_g = invJ[:, 2] * (x - xp) + invJ[:, 3] * (y - yp)
            
            res = np.dot(dxi_g, dxi_g) + np.dot(deta_g, deta_g)
            
            xi_g = xi_g + dxi_g
            eta_g = eta_g + deta_g
            
            xi_g = np.clip(xi_g, 0, 1)
            eta_g = np.clip(eta_g, 0, 1)
            
            print(f"Itération {k} | résidu : {res}")
            if res < 1.0e-6:
                break
            
        return xi_g, eta_g

    def DICIntegrationPixel(self, m, cam, P=None):
        """
        Building integration operator where integration points are located
        in center of pixels
        
        Parameter:
        ---------
        
        m : pyxel bspline mesh
        
        cam : camera model from pyxel
        """
        
        # Approximation of the patch using a polygon
        patch = self.Init_Polygon(cam)
        
        # Computing pixels that are inside the bspline patch
        pixel = self.IsInPatch(patch)
        
        # Inversing camera model to compute physical coordinate of those pixel
        xg, yg = cam.PinvNL(pixel[:, 1], pixel[:, 0])
        
        # Inversing bspline mapping
        xi_g, eta_g = self.InverseBSplineapping(xg, yg)
        
        spline = self.Get_spline()
        phi = spline.DN(np.array([xi_g, eta_g]), k=[0,0])
        
        self.npg = phi.shape[0]  
        self.wdetJ = np.ones_like(xi_g)
        nbf = self.Get_nbf()
        zero = sps.csr_matrix((self.npg, nbf))
        self.phi = phi
        self.phix = sps.hstack((phi, zero),  'csc')
        self.phiy = sps.hstack((zero, phi),  'csc')
        
        if P is None:
            P = self.Get_P()
        

        self.pgx = phi @ P[:, 0]
        self.pgy = phi @ P[:, 1]
        
        
        
    def GaussIntegration(self, npg=None, P=None):
        """ Gauss integration: build of the global differential operators """
        if npg is None:
            nbg_xi = self.degree[0]+1
            nbg_eta = self.degree[1]+1
        else:
            nbg_xi = npg[0]
            nbg_eta = npg[1]

        GaussLegendre = np.polynomial.legendre.leggauss

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

        
        """ Measures of elements """
        mes_xi = e_xi[1:] - e_xi[:-1]
        mes_eta = e_eta[1:] - e_eta[:-1]

        mes_xi = np.kron(mes_xi, np.ones(nbg_xi))
        mes_eta = np.kron(mes_eta, np.ones(nbg_eta))

        """ Going from the reference element to the parametric space  """
        # Aranged gauss points in  xi direction
        xi = xi_min + 0.5 * (xi_g + 1) * mes_xi     
        # Aranged gauss points in  eta direction
        eta = eta_min + 0.5 * (eta_g + 1) * mes_eta


        wg_xi = np.kron(np.ones(ne_xi), Gauss_xi[1])
        wg_eta = np.kron(np.ones(ne_eta), Gauss_eta[1])

        mes_xi = np.kron(np.ones(eta.shape[0]), mes_xi)
        mes_eta = np.kron(mes_eta, np.ones(xi.shape[0]))

        if P is None:
            P = self.Get_P()

        """ Spatial derivatives """
        
        phi, dphidx, dphidy, detJ = self.ShapeFunctions(xi, eta, P=P)
        self.npg = phi.shape[0]
        
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
        
        # Nombre de point d'intégration par élément
        nbg_xi = n[0]
        nbg_eta = n[1]
        
        """Espace de référence [-1,1]"""
        pxi = 1.0 / nbg_xi
        peta = 1.0 / nbg_eta
        Rect_xi = np.linspace(-1+pxi, 1-pxi, nbg_xi)
        Rect_eta = np.linspace(-1+peta, 1-peta, nbg_eta)
        
        
        nbf = self.Get_nbf()
        
        """
        On met en place le mapping de [-1,1] à [xi_i, xi_i+1] un élément
        de l'espace paramétrique
        """
        
        # Array des éléments uniques du vecteur de neud
        e_xi = np.unique(self.knotVect[0])
        e_eta = np.unique(self.knotVect[1])
        
        # Nombre d'élément
        ne_xi = e_xi.shape[0]-1
        ne_eta = e_eta.shape[0]-1
        
        # Pour tout éléments, on calcule leur borne inférieur xi_i / eta_i
        xi_min = np.kron(e_xi[:-1], np.ones(nbg_xi))
        xi_max = np.kron(e_xi[1:], np.ones(nbg_xi))
        
        # Pour tout éléments on calcule leur borne supérieur xi_i+1 / eta_i+1
        eta_min = np.kron(e_eta[:-1], np.ones(nbg_eta))
        eta_max = np.kron(e_eta[1:], np.ones(nbg_eta))
        
        # On initialise le vecteur contenant les points d'intégration pour
        # chaque élément
        xi_g = np.kron(np.ones(ne_xi), Rect_xi)
        eta_g = np.kron(np.ones(ne_eta), Rect_eta)
        

        """ Passage de l'élément de référence à l'espace paramétrique """
        # Mapping des xi
        xi = 0.5 * (xi_min + xi_max) + 0.5 * (xi_max - xi_min) * xi_g
        
        # Mapping des eta
        eta = 0.5 * (eta_min + eta_max) + 0.5 * (eta_max - eta_min) * eta_g
                
        """
        Calcul des matrices contenant les fonctions de formes évaluée aux 
        points d'intégrations
        """
        phi, dphidx, dphidy, detJ = self.ShapeFunctions(xi, eta)
        self.npg = phi.shape[0]
        
        P = self.Get_P()
        self.wdetJ = np.ones_like(detJ)
        
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


    def DIntegration(self, n=100):
        
        if type(n) == int:
            n = np.array([n, n], dtype=int)
        
        n = np.maximum(self.degree + 1, n)
        
        # Nombre de point d'intégration par élément
        nbg_xi = n[0]
        nbg_eta = n[1]
        
        """Espace de référence [-1,1]"""
        pxi = 1.0 / nbg_xi
        peta = 1.0 / nbg_eta
        xi_g = np.linspace(pxi, 1-pxi, nbg_xi)
        eta_g = np.linspace(peta, 1-peta, nbg_eta)
        
        nbf = self.Get_nbf()
        
        """
        Calcul des matrices contenant les fonctions de formes évaluée aux 
        points d'intégrations
        """
        phi, _, _, detJ = self.ShapeFunctions(xi_g, eta_g)
        self.npg = phi.shape[0]
        
        P = self.Get_P()
        
        self.wdetJ = np.ones_like(detJ)
        
        zero = sps.csr_matrix((self.npg, nbf))
        self.phi = phi
        self.phix = sps.hstack((phi, zero),  'csc')
        self.phiy = sps.hstack((zero, phi),  'csc')
        

        self.pgx = self.phi @ P[:, 0]
        self.pgy = self.phi @ P[:, 1]





    def DVC_Integration(self, n=None, G=False):
        """ DVC integration: build of the global differential operators """
        if hasattr(n, 'rz'):
            # if n is a camera then n is autocomputed
            n = self.GetApproxElementSize(n)
        if type(n) == int:
            n = np.array([n, n, n], dtype=int)
        
        n = np.maximum(self.degree + 1, n)
        nbg_xi = n[0]
        nbg_eta = n[1]
        nbg_zeta = n[2]

        # Computing reference spaces and weights
        Rect_xi = np.linspace(-1, 1, nbg_xi)
        Rect_eta = np.linspace(-1, 1, nbg_eta)
        Rect_zeta = np.linspace(-1, 1, nbg_zeta)
        
        nbf = self.Get_nbf()
        

        e_xi = np.unique(self.knotVect[0])
        ne_xi = e_xi.shape[0]-1
        e_eta = np.unique(self.knotVect[1])
        ne_eta = e_eta.shape[0]-1
        e_zeta = np.unique(self.knotVect[2])
        ne_zeta = e_zeta.shape[0]-1
        
        
        xi_min = np.kron(e_xi[:-1], np.ones(nbg_xi))
        eta_min = np.kron(e_eta[:-1], np.ones(nbg_eta))
        zeta_min = np.kron(e_zeta[:-1], np.ones(nbg_zeta))
        
        xi_max = np.kron(e_xi[1:], np.ones(nbg_xi))
        eta_max = np.kron(e_eta[1:], np.ones(nbg_eta))
        zeta_max = np.kron(e_zeta[1:], np.ones(nbg_zeta))       
        
        xi_g = np.kron(np.ones(ne_xi), Rect_xi)
        eta_g = np.kron(np.ones(ne_eta), Rect_eta)
        zeta_g = np.kron(np.ones(ne_zeta), Rect_zeta)



        """ Going from the reference element to the parametric space  """
        # Aranged gauss points in  xi direction  
        xi = 0.5 * (xi_min + xi_max) + 0.5 * (xi_max - xi_min) * xi_g
        # Aranged gauss points in  eta direction
        eta = 0.5 * (eta_min + eta_max) + 0.5 * (eta_max - eta_min) * eta_g    
        # Aranged gauss points in  zeta direction
        zeta = 0.5 * (zeta_min + zeta_max) + 0.5 * (zeta_max - zeta_min) * zeta_g 
        
    
        phi, dphidx, dphidy, dphidz, detJ = self.ShapeFunctions(xi, eta, zeta)
        self.npg = phi.shape[0]

        P = self.Get_P()
        
        """ Integration weights + measures + Jacobian of the transformation """
        self.wdetJ = np.ones_like(detJ)

        zero = sps.csr_matrix((self.npg, nbf))
        self.phi = phi
        
        self.phix = sps.hstack((phi, zero, zero),  'csc')
        self.phiy = sps.hstack((zero, phi, zero),  'csc')
        self.phiz = sps.hstack((zero, zero, phi),  'csc')
        
        self.dphixdx = sps.hstack((dphidx, zero, zero),  'csc')
        self.dphixdy = sps.hstack((dphidy, zero, zero),  'csc')
        self.dphixdz = sps.hstack((dphidz, zero, zero),  'csc')
        
        self.dphiydx = sps.hstack((zero, dphidx, zero),  'csc')
        self.dphiydy = sps.hstack((zero, dphidy, zero),  'csc')
        self.dphiydz = sps.hstack((zero, dphidy, zero),  'csc')
        
        self.dphizdx = sps.hstack((zero, zero, dphidx),  'csc')
        self.dphizdy = sps.hstack((zero, zero, dphidy),  'csc')
        self.dphizdz = sps.hstack((zero, zero, dphidy),  'csc')

        self.pgx = self.phi @ P[:, 0]
        self.pgy = self.phi @ P[:, 1]
        self.pgz = self.phi @ P[:, 2]


    def ShapeFunctions(self, xi, eta, zeta=None, P=None):
        """ xi, eta (and zeta in 3D) are the 1d points 
        This method computes the basis functions on the mesh-grid point 
        obtained from the 1d vector points xi, eta (and zeta)
        """
            
        
        spline = bs.BSpline(self.degree, self.knotVect)
                       
        if self.dim == 2:    
            
            #print(spline.getSpans())
            phi = spline.DN([xi, eta], k=[0, 0])
                     
            dphidxi = spline.DN([xi, eta], k=[1, 0])
            dphideta = spline.DN([xi, eta], k=[0, 1])
            
            
            if P is None:
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
            
            return phi, dphidx, dphidy, detJ
            
        
        elif self.dim == 3 :
            
            phi = spline.DN([xi, eta, zeta], k=[0, 0, 0])
                   
            dphidxi = spline.DN([xi, eta, zeta], k=[1, 0, 0]).tocsc()
            dphideta = spline.DN([xi, eta, zeta], k=[0, 1, 0]).tocsc()
            dphidzeta = spline.DN([xi, eta, zeta], k=[0, 0, 1]).tocsc()
            
            if P is None:
                P = self.Get_P()
            
            dxdxi = dphidxi.dot(P[:, 0])
            dxdeta = dphideta.dot(P[:, 0])
            dxdzeta = dphidzeta.dot(P[:, 0])
            
            dydxi = dphidxi.dot(P[:, 1])
            dydeta = dphideta.dot(P[:, 1])
            dydzeta = dphidzeta.dot(P[:, 1])
            
            dzdxi = dphidxi.dot(P[:, 2])
            dzdeta = dphideta.dot(P[:, 2])
            dzdzeta = dphidzeta.dot(P[:, 2])            
            
            # Applying Sarrus Rules :
            aei = dxdxi*dydeta*dzdzeta
            dhc = dxdeta*dydzeta*dzdxi
            bfg = dxdzeta*dydxi*dzdeta
            gec = dxdzeta*dydeta*dzdxi
            dbi = dxdeta*dydxi*dzdzeta
            ahf = dxdxi*dydzeta*dzdeta
            
            detJ = aei + dhc + bfg - gec - dbi - ahf
                      
            # Computing comatrix of J
            
            ComJ_11 = dydeta * dzdzeta - dzdeta * dydzeta
            ComJ_21 = - (dxdeta * dzdzeta - dzdeta * dxdzeta)
            ComJ_31 = dxdeta * dzdzeta - dydeta * dxdzeta
            
            ComJ_12 = -(dydxi * dzdzeta - dzdxi * dydzeta)
            ComJ_22 = dxdxi * dzdzeta - dzdxi * dxdzeta
            ComJ_32 = -(dxdxi * dydzeta - dydxi * dxdzeta)
            
            ComJ_13 = dydxi * dzdeta - dzdxi * dydeta
            ComJ_23 = -(dxdxi * dzdeta - dzdxi * dxdeta)
            ComJ_33 = dxdxi * dydeta - dydxi * dzdeta
            

            dphidx = sps.diags(ComJ_11/detJ).dot(dphidxi) + \
                sps.diags(ComJ_12/detJ).dot(dphideta) + \
                sps.diags(ComJ_13/detJ).dot(dphidzeta)
            dphidy = sps.diags(ComJ_21/detJ).dot(dphidxi) + \
                sps.diags(ComJ_22/detJ).dot(dphideta) + \
                sps.diags(ComJ_23/detJ).dot(dphidzeta)
            dphidz = sps.diags(ComJ_31/detJ).dot(dphidxi) + \
                sps.diags(ComJ_32/detJ).dot(dphideta) + \
                sps.diags(ComJ_33/detJ).dot(dphidzeta)  
                              
            return phi, dphidx, dphidy, dphidz, detJ
              


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


def SplineFromROI(roi, dx, degree=np.array([2, 2])):
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
    m.ctrlPts = m.N2CtrlPts()
    p = np.array([1., 0., 0., 0.])
    cam = Camera(p)
    return m, cam

def make_stl_mesh(spline, ctrl_pts, n_eval_per_elem=10, remove_empty_areas=True):
    
    try:
        from stl import mesh
    except:
        raise Exception("stl is not installed")
    tri = []
    XI = spline.linspace(n_eval_per_elem=n_eval_per_elem)
    shape = [xi.size for xi in XI]
    for axis in range(3):
        XI_axis = [xi for xi in XI]
        shape_axis = [shape[i] for i in range(len(shape)) if i!=axis]
        XI_axis[axis] = np.zeros(1)
        pts_l = spline(ctrl_pts, tuple(XI_axis), [0, 0, 0]).reshape([3] + shape_axis)
        XI_axis[axis] = np.ones(1)
        pts_r = spline(ctrl_pts, tuple(XI_axis), [0, 0, 0]).reshape([3] + shape_axis)
        for pts in [pts_l, pts_r]:
            A = pts[:,  :-1,  :-1].reshape((3, -1)).T[:, None, :]
            B = pts[:,  :-1, 1:  ].reshape((3, -1)).T[:, None, :]
            C = pts[:, 1:  ,  :-1].reshape((3, -1)).T[:, None, :]
            D = pts[:, 1:  , 1:  ].reshape((3, -1)).T[:, None, :]
            tri1 = np.concatenate((A, B, C), axis=1)
            tri2 = np.concatenate((D, C, B), axis=1)
            tri.append(np.concatenate((tri1, tri2), axis=0))
    tri = np.concatenate(tri, axis=0)
    data = np.empty(tri.shape[0], dtype=mesh.Mesh.dtype)
    data['vectors'] = tri
    m = mesh.Mesh(data, remove_empty_areas=remove_empty_areas)
    return m

# %% Test

if __name__ == "__main__":
    

    f = px.Image('zoom-0053_1.tif').Load()
    # f.Plot()
    g = px.Image('zoom-0070_1.tif').Load()

    a = 0.925
    Xi = np.array([[0.5, 0.75, 1],
                [0.5*a, 0.75*a, 1*a],
                [0,0, 0]])
    Yi = np.array([[0, 0, 0],
                [0.5*a, 0.75*a, 1*a],
                [0.5, 0.75, 1]])

    ctrlPts = np.array([Xi.T, Yi.T])
    degree = [2, 2]
    kv1 = np.array([0, 0, 0, 1, 1, 1])
    kv2 = np.array([0, 0, 0, 1, 1, 1])
    knotVect = [kv1, kv2]

    n = 5
    newr = np.linspace(0, 1, n+2)[1:-1]
    n = 10
    newt = np.linspace(0, 1, n+2)[1:-1]
    m = BSplinePatch(ctrlPts, degree, knotVect)
    m.Plot()


    m.KnotInsertion([newr, newt])
    # m.DegreeElevation(np.array([3, 3]))
    #m.Plot()

    cam = px.Camera([100, 6.95, -5.35, 0])

    m.Connectivity()
    # %% gauss
    m.GaussIntegration()
    
    
    
    
    # %% Pas gauss
    # m.DICIntegration(cam)
    U = px.MultiscaleInit(f, g, m, cam, scales=[2, 1])
    U, res = px.Correlate(f, g, m, cam, U0=U)

    m.Plot(U, alpha=0.5)
    m.Plot(U=3*U)


