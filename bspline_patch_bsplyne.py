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
import time
from tqdm import tqdm
try :
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except :
    print('mpi4py is not installed')


# %%


class Elem_2D:
    def __init__(self, xi, eta, num):
        
        self.num = num
        self.xi = xi
        self.eta = eta
        self.mes_xi = self.xi[1] - self.xi[0]
        self.mes_eta = self.eta[1] - self.eta[0]
        
        self.wdetJ = 0
        
        self.xig = 0
        self.etag = 0
        
        self.pgx = 0
        self.pgy = 0
        
class Elem_3D:
    def __init__(self, xi, eta, zeta, num):
        
        self.num = num
        self.xi = xi
        self.eta = eta
        self.zeta = zeta
        self.mes_xi = self.xi[1] - self.xi[0]
        self.mes_eta = self.eta[1] - self.eta[0]
        self.mes_zeta = self.zeta[1] - self.zeta[0]
        
        self.wdetJ = 0
        
        self.xig = 0
        self.etag = 0
        self.zetag = 0
        
        self.pgx = 0
        self.pgy = 0
        self.pgz = 0
        
        
class BSplinePatch(object):
    def __init__(self, e, n, degree, knotVect):
        """
        Nurbs surface from R^2 (xi,eta)--->R^2 (x,y) 
        ctrlPts = [X,Y] or ctrlPts = [X,Y,Z]
        """
        self.dim = n.shape[1]
        # self.ctrlPts = ctrlPts
        # n = self.CtrlPts2N()
        self.n = n
        self.e = e
        self.dof = n
        self.ndof = 0
        self.conn = []
        self.degree = np.array(degree)
        self.knotVect = knotVect
        self.spline = bs.BSpline(self.degree, self.knotVect)
        self.nbf = self.spline.getNbFunc()


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
        
        # Dictionary containing the element of the bspline mesh
        if self.dim == 2:
            self.elem = self.Init_elem_2D()
        elif self.dim == 3:
            self.elem = self.Init_elem_3D()
            
        # self.C = self.Connectivity_Matrix()
        self.nelem = 0
        # self.n_knots = self.Init_n_knots()
        # self.e_knots = {0 : np.arange(self.n_knots.shape[0])}
        
        """ Attributes when using vectorization  """
        """ In this case, the implicit connectivity of the structured B-spline parametric space is used """
        self.npg = 0
        self.phix = 0      # Matrix (N,0)
        self.phiy = 0      # Matrix (0,N)
        self.phiz = 0     # Matrix ?????

        self.dphix = np.empty(0)
        self.dphiy = np.empty(0)
        self.dphiz = np.empty(0)

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
        m = BSplinePatch(self.e.copy(), self.n.copy(), self.degree.copy(),
                         self.knotVect.copy())
        m.conn = self.conn.copy()
        m.dof = self.dof.copy()
        m.ndof = self.ndof
        m.dim = self.dim
        m.npg = self.npg
        m.pgx = self.pgx.copy()
        m.pgy = self.pgy.copy()
        m.pgz = self.pgz.copy()
        m.phi = self.phi.copy()
        # m.phix = self.phix.copy()
        #m.phiy = self.phiy.copy()
        # m.phiz = self.phiy.copy()
        m.dphixdx = self.dphixdx.copy()
        m.dphixdy = self.dphixdy.copy()
        m.dphixdz = self.dphixdz.copy()
        
        m.dphiydx = self.dphiydx.copy()
        m.dphiydy = self.dphiydy.copy()
        m.dphiydz = self.dphiydz.copy()
        
        m.dphizdx = self.dphizdx.copy()
        m.dphizdy = self.dphizdy.copy()
        m.dphizdz = self.dphizdz.copy()
        m.wdetJ = self.wdetJ.copy()
        # m.C = self.C.copy()
        return m

    def IsRational(self):
        ctrlPts = self.N2CtrlPts()
        return (ctrlPts[3] != 1).any()



    def Get_nbf_1d(self):
        """ Get the number of basis functions per parametric direction """
        ctrlPts = self.N2CtrlPts()
        return ctrlPts.shape[1:]

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


    def Init_elem_2D(self):
        """
        Create a dictionnary containing all the element

        Returns
        -------
        dictionnary

        """

        e = dict()
        xiu = np.unique(self.knotVect[0])
        etau = np.unique(self.knotVect[1])
        n = 0
        for i in range(xiu.shape[0]-1):
            xi = np.array([xiu[i], xiu[i+1]])
            for j in range(etau.shape[0]-1):
        
                eta = np.array([etau[j], etau[j+1]])
                
    
                e[n] = Elem_2D(xi, eta, n)
                n += 1              
        
        self.nelem = n
        return e
    
    def Init_elem_3D(self):
        """
        Create a dictionnary containing all the element

        Returns
        -------
        dictionnary

        """

        e = dict()
        xiu = np.unique(self.knotVect[0])
        etau = np.unique(self.knotVect[1])
        zetau = np.unique(self.knotVect[2])
        self.nelem = (xiu.shape[0]-1) * (etau.shape[0]-1) * (zetau.shape[0]-1)
        n = 0
        for i in range(xiu.shape[0]-1):
            xi = np.array([xiu[i], xiu[i+1]])
            for j in range(etau.shape[0]-1):
                eta = np.array([etau[j], etau[j+1]])
                for k in range(zetau.shape[0]-1):
                    
                    zeta = np.array([zetau[k], zetau[k+1]])
                    e[n] = Elem_3D(xi, eta, zeta, n)
                    n += 1              
           
        return e    
    
    
    
    def Init_n_knots(self):
        
        
        n_knots = np.empty(self.dim)
        
        spline = self.Get_spline()
        P = self.n
        for key in self.elem.keys():
            
            elem = self.elem[key]
            
            for i in range(2):
                
                N = spline.DN(np.array([elem.xi[i], elem.eta[i], elem.zeta[i]]))
                xk, yk, zk = N @ P [:, 0], N @ P [:, 1], N @ P [:, 2]
                n_knots = np.vstack((n_knots, np.array([xk, yk, zk]).T))
                # self.n_knots.append(np.array([xk, yk, zk]))
         
        
        return n_knots
            
  

    # def _get_e(self):
    #     """
    #     Generate a list that number the control points

    #     Returns
    #     -------
    #     p : ndarray.

    #     """
        
    #     nu, ind = np.unique(self.n, axis=0, return_index=True)
    #     nu = nu[np.argsort(ind)]
    #     e = np.zeros(self.n.shape[0])
    #     for i in range(len(self.n)):
            
    #         for j in range(len(nu)):
    #             if (self.n[i] == nu[j]).all():
    #                 index = j
              
    #         e[i] = index

    #     return e

    def Connectivity(self):
        print("Connectivity.")
        # used_nodes = np.zeros(0, dtype=int)
        # for je in self.e.keys():
        #     used_nodes = np.unique(np.append(used_nodes, self.e[je].ravel()))
        # nn = len(used_nodes)
        nn = len(self.n)
        self.ndof = nn * self.dim
        # self.conn = -np.ones(self.dof.shape[0], dtype=int)
        # self.conn[used_nodes] = np.arange(nn)
        self.conn = np.arange(nn)
        
        if self.dim == 2:
            # self.conn = np.c_[self.conn, self.conn + nn * (self.conn >= 0)]
            self.conn = np.c_[self.conn + 0 * nn, 
                              self.conn + 1 * nn]
        else:
            self.conn = np.c_[
                self.conn + 0 * nn,
                self.conn + 1 * nn,
                self.conn + 2 * nn,
                # self.conn + nn * (self.conn >= 0),
                # self.conn + 2 * nn * (self.conn >= 0),
                ]
        
      
      
    def Connectivity_Matrix(self):    
        """
        Return a connectivity matrix C of shape (N_PC, N_Nodes). 
        It allows the shape function evaluation matrix self.phi matrix to

        Returns
        -------
        None.

        """
        e = self.e[0].ravel()
        e_uniq, count = np.unique(e, return_counts=True)
    
        row = np.arange(len(e))
        col = e
        val = np.ones_like(col) 
        
        C = sps.csr_matrix((val, (row, col)), shape=(self.nbf, self.n.shape[0]))
        # self.C = C
        return C
        
        
        
    def ConnectivityKnots(self):
        print("Knot Connectivity.")
        used_knots = np.zeros(0, dtype=int)
        for je in self.e_knots.keys():
            used_knots = np.unique(np.append(used_knots, self.e_knots[je].ravel()))
        nn = len(used_knots)
        # self.ndof = nn * self.dim
        self.conn_knots = -np.ones(self.n_knots.shape[0], dtype=int)
        self.conn_knots[used_knots] = np.arange(nn)
       
        if self.dim == 2:
            self.conn = np.c_[self.conn_knots, self.conn_knots + nn * (self.conn_knots >= 0)]
        else:
            self.conn = np.c_[
                self.conn_knots,
                self.conn_knots + nn * (self.conn_knots >= 0),
                self.conn_knots + 2 * nn * (self.conn_knots >= 0),
                ]


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
        # n = np.unique(n, axis=0)
        return n

    def N2CtrlPts(self, n=None):
        # n should be in the right order (xi, eta) meshgrid
        if n is None:
            n = self.n.copy()
        if self.dim == 2:
            cpx = n[self.e[0], 0]
            cpy = n[self.e[0], 1]
            ctrlPts = np.array([cpx, cpy])
            # nbf = self.Get_nbf_1d()
            # ctrlPts = np.array([n[:, 0].reshape(nbf),
            #                     n[:, 1].reshape(nbf)])
        elif self.dim == 3:
            cpx = n[self.e[0], 0]
            cpy = n[self.e[0], 1]
            cpz = n[self.e[0], 2]
            ctrlPts = np.array([cpx, cpy, cpz])
        return ctrlPts

    def saveParaview(self, fname, cam=None, U=None):
        """
        Save a pvd file corresponding to the bspline mesh

        Returns
        -------
        None.
        """
        
        spline = self.spline
        if U is None:
            spline.saveParaview(self.ctrlPts, ".", fname)
        else:
            U = U.reshape(3,-1)
            U = np.array(cam.P(*U))
            Ures = U.reshape((1,*self.N2CtrlPts().shape))
            D = {"U" : Ures}
            spline.saveParaview(self.N2CtrlPts(), './', 'deformed', fields=D)


    def Plot(self, U=None, n=None, neval=None, **kwargs):
        """ Physical elements = Image of the parametric elements on Python """
        
        if neval == None:
            if self.dim == 2:
                neval = [100, 100]
            elif self.dim == 3:
                neval = [30, 30, 30]
        
        alpha = kwargs.pop("alpha", 1)
        edgecolor = kwargs.pop("edgecolor", "k")
        
        if n is None:
            Px = self.n[:, 0]
            Py = self.n[:, 1]
        else:
            Px = n[:, 0]
            Py = n[:, 1]           
        
        if U is None:
            U = np.zeros(self.dim * self.n.shape[0])
  
        
        U = U.reshape((self.dim, -1))
       

    
        Pxm = Px + U[0]
        Pym = Py + U[1]
        P = np.c_[Pxm, Pym]
        
        P = self.N2CtrlPts(P)
        
        xi = np.linspace(
            self.knotVect[0][self.degree[0]], self.knotVect[0][-self.degree[0]], neval[0])
        eta = np.linspace(
            self.knotVect[1][self.degree[1]], self.knotVect[1][-self.degree[1]], neval[1])

        # Iso parameters for the elemnts
        xiu = np.unique(self.knotVect[0])
        etau = np.unique(self.knotVect[1])

        # Mapping from parametric to physical space
        # P = self.CtrlPts2N(P)
        P = self.CtrlPts2N(P)
        xe1, ye1 = self.Mapping(xiu, eta, type_eval='grid', P=P)
        xe2, ye2 = self.Mapping(xi, etau, type_eval='grid', P=P)

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
        # plt.title(f"BSpline in physical space with {self.n.shape[0]} nodes")
        plt.show()

    def _init_cube(self, neval=50):
        
        if type(neval) == int:
            neval = np.array([neval, neval, neval])
        
        x = np.linspace(0, 1, neval[0]) 
        y = np.linspace(0, 1, neval[1])
        z = np.linspace(0, 1, neval[2])
        meshgrid = np.meshgrid(x, y, z, indexing='ij') 
        coordonnees_surface_cube = []
        for i in range(len(x)):
           for j in range(len(y)):
               for k in range(len(z)):
                   if i == 0 or i == len(x) - 1 or j == 0 or j == len(y) - 1 or k == 0 or k == len(z) - 1:
                       coordonnees_surface_cube.append((meshgrid[0][i, j, k], meshgrid[1][i, j, k], meshgrid[2][i, j, k]))

        return np.array(coordonnees_surface_cube)
    
    def Plot3D(self, n=None, neval=None, U=None, **kwargs):
        
        
        if neval == None:
            neval = [50, 50, 50]
        alpha = kwargs.pop("alpha", 1)
        edgecolor = kwargs.pop("edgecolor", "k")
        nbf = self.nbf
        if n is None:
            n = self.n # control points
        if U is None:
            U = np.zeros(self.dim *nbf)
        U = U.reshape((self.dim, -1))

        Pxm = n[:, 0] + U[0]
        Pym = n[:, 1] + U[1]
        Pzm = n[:, 2] + U[2]
        
        coord = self._init_cube(neval=neval).T
        
        x, y, z = self.Mapping(xi=coord[0], eta=coord[1], zeta=coord[2])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x.T, y.T, z.T)
        plt.show()
        
    """
        xi = np.linspace(
            self.knotVect[0][self.degree[0]], self.knotVect[0][-self.degree[0]], neval[0])
        eta = np.linspace(
            self.knotVect[1][self.degree[1]], self.knotVect[1][-self.degree[1]], neval[1])
        zeta = np.linspace(
            self.knotVect[2][self.degree[2]], self.knotVect[2][-self.degree[1]], neval[2])
    
        # Iso parameters for the elemnts
        xiu = np.unique(self.knotVect[0])
        etau = np.unique(self.knotVect[1])
        zetau = np.unique(self.knotVect[2])
    
        # Basis functions
        spline = self.Get_spline()
        spline = bs.BSpline(self.degree, self.knotVect)
    
        phi1 = spline.DN([xiu, eta, zeta], k=[0, 0, 0])
        phi2 = spline.DN([xi, etau, zeta], k=[0, 0, 0])
        phi3 = spline.DN([xi, eta, zetau], k=[0, 0, 0])
    
        #xe, ye1 = spline(self.ctrlPts)
        xe1 = phi1.dot(Pxm)
        ye1 = phi1.dot(Pym)
        ze1 = phi1.dot(Pzm)
        xe2 = phi2.dot(Pxm)
        ye2 = phi2.dot(Pym)
        ze2 = phi2.dot(Pzm)
        xe3 = phi3.dot(Pxm)
        ye3 = phi3.dot(Pym)
        ze3 = phi3.dot(Pzm)
    
        xe1 = xe1.reshape((xiu.size, neval[1], neval[2]))
        ye1 = ye1.reshape((xiu.size, neval[1], neval[2]))
        ze1 = ze1.reshape((xiu.size, neval[1], neval[2]))
        xe2 = xe2.reshape((neval[0], etau.size, neval[2]))
        ye2 = ye2.reshape((neval[0], etau.size, neval[2]))
        ze2 = ze2.reshape((neval[0], etau.size, neval[2]))
        xe3 = xe3.reshape((neval[0], neval[1], zetau.size))
        ye3 = ye3.reshape((neval[0], neval[1], zetau.size))
        ze3 = ze3.reshape((neval[0], neval[1], zetau.size))
    
        
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(xiu.size):
            # loop on xi
            # Getting one eta iso-curve
            for j in range(neval[1]):
                ax.plot(xe1[i, j, :], ye1[i, j, :], ze1[i, j, :], color=edgecolor,
                     alpha=alpha, **kwargs)
        
        for i in range(etau.size):
            # loop on xi
            # Getting one eta iso-curve
            for j in range(neval[1]):
                ax.plot(xe2[:, i, j], ye2[:, i, j], ze2[:, i, j], color=edgecolor,
                     alpha=alpha, **kwargs)
        
        for i in range(zetau.size):
            # loop on xi
            # Getting one eta iso-curve
            for j in range(neval[0]):
                ax.plot(xe3[j, :, i], ye3[j, :, i], ze3[j, :, i], color=edgecolor,
                     alpha=alpha, **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter(Pxm, Pym, Pzm, c=edgecolor)
        plt.show()
        """
    


    def DegreeElevation(self, new_degree):

        spline = self.spline

        t = new_degree - self.degree
        self.ctrlPts = spline.orderElevation(self.ctrlPts, t)
        self.degree = np.array(new_degree)
        self.knotVect = spline.getKnots()
        self.n = self.CtrlPts2N()
        self.e = {0: np.arange(self.n.shape[0])}
        self.spline = bs.BSpline(self.degree, self.knotVect)
        self.nbf = self.spline.getNbFunc()
        if self.dim == 2:
            self.elem = self.Init_elem_2D()
        else:
            self.elem = self.Init_elem_3D()

    def KnotInsertion(self, knots):

        spline = self.spline
        ctrlPts = self.N2CtrlPts()
        ctrlPts = spline.knotInsertion(ctrlPts, knots)
        self.knotVect = spline.getKnots()
        self.n = self.CtrlPts2N(ctrlPts=ctrlPts)
        self.e = {0: np.arange(self.n.shape[0])}
        self.spline = bs.BSpline(self.degree, self.knotVect)
        self.nbf = self.spline.getNbFunc()
        # self.C = self.Connectivity_Matrix()
        
        
        if self.dim == 2:
            self.elem = self.Init_elem_2D()
        elif self.dim == 3:
            self.elem = self.Init_elem_3D()

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
        """Assembles Tikhonov (Laplacian) Operator"""
        
        # if not hasattr(self, "dphixdx"):
        #     m = self.Copy()
        #     m.GaussIntegration()
        # else:
        #     m = self
        if self.dphixdx.shape[0] == 0:
            m2 = self.Copy()
            m2.GaussIntegration()
        else:
            m2 = self
        wdetJ = sp.sparse.diags(m2.wdetJ)
        if self.dim == 3:
            #print(m.dphixdx.shape, wdetJ.shape)
            # print((m2.dphixdx.T @ wdetJ @ m2.dphixdx).shape)
            L = m2.dphixdx.T @ wdetJ @ m2.dphixdx + \
                m2.dphixdy.T @ wdetJ @ m2.dphixdy + \
                m2.dphixdz.T @ wdetJ @ m2.dphixdz + \
                m2.dphiydx.T @ wdetJ @ m2.dphiydx + \
                m2.dphiydy.T @ wdetJ @ m2.dphiydy + \
                m2.dphiydz.T @ wdetJ @ m2.dphiydz + \
                m2.dphizdx.T @ wdetJ @ m2.dphizdx + \
                m2.dphizdy.T @ wdetJ @ m2.dphizdy + \
                m2.dphizdz.T @ wdetJ @ m2.dphizdz
                
            
        else:
            L = m2.dphixdx.T @ wdetJ @ m2.dphixdx + \
                m2.dphiydy.T @ wdetJ @ m2.dphiydy + \
                m2.dphixdy.T @ wdetJ @ m2.dphixdy + \
                m2.dphiydx.T @ wdetJ @ m2.dphiydx
 
        return L

    def DoubleLaplacian(self):
        wg = sps.diags(self.wdetJ)
        return 2*self.dphixdxx.T.dot(wg.dot(self.dphixdyy)) +\
            2*self.dphiydxx.T.dot(wg.dot(self.dphiydyy)) +\
            self.dphixdxx.T.dot(wg.dot(self.dphixdxx)) +\
            self.dphixdyy.T.dot(wg.dot(self.dphixdyy)) +\
            self.dphiydxx.T.dot(wg.dot(self.dphiydxx)) +\
            self.dphiydyy.T.dot(wg.dot(self.dphiydyy))

    

    def compute_bbox_area(self, cam, e):
        """
        Compute a box around an bspline element e

        Parameters
        ----------
        cam : Camera model
        
        e : element from  Elem_2D / Elem_3D
            .

        Returns
        -------
        bbox_area : float
            area of the box around element

        """        
        if self.dim == 2: 
            
            x, y = self.Mapping(xi=e.xi, eta=e.eta, type_eval='grid')
            
            u, v = cam.P(x, y)
            
            
            l = max(u) - min(u)
            L = max(v) - min(v)
            
            bbox_area = l * L
            return bbox_area
        
        if self.dim == 3: 
            
            x, y, z = self.Mapping(xi=e.xi, eta=e.eta, zeta=e.zeta, type_eval='grid')
            u, v, w = cam.P(x, y, z)
            
            
            l = max(u) - min(u)
            L = max(v) - min(v)
            LL = max(w) - min(w)

            bbox_area = l * L * LL
            return bbox_area
        
     
     
    def compute_largest_edge(self, cam, e):
        """
        Compute the largest edge of  a rectange that approximate the element e

        Parameters
        ----------
       cam : Camera model
       
       e : element from  Elem_2D / Elem_3D
           .

       Returns
       -------
       L_max : int
           Rounded lenght of the longest edge

        """

        
        if self.dim == 2:
            
            x, y = self.Mapping(xi=e.xi, eta=e.eta, type_eval='grid')
            u, v = cam.P(x, y)
            
            # First corner
            L1 = np.sqrt( (u[1] - u[0]) ** 2 + (v[1]- v[0]) ** 2)
            L2 = np.sqrt( (u[2] - u[0]) ** 2 + (v[2]- v[0]) ** 2)
            
            # Last corner
            L3 = np.sqrt( (u[-2] - u[-1]) ** 2 + (v[1]- v[0]) ** 2)
            L4 = np.sqrt( (u[-3] - u[-1]) ** 2 + (v[2]- v[0]) ** 2)

            
            L_max = max(L1, L2, L3, L4)
            return int(np.ceil(L_max))
        
        if self.dim == 3:
            
            x, y, z = self.Mapping(xi=e.xi, eta=e.eta, zeta=e.zeta, type_eval='grid')
            u, v, w = cam.P(x, y, z)
            
            
            # First corner
            L1 = np.sqrt( (u[1] - u[0]) ** 2 + (v[1]- v[0]) ** 2 + (w[1]- w[0]) ** 2)
            L2 = np.sqrt( (u[2] - u[0]) ** 2 + (v[2]- v[0]) ** 2 + (w[2]- w[0]) ** 2)
            L3 = np.sqrt( (u[3] - u[0]) ** 2 + (v[3]- v[0]) ** 2 + (w[3]- w[0]) ** 2)
            
            # Last corner
            L4 = np.sqrt( (u[-2] - u[-1]) ** 2 + (v[-2]- v[-1]) ** 2 + (w[-2]- w[-1]) ** 2)
            L5 = np.sqrt( (u[-3] - u[-1]) ** 2 + (v[-3]- v[-1]) ** 2 + (w[-3]- w[-1]) ** 2)
            L6 = np.sqrt( (u[-4] - u[-1]) ** 2 + (v[-4]- v[-1]) ** 2 + (w[-4]- w[-1]) ** 2)
            

            
            L_max = max(L1, L2, L3, L4, L5, L6)
            return int(L_max)
            

    def InverseBSplineMapping(self, x, y, init=None, elem=None):
        """ 
        Inverse the BSpline mapping in order to map the coordinates
        of any physical points (x, y, z) to their corresponding position in
        the parametric space (xi_g, eta_g).

        Parameter : 
            m : bspline mesh from pyxel
            x : x coordinate in physical space
            y : y coordinate in physical space

        Return :
            xg, yg : x coordinate 
        """

        # Basis function
        spline = self.spline
        ctrlPts = self.N2CtrlPts()
        
        # Coordinates of controls points
        xn = ctrlPts[0].ravel()
        yn = ctrlPts[1].ravel()

        # Initializing  parametric integration points to zero
        if init is None :
            if elem is None:
                xi_g =  0 * x
                eta_g = 0 * y
            
            else:
                xi_g = elem.xi[0] * np.ones_like(x)
                eta_g = elem.eta[0] * np.ones_like(y)
            
        else :
            xi_g = init[0]
            eta_g = init[1]
        
        
        maxiter = 3
        # Newton loop
        for k in range(maxiter):
            
            
            
            N = spline.DN(np.array([xi_g, eta_g]), k=[0, 0])
            dNdxi = spline.DN(np.array([xi_g, eta_g]), k=[1, 0])
            dNdeta = spline.DN(np.array([xi_g, eta_g]), k=[0, 1])

            dxdxi = dNdxi @ xn
            dydxi = dNdxi @ yn
            dxdeta = dNdeta @ xn
            dydeta = dNdeta @ yn

            detJ = dxdxi * dydeta - dydxi * dxdeta
            invJ = np.array([dydeta / detJ, -dxdeta / detJ,
                             -dydxi / detJ, dxdxi / detJ]).T

            xp = N @ xn
            yp = N @ yn

            dxi_g = invJ[:, 0] * (x - xp) + invJ[:, 1] * (y - yp)
            deta_g = invJ[:, 2] * (x - xp) + invJ[:, 3] * (y - yp)

            xi_g = xi_g + dxi_g
            eta_g = eta_g + deta_g
            
            res = max(np.linalg.norm(x - xp),
                      np.linalg.norm(y - yp))
            
            if elem is None:
                
                if k < 1:
                    
                    select = (xi_g == np.clip(xi_g, 0, 1)) &\
                             (eta_g == np.clip(eta_g, 0, 1))
                
                    xi_g = xi_g[select]
                    eta_g = xi_g[select]
                    
                    x = x[select]
                    y = y[select]
                
                else:
                    
                    xi_g = np.clip(xi_g, 0, 1)
                    eta_g = np.clip(eta_g, 0, 1)
                    
                
            else :
                
                if k < 1:
                    
                    select = (xi_g == np.clip(xi_g, elem.xi[0], elem.xi[1])) &\
                             (eta_g == np.clip(eta_g, elem.eta[0], elem.eta[1]))&\
                             (np.isfinite(xi_g))&\
                             (np.isfinite(eta_g))
                
                    xi_g = xi_g[select]
                    eta_g = eta_g[select]
                    x = x[select]
                    y = y[select]
             
            # print("XI", xi_g.min(), xi_g.max())
            # print("ETA", eta_g.min(), eta_g.max())
            print(f"Res : {res}")
            if res < 1.0e-4:
                break
        
        
        return xi_g, eta_g
    
    def DICIntegrationPixel(self, f, cam):
        
        
        ctrlPts = self.N2CtrlPts()
        Px = ctrlPts[0].ravel()
        Py = ctrlPts[1].ravel()
        
        spline = self.spline
        
        xi_glob = np.empty(0)
        eta_glob = np.empty(0)
        
        for key in self.elem:
            
            elem = self.elem[key]
            
            print(f"\nÉlément {key}")
            
            N_pix = self.compute_largest_edge(cam, elem)
            N_pix = int(N_pix * 3)
            
            xi = np.linspace(elem.xi[0], elem.xi[1], N_pix)
            eta = np.linspace(elem.eta[0], elem.eta[1], N_pix)
            param = np.meshgrid(xi, eta, indexing='ij')
            
            N = spline.DN([xi, eta], k=[0,0])
            # N = N @ self.C
            
            u, v = cam.P(N @ Px, N @ Py)
            
            ur = np.round(u).astype('uint16')
            vr = np.round(v).astype('uint16')
            
            Nx = f.pix.shape[0]
            Ny = f.pix.shape[1]
            
            idpix = np.ravel_multi_index((ur, vr), (Nx, Ny))
            _, rep = np.unique(idpix, return_index=True)
                    
            ur = ur[rep]
            vr = vr[rep]
            
            xi_init = param[0].ravel()[rep]
            eta_init = param[1].ravel()[rep]
                    
            init = [xi_init, eta_init]
            
            xg, yg = cam.Pinv(ur.astype(float), vr.astype(float))
            
            xi, eta = self.InverseBSplineMapping(xg, yg, init=init, elem=elem)
            
            select = (xi > elem.xi[0]) & (xi < elem.xi[1]) &\
                 (eta > elem.eta[0]) & (eta < elem.eta[1]) 
            
            xi = xi[select]
            eta = eta[select]
            
            xi_glob = np.hstack((xi_glob, xi))
            eta_glob = np.hstack((eta_glob, eta))
            
        phi = self.ReshapePhi(spline.DN(np.array([xi_glob, eta_glob]), k=[0, 0]))

        self.npg = phi.shape[0]
        self.wdetJ = np.ones_like(xi_glob)
        
        
        self.phi = phi # @ self.C

        self.pgx = self.phi @ Px
        self.pgy = self.phi @ Py 
            
    
    def GaussIntegration(self, npg=None, P=None):
        """ Gauss integration: build of the global differential operators """
        if self.dim == 2:
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
                ctrlPts = self.N2CtrlPts()
                Px = ctrlPts[0].ravel()
                Py = ctrlPts[1].ravel()
            else:
                Px = P[:, 0]
                Py = P[:, 1]
    
            """ Spatial derivatives """
    
            N, dNdx, dNdy, detJ = self.ShapeFunctions(xi, eta, type_eval='grid')
            
            self.npg = N.shape[0]
    
            """ Integration weights + measures + Jacobian of the transformation """
            self.wdetJ = np.kron(wg_eta, wg_xi)*np.abs(detJ)*mes_xi*mes_eta/4
            zero = sps.csr_matrix((self.npg, self.n.shape[0]))
            self.phi = self.ReshapePhi(N) 
            self.dphidx = self.ReshapePhi(dNdx)
            self.dphidy = self.ReshapePhi(dNdy)
            
            # self.phix = sps.hstack((phi, zero),  'csc')
            # self.phiy = sps.hstack((zero, phi),  'csc')
            self.dphixdx = sps.hstack((self.dphidx, zero),  'csc')
            self.dphixdy = sps.hstack((self.dphidy, zero),  'csc')
            self.dphiydx = sps.hstack((zero, self.dphidx),  'csc')
            self.dphiydy = sps.hstack((zero, self.dphidy),  'csc')
            
            # print(N.shape, Px.shape)
            self.pgx = N @ Px
            self.pgy = N @ Py
            
        if self.dim == 3 :
            
            if npg is None:
                    nbg_xi = self.degree[0]+1
                    nbg_eta = self.degree[1]+1
                    nbg_zeta = self.degree[2]+1
            else:
                    nbg_xi = npg[0]
                    nbg_eta = npg[1]
                    nbg_zeta = npg[2]
        
            GaussLegendre = np.polynomial.legendre.leggauss
        
            Gauss_xi = GaussLegendre(nbg_xi)
            Gauss_eta = GaussLegendre(nbg_eta)
            Gauss_zeta = GaussLegendre(nbg_zeta)
        
            e_xi = np.unique(self.knotVect[0])
            ne_xi = e_xi.shape[0]-1
            e_eta = np.unique(self.knotVect[1])
            ne_eta = e_eta.shape[0]-1
            e_zeta = np.unique(self.knotVect[2])
            ne_zeta = e_zeta.shape[0]-1
            xi_min = np.kron(e_xi[:-1], np.ones(nbg_xi))
            eta_min = np.kron(e_eta[:-1], np.ones(nbg_eta))
            zeta_min = np.kron(e_zeta[:-1], np.ones(nbg_zeta))
            xi_g = np.kron(np.ones(ne_xi), Gauss_xi[0])
            eta_g = np.kron(np.ones(ne_eta), Gauss_eta[0])
            zeta_g = np.kron(np.ones(ne_zeta), Gauss_zeta[0])
        
    
            """ Measures of elements """
            mes_xi = e_xi[1:] - e_xi[:-1]
            mes_eta = e_eta[1:] - e_eta[:-1]
            mes_zeta = e_zeta[1:] - e_zeta[:-1]
    
            mes_xi = np.kron(mes_xi, np.ones(nbg_xi))
            mes_eta = np.kron(mes_eta, np.ones(nbg_eta))
            mes_zeta = np.kron(mes_zeta, np.ones(nbg_zeta))
    
            """ Going from the reference element to the parametric space  """
            # Aranged gauss points in  xi direction
            xi = xi_min + 0.5 * (xi_g + 1) * mes_xi
            # Aranged gauss points in  eta direction
            eta = eta_min + 0.5 * (eta_g + 1) * mes_eta
            # Aranged gauss points in  zeta direction
            zeta = zeta_min + 0.5 * (zeta_g + 1) * mes_zeta
    
            wg_xi = np.kron(np.ones(ne_xi), Gauss_xi[1])
            wg_eta = np.kron(np.ones(ne_eta), Gauss_eta[1])
            wg_zeta = np.kron(np.ones(ne_zeta), Gauss_zeta[1])
    
    
            mes_xi = np.kron(np.ones(eta.shape[0]), mes_xi)
            mes_xi = np.kron(np.ones(zeta.shape[0]), mes_xi)
            
            mes_eta = np.kron(mes_eta, np.ones(xi.shape[0]))
            mes_eta = np.kron(np.ones(zeta.shape[0]), mes_eta)
            
            mes_zeta = np.kron(mes_zeta, np.ones(eta.shape[0]))
            mes_zeta = np.kron(mes_zeta, np.ones(xi.shape))
    
            w = np.kron(wg_zeta, np.kron(wg_eta, wg_xi))
            detGauss = mes_xi * mes_eta * mes_zeta / 8
    
            if P is None:
                ctrlPts = self.N2CtrlPts()
                Px = ctrlPts[0].ravel()
                Py = ctrlPts[1].ravel()
                Pz = ctrlPts[2].ravel()
            else:
                Px = P[:, 0]
                Py = P[:, 1]
                Pz = P[:, 2]

    
            """ Spatial derivatives """
    
            N, dNdx, dNdy, dNdz, detJ = self.ShapeFunctions(xi, eta, zeta, type_eval='grid')
        
            
            self.npg = N.shape[0]
            
            """ Integration weights + measures + Jacobian of the transformation """
            self.wdetJ = w*np.abs(detJ)*detGauss
            
            
            
            self.phi = self.ReshapePhi(N)
            self.dphidx = self.ReshapePhi(dNdx)
            self.dphidy = self.ReshapePhi(dNdy)
            self.dphidz = self.ReshapePhi(dNdz)
            
            zero = sps.csr_matrix((self.npg, self.phi.shape[1]))
            
            self.dphixdx = sps.hstack((self.dphidx, zero, zero),  'csc')
            self.dphixdy = sps.hstack((self.dphidy, zero, zero),  'csc')
            self.dphixdz = sps.hstack((self.dphidz, zero, zero),  'csc')
            self.dphiydx = sps.hstack((zero, self.dphidx, zero),  'csc')
            self.dphiydy = sps.hstack((zero, self.dphidy, zero),  'csc')
            self.dphiydz = sps.hstack((zero, self.dphidz, zero),  'csc')
            self.dphizdx = sps.hstack((zero, zero, self.dphidx), 'csc')
            self.dphizdy = sps.hstack((zero, zero, self.dphidy), 'csc')
            self.dphizdz = sps.hstack((zero, zero, self.dphidz), 'csc')
            #print(phi.shape, P[:, 0].shape)
            self.pgx = N @ Px
            self.pgy = N @ Py
            self.pgz = N @ Pz


    def GetApproxElementSize(self, cam=None):
        if self.dim == 2:
            if cam is None:
                # in physical unit
                m2 = self.Copy()
                m2.GaussIntegration(npg=[1, 1])
                n = np.max(np.sqrt(m2.wdetJ))
            else:
                # in pyxel unit (int)
    
                m2 = self.Copy()
                m2.GaussIntegration(npg=[1, 1])
                n = int(np.floor(np.max(np.sqrt(m2.wdetJ))))
        
        if self.dim == 3:
            if cam is None:
                m2 = self.Copy()
                m2.GaussIntegration(npg=[1, 1, 1])
                n = int(np.floor(np.max(np.cbrt(m2.wdetJ))))
                
            else:
                m2 = self.Copy()
                m2.GaussIntegration(npg=[1, 1, 1])
                n = int(np.floor(np.max(np.cbrt(m2.wdetJ))))
                
        return n

    def DICIntegrationFast(self, n=10):
        self.DICIntegration(n)

    def DICIntegration(self, npt=10):
        """ DIC integration: build of the global differential operators """
        if hasattr(npt, 'rz'):
            # if n is a camera then n is autocomputed
            npt = self.GetApproxElementSize(npt)
        if type(npt) == int:
            npt = np.array([npt, npt], dtype=int)

        npt = np.maximum(self.degree + 1, npt)

        # Nombre de point d'intégration par élément
        nbg_xi = npt[0]
        nbg_eta = npt[1]

        """Espace de référence [-1,1]"""
        pxi = 1.0 / nbg_xi
        peta = 1.0 / nbg_eta
        Rect_xi = np.linspace(-1+pxi, 1-pxi, nbg_xi)
        Rect_eta = np.linspace(-1+peta, 1-peta, nbg_eta)
 
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
        #phi, dphidx, dphidy, detJ = self.ShapeFunctions(xi, eta)
        spline = self.spline
        phi = self.ReshapePhi(spline.DN([xi, eta], k=[0, 0]))
        self.npg = phi.shape[0]
        self.wdetJ = np.ones(self.npg)

        self.phi = phi
        
        ctrlPts = self.N2CtrlPts()
        Px = ctrlPts[0].ravel()
        Py = ctrlPts[1].ravel()
        
        self.pgx = self.phi @ self.n[:, 0] 
        self.pgy = self.phi @ self.n[:, 1]  


    def InverseBSplineMapping3D(self, x, y, z, init=None, elem=None, return_pg=False):
        """ 
        Inverse the BSpline mapping in order to map the coordinates
        of any physical points (x, y, z) to their corresponding position in
        the parametric space (xg, yg).

        Parameter : 
            m : bspline mesh from pyxel
            xpix : x coordinate in physical space
            ypix : y coordinate in physical space

        Return :
            xg, yg : x coordinate 
        """
        # Basis function
        spline = self.spline
        
        # Coordinates of controls points
        
        ctrlPts = self.N2CtrlPts()
        
        xn = ctrlPts[0].ravel()
        yn = ctrlPts[1].ravel()
        zn = ctrlPts[2].ravel()
        
        # Initializing  parametric integration points to zero
        if init is None :
            if elem is None:
                xi_g = 0 * x
                eta_g = 0 * y
                zeta_g = 0 * z
            
            else:
                xi_g = elem.xi[0] * np.ones_like(x)
                eta_g = elem.eta[0] * np.ones_like(y)
                zeta_g = elem.zeta[0] * np.ones_like(z)
            
        else :
            xi_g =  init[0]
            eta_g = init[1]
            zeta_g = init[2]
            
        maxiter = 7
        # Newton loop
        for k in range(maxiter):
            N = spline.DN(np.array([xi_g, eta_g, zeta_g]), k=[0,0,0])
            dNdxi = spline.DN(np.array([xi_g, eta_g, zeta_g]), k=[1,0,0]) 
            dNdeta = spline.DN(np.array([xi_g, eta_g, zeta_g]), k=[0,1,0]) 
            dNdzeta = spline.DN(np.array([xi_g, eta_g, zeta_g]), k=[0,0,1]) 
            
            # Projection into the physical space of evaluation point
            xp = N @ xn
            yp = N @ yn
            zp = N @ zn

            # Jacobian 
            J1 = dNdxi @ xn
            J4 = dNdxi @ yn
            J7 = dNdxi @ zn
            J2 = dNdeta @ xn
            J5 = dNdeta @ yn
            J8 = dNdeta @ zn
            J3 = dNdzeta @ xn
            J6 = dNdzeta @ yn
            J9 = dNdzeta @ zn
            

            
            del N, dNdxi, dNdeta, dNdzeta
       
            """
            J = np.array([[dxdxi, dxdeta, dxdzeta],
                          [dydxi, dydeta, dydzeta],
                          [dzdxi, dzdeta, dzdzeta]]).transpose(2, 0, 1)
            """


            detJ = J1*J5*J9 + J2*J6*J7 + J4*J8*J3 -\
                (J3*J5*J7 + J2*J4*J9 + J1*J6*J8)
            


            # Applying Sarrus Rules

            ComJ1 = J5*J9-J6*J8
            ComJ2 = -(J4*J9-J6*J7)
            ComJ3 = J4*J8-J5*J7
            ComJ4 = -(J2*J9-J3*J8)
            ComJ5 = J1*J9-J3*J7
            ComJ6 = -(J1*J8-J2*J7)
            ComJ7 = J2*J6-J3*J5
            ComJ8 = -(J1*J6-J3*J4)
            ComJ9 = J1*J5-J2*J4
            del J1,J2, J3, J4, J5, J6, J7, J8, J9
            # invJ = np.array([ComJ1/detJ, ComJ4/detJ, ComJ7/detJ,
            #                  ComJ2/detJ, ComJ5/detJ, ComJ8/detJ,
            #                  ComJ3/detJ, ComJ6/detJ, ComJ9/detJ]).T
            

            dxi_g = (ComJ1 * (x - xp) +\
                ComJ4 * (y - yp) +\
                ComJ7 * (z - zp)) / detJ

            deta_g = (ComJ2 * (x - xp) +\
                ComJ5 * (y - yp) +\
                ComJ8 * (z - zp)) / detJ

            dzeta_g = (ComJ3 * (x - xp) +\
                ComJ6 * (y - yp) +\
                ComJ9 * (z - zp)) / detJ
                
            xi_g = xi_g + dxi_g
            eta_g = eta_g + deta_g
            zeta_g = zeta_g + dzeta_g
            
            
            del dxi_g, deta_g, dzeta_g
            # res = np.linalg.norm(x - xp) + \
            #       np.linalg.norm(y - yp) + \
            #       np.linalg.norm(z - zp)
            
            
            res = max(np.linalg.norm(x - xp),
                  np.linalg.norm(y - yp),
                  np.linalg.norm(z - zp))

            if elem is None:
                
                if k < 1:
                    select = (xi_g == np.clip(xi_g, 0, 1)) &\
                        (eta_g == np.clip(eta_g, 0, 1)) &\
                        (zeta_g == np.clip(zeta_g, 0, 1))
                    
                    xi_g = xi_g[select]
                    eta_g = xi_g[select]
                    zeta_g = zeta_g[select]
                    
                    x = x[select]
                    y = y[select]
                    z = z[select]
                
                else :
                    xi_g = np.clip(xi_g, 0, 1)
                    eta_g = np.clip(eta_g, 0, 1)
                    zeta_g = np.clip(zeta_g, 0, 1)
                    
            else :
                
                if k < 1:
                    xi_g = np.clip(xi_g, elem.xi[0], elem.xi[1])
                    eta_g = np.clip(eta_g, elem.eta[0], elem.eta[1])
                    zeta_g = np.clip(zeta_g, elem.zeta[0], elem.zeta[1])
                else :
                    
                    select = (xi_g == np.clip(xi_g, elem.xi[0], elem.xi[1])) &\
                        (eta_g == np.clip(eta_g, elem.eta[0], elem.eta[1])) &\
                        (zeta_g == np.clip(zeta_g, elem.zeta[0], elem.zeta[1]))&\
                        (np.isfinite(xi_g))&\
                        (np.isfinite(eta_g))&\
                        (np.isfinite(zeta_g))
                    # select = (xi_g > 0) & (xi_g < 1) &\
                    #         (eta_g > 0) & (eta_g < 1) &\
                    #         (zeta_g > 0) & (zeta_g < 1) 
                    
                    xi_g = xi_g[select]
                    eta_g = eta_g[select]
                    zeta_g = zeta_g[select]
            
                    x = x[select]
                    y = y[select]
                    z = z[select]
            # print(f"Itération {k} | résidu : {res}")
            
            if res < 1.0e-3 or np.math.isnan(res):
                break            
        
        if return_pg:

            return xi_g, eta_g, zeta_g, x, y, z
        
        else:
            
            return xi_g, eta_g, zeta_g
        



    def DVCIntegrationPixelElem(self, f, cam, P=None, fname=None):
                   
                
        if P is None :
            ctrlPts = self.N2CtrlPts()
            Px = ctrlPts[0].ravel()
            Py = ctrlPts[1].ravel()
            Pz = ctrlPts[2].ravel()
            
        spline = self.spline
        
        for key in tqdm(self.elem):
            e = self.elem[key]
            print(f"\nÉlément {key} :")
            
            # Compute largest eedge of the element
            N_eval = self.compute_largest_edge(cam, e)
            
            # Inflating the number of eval point to ensure to have all pixels
            N_eval = int(N_eval)
            # print(f"N_eval = {N_eval ** 3}")
            
            # Setting the eval points to find all pxel center
            xi = np.linspace(e.xi[0], e.xi[1], N_eval)
            eta = np.linspace(e.eta[0], e.eta[1], N_eval)
            zeta = np.linspace(e.zeta[0], e.zeta[1], N_eval)
            
            # Going from parametric space to image space
            N = spline.DN([xi, eta, zeta], k=[0, 0, 0]) # @ self.C
            x, y, z = self.Mapping(xi, eta, zeta, type_eval='grid')
            u, v, w = cam.P(x, y, z)
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
            xi, eta, zeta, pgx, pgy, pgz = self.InverseBSplineMapping3D(xg, 
                                                                        yg, 
                                                                        zg, 
                                                                        init=init, 
                                                                        elem=e, 
                                                                        return_pg=True) 
            
            e.wdetJ = np.ones_like(xi)
            
            e.xig = xi
            e.etag = eta
            e.zetag = zeta
            
            e.pgx = pgx
            e.pgy = pgy
            e.pgz = pgz
        
                


    def DVCIntegrationPixel(self, f, cam, P=None, m2=None, fname=None):
        
        if fname is not None :
            file = './'        
            file_path = os.path.join(file, fname)
        else:
            file_path = './not_existing_file'
            
        if m2 is None :
            m2 = self
                
        if P is None :
                P = self.n
                
        if not os.path.exists(file_path):
                
            spline = self.spline
            
            # xi_glob = np.empty(0)
            # eta_glob = np.empty(0)
            # zeta_glob = np.empty(0)  
            for key in tqdm(m2.elem):
                e = m2.elem[key]
                print(f"\nÉlément {key} :")
                
                # Compute largest eedge of the element
                N_eval = m2.compute_largest_edge(cam, e)
                
                # Inflating the number of eval point to ensure to have all pixels
                N_eval = int(N_eval)
                # print(f"N_eval = {N_eval ** 3}")
                
                # Setting the eval points to find all pxel center
                xi = np.linspace(e.xi[0], e.xi[1], N_eval)
                eta = np.linspace(e.eta[0], e.eta[1], N_eval)
                zeta = np.linspace(e.zeta[0], e.zeta[1], N_eval)
                
                # Going from parametric space to physical space
                x, y, z = m2.Mapping(xi, eta, zeta, type_eval='grid')
                
                # Going from physical space to image space
                u, v, w = cam.P(x, y, z)
                
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
                # del xi, eta, zeta
                
                # Stacking the local evaluation matrix
                if key  == 0:
                    phi = phi_loc
                    
                else :
                    phi = sps.vstack((phi, phi_loc))
                
                #print(f"{xi.shape, eta.shape, zeta.shape}")
                # xi_glob = np.hstack((xi_glob, xi))
                # eta_glob = np.hstack((eta_glob, eta))
                # zeta_glob = np.hstack((zeta_glob, zeta))
                # break
                
            # phi = spline.DN(np.array([xi_glob, eta_glob, zeta_glob]), k=[0,0,0])
            del phi_loc
            self.phi = self.ReshapePhi(phi) # @ self.C
            
            # Saving phi matrix
            if fname is not None:
                sps.save_npz(file, phi)
            
            
            del phi
            self.npg = self.phi.shape[0]
            self.wdetJ = np.ones(self.phi.shape[0])
            
            
            # nbf = self.Get_nbf()
            # zero = sps.csr_matrix((self.npg, nbf))
            # print("Création des matrices phix, phiy, phiz")
            # self.phix = sps.hstack((self.phi, zero, zero),  'csc')
            # self.phiy = sps.hstack((zero, self.phi, zero),  'csc')
            # self.phiz = sps.hstack((zero, zero, self.phi),  'csc')
            
            self.pgx = self.phi @ P[:, 0]
            self.pgy = self.phi @ P[:, 1]
            self.pgz = self.phi @ P[:, 2]
        
        else:
            
            print("LOADING PHI FROM CURRENT FILE")
            self.phi = self.ReshapePhi(sps.load_npz(fname))
            
            self.npg = self.phi.shape[0]
            self.wdetJ = np.ones(self.phi.shape[0])
            
            self.pgx = self.phi @ P[:, 0]
            self.pgy = self.phi @ P[:, 1]
            self.pgz = self.phi @ P[:, 2]
        

            


    # def DVCIntegrationPixelPara(self, f, cam, P=None, m2=None):
        
        
    #     from paral_utils import _compute_phi_pixel
    #     self.phi = _compute_phi_pixel(self, f, cam, m2=m2)
    #     if self.phi is not None :
            
    #         P = self.n
    #         self.npg = self.phi.shape[0]
            
            
    #         self.wdetJ = np.ones(self.phi.shape[0])
            
    #         self.pgx = self.phi @ P[:, 0]
    #         self.pgy = self.phi @ P[:, 1]
    #         self.pgz = self.phi @ P[:, 2]  


    def DVCIntegration(self, n=None, P=None):
        #  DVC integration: build of the global differential operators
        if hasattr(n, 'rz'):
            # if n is a camera then n is autocomputed
            n = self.GetApproxElementSize(n)
        if type(n) == int:
            n *= 2 
            n = np.array([n, n, n], dtype=int)

        nbg_xi = n[0]
        nbg_eta = n[1]
        nbg_zeta = n[2]

        # Computing reference spaces and weights
        Rect_xi = np.linspace(-1, 1, nbg_xi)
        Rect_eta = np.linspace(-1, 1, nbg_eta)
        Rect_zeta = np.linspace(-1, 1, nbg_zeta)

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

        # Going from the reference element to the parametric space
        
        # Aranged gauss points in  xi direction
        xi = 0.5 * (xi_min + xi_max) + 0.5 * (xi_max - xi_min) * xi_g
        # Aranged gauss points in  eta direction
        eta = 0.5 * (eta_min + eta_max) + 0.5 * (eta_max - eta_min) * eta_g
        # Aranged gauss points in  zeta direction
        zeta = 0.5 * (zeta_min + zeta_max) + 0.5 * \
            (zeta_max - zeta_min) * zeta_g

        spline = self.spline
        
        N = spline.DN([xi, eta, zeta], k=[0,0,0])
        
        self.phi = self.ReshapePhi(N)
        self.npg = self.phi.shape[0]
        self.wdetJ = np.ones(self.phi.shape[0])
        # Integration weights + measures + Jacobian of the transformation
        #nbf = self.Get_nbf()
        #zero = sps.csr_matrix((self.npg, nbf))
        
        # self.phix = sps.hstack((self.phi, zero, zero),  'csc')
        # self.phiy = sps.hstack((zero, self.phi, zero),  'csc')
        # self.phiz = sps.hstack((zero, zero, self.phi),  'csc')
        
        if P is None:
            ctrlPts = self.N2CtrlPts()
            Px = ctrlPts[0].ravel()
            Py = ctrlPts[1].ravel()
            Pz = ctrlPts[2].ravel()
            
        else:
            Px = self.n[:, 0]
            Py = self.n[:, 1]
            Pz = self.n[:, 2]
            
            
        self.pgx = N @ Px
        self.pgy = N @ Py
        self.pgz = N @ Pz

    def ReshapePhi(self, phi):
        row, col, val = sps.find(phi)
        col = self.conn[self.e[0].ravel()[col], 0]
        phi = sps.csr_matrix((val, (row, col)),
                             shape=(phi.shape[0], self.ndof // self.dim)) 
        return phi


    def Mapping(self, xi, eta, zeta=None, type_eval='list', P=None):
        """
        Compute the mapping between parametric points (xi, eta, zeta) and 
        physical point (x, y, z)
        
        'type_eval' argument allow the function to perform 

        Parameters
        ----------
        xi : array numpy (Nxi,)
            Array containing the xi value of parametric points.
        eta : array numpy (Neta,)
            Array containing the eta value of parametric points.
        zeta : array numpy (Nzeta,), optional
            Array containing the zeta value of parametric points in 3D
            The default is None.
        type_eval : string, optional
            Define if the evaluation of parametric points are done by list or
            by grid. 
            If type_val is 'list' then we must have Nxi = Neta = Nzeta.
            If type_eval is 'grid' the evaluation willb e done by tensor product
            The default is 'list'.
        P: array numpy

        Returns
        -------
        array numpy
            (x, y, z) array containing the coordinate in physical space.
            If type_eval is 'list' x, y, z have the shape (Nxi,) 
            If type_eval is 'grid' x, y, z have the shape (Nxi * Neta * Nzeta,) 

        """
        
        spline = self.spline
        ctrlPts = self.N2CtrlPts()
        if self.dim == 2:
             
            XI = [xi, eta]
            
            if type_eval == 'list':
                XI = np.array(XI)
            
            if P is None:
                Px = ctrlPts[0].ravel()
                Py = ctrlPts[1].ravel()
            else:
                Px = P[:, 0]
                Py = P[:, 1]
            
            N = spline.DN(XI, k=[0, 0])

            x = N @ Px
            y = N @ Py
            
            return x, y
        
        elif self.dim == 3:
            
            XI = [xi, eta, zeta]
            
            if type_eval == 'list':
                XI = np.array(XI)
            
            
            N = spline.DN(XI, k=[0, 0, 0])
            
            if P is None:
                Px = ctrlPts[0].ravel()
                Py = ctrlPts[1].ravel()
                Pz = ctrlPts[2].ravel()
            else:
                Px = P[:, 0]
                Py = P[:, 1]
                Pz = P[:, 2]

            x = N @ Px
            y = N @ Py
            z = N @ Pz
            
            return x, y, z
            

    def ShapeFunctions(self, xi, eta, zeta=None, type_eval='list'):
        """
        

        Parameters
        ----------
        xi : array numpy
            DESCRIPTION.
        eta : array numpy
            DESCRIPTION.
        zeta : array numpy, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        N, dNdxi, dNdeta, dNdzeta
            Matrice 

        """

        spline = self.spline
        ctrlPts = self.N2CtrlPts()


        if self.dim == 2:
            
            XI = [xi, eta]
            if type_eval == 'list':
                XI = np.array(XI)
            
            Px = ctrlPts[0].ravel()
            Py = ctrlPts[1].ravel()

            N = spline.DN(XI, k=[0, 0])      
            dNdxi = spline.DN(XI, k=[1, 0])
            dNdeta = spline.DN(XI, k=[0, 1])


            dxdxi = dNdxi @ Px
            dxdeta = dNdeta @ Px 
            dydxi = dNdxi @ Py
            dydeta = dNdeta @ Py 
            
            detJ = dxdxi*dydeta - dydxi*dxdeta

            dNdx = sps.diags(dydeta/detJ).dot(dNdxi) + \
                sps.diags(-dydxi/detJ).dot(dNdeta)
            dNdy = sps.diags(-dxdeta/detJ).dot(dNdxi) + \
                sps.diags(dxdxi/detJ).dot(dNdeta)

            return N, dNdx, dNdy, detJ

        elif self.dim == 3:

            XI = [xi, eta, zeta]
            if type_eval == 'list':
                XI = np.array(XI)            

            Px = ctrlPts[0].ravel()
            Py = ctrlPts[1].ravel()
            Pz = ctrlPts[2].ravel()
            
            N = spline.DN(XI, k=[0, 0, 0]) 
            dNdxi = spline.DN(XI, k=[1, 0, 0])
            dNdeta = spline.DN(XI, k=[0, 1, 0])
            dNdzeta = spline.DN(XI, k=[0, 0, 1])


            dxdxi = dNdxi @ Px  #dphidxi.dot(P[:, 0])
            dxdeta = dNdeta @ Px #dphideta.dot(P[:, 0])
            dxdzeta = dNdzeta @ Px #dphidzeta.dot(P[:, 0])

            dydxi = dNdxi @ Py # dphidxi.dot(P[:, 1])
            dydeta = dNdeta @ Py #dphideta.dot(P[:, 1])
            dydzeta = dNdzeta @ Py #dphidzeta.dot(P[:, 1])

            dzdxi = dNdxi @ Pz #dphidxi.dot(P[:, 2])
            dzdeta = dNdeta @ Pz #dphideta.dot(P[:, 2])
            dzdzeta = dNdzeta @ Pz #dphidzeta.dot(P[:, 2])

            # Applying Sarrus Rules :
            """
            aei = dxdxi*dydeta*dzdzeta
            dhc = dxdeta*dydzeta*dzdxi
            bfg = dxdzeta*dydxi*dzdeta
            gec = dxdzeta*dydeta*dzdxi
            dbi = dxdeta*dydxi*dzdzeta
            ahf = dxdxi*dydzeta*dzdeta
            """
            aei = dxdxi*dydeta*dzdzeta
            dhc = dxdeta*dydzeta*dzdxi
            bfg = dxdzeta*dydxi*dzdzeta
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

            dNdx = sps.diags(ComJ_11/detJ).dot(dNdxi) + \
                sps.diags(ComJ_12/detJ).dot(dNdeta) + \
                sps.diags(ComJ_13/detJ).dot(dNdzeta)
            dNdy = sps.diags(ComJ_21/detJ).dot(dNdxi) + \
                sps.diags(ComJ_22/detJ).dot(dNdeta) + \
                sps.diags(ComJ_23/detJ).dot(dNdzeta)
            dNdz = sps.diags(ComJ_31/detJ).dot(dNdxi) + \
                sps.diags(ComJ_32/detJ).dot(dNdeta) + \
                sps.diags(ComJ_33/detJ).dot(dNdzeta)

            return N, dNdx, dNdy, dNdz, detJ

    def PlaneWave(self, T):
        V = np.zeros(self.ndof)
        V[self.conn[:, 0]] = np.cos(self.n[:, 1] / T * 2 * np.pi)
        return V
    
    def RemoveUnusedNodes(self):
        """
        Removes all the nodes that are not connected to a Patch and
        renumbers the element table. Both self.e and self.n are changed

        Usage :
            m.RemoveUnusedNodes()

        Returns
        -------
        None.

        """
        used_nodes = np.zeros(0, dtype=int)
        for ie in self.e.keys():
            used_nodes = np.hstack((used_nodes, self.e[ie].ravel()))
        used_nodes = np.unique(used_nodes)
        table = np.zeros(len(self.n), dtype=int)
        table[used_nodes] = np.arange(len(used_nodes))
        self.n = self.n[used_nodes, :]
        for ie in self.e.keys():
            self.e[ie] = table[self.e[ie]]
            
            
    def RemoveDoubleNodes(self, eps=None):
        """
        Removes the double nodes thus changes connectivity
        Warning: both self.e and self.n are modified!

        Usage :
            m.RemoveDoubleNodes()

        """

        if eps is None:
            eps = 1e-5 * self.GetApproxElementSize()
        scale = 10 ** np.floor(np.log10(eps))  # tolerance between two nodes
        nnew = np.round(self.n/scale) * scale
        _, ind, inv = np.unique(nnew, axis=0, return_index=True,
                                   return_inverse=True)
        self.n = self.n[ind]  # keep the initial precision of remaining nodes
        for k in self.e.keys():
            self.e[k] = inv[self.e[k]]
        
        # self.C = self.Connectivity_Matrix()
            
    def RemoveDoubleNodesKnots(self, eps=None):
        """
        Removes the double nodes thus changes connectivity
        Warning: both self.e and self.n are modified!

        Usage :
            m.RemoveDoubleNodes()

        """

        if eps is None:
            eps = 1e-5 * self.GetApproxElementSize()
        scale = 10 ** np.floor(np.log10(eps))  # tolerance between two nodes
        nnew = np.round(self.n_knots/scale) * scale
        _, ind, inv = np.unique(nnew, axis=0, return_index=True,
                                   return_inverse=True)
        self.n_knots = self.n_knots[ind]  # keep the initial precision of remaining nodes
        for k in self.e_knots.keys():
            self.e_knots[k] = inv[self.e_knots[k]]


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
        shape_axis = [shape[i] for i in range(len(shape)) if i != axis]
        XI_axis[axis] = np.zeros(1)
        pts_l = spline(ctrl_pts, tuple(XI_axis), [
                       0, 0, 0]).reshape([3] + shape_axis)
        XI_axis[axis] = np.ones(1)
        pts_r = spline(ctrl_pts, tuple(XI_axis), [
                       0, 0, 0]).reshape([3] + shape_axis)
        for pts in [pts_l, pts_r]:
            A = pts[:, :-1, :-1].reshape((3, -1)).T[:, None, :]
            B = pts[:, :-1, 1:].reshape((3, -1)).T[:, None, :]
            C = pts[:, 1:, :-1].reshape((3, -1)).T[:, None, :]
            D = pts[:, 1:, 1:].reshape((3, -1)).T[:, None, :]
            tri1 = np.concatenate((A, B, C), axis=1)
            tri2 = np.concatenate((D, C, B), axis=1)
            tri.append(np.concatenate((tri1, tri2), axis=0))
    tri = np.concatenate(tri, axis=0)
    data = np.empty(tri.shape[0], dtype=mesh.Mesh.dtype)
    data['vectors'] = tri
    m = mesh.Mesh(data, remove_empty_areas=remove_empty_areas)
    return m
