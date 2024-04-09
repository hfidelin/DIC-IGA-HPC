#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)


def InverseBSplineMapping3D(self, x, y, z, init=None, elem=None):
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
    spline = self.Get_spline()
    # P = self.Get_P()
    # Coordinates of controls points
    xn = self.n[:, 0]
    yn = self.n[:, 1]
    zn = self.n[:, 2]
    
    nan = float('nan')
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
        N_xi = spline.DN(np.array([xi_g, eta_g, zeta_g]), k=[1,0,0])
        N_eta = spline.DN(np.array([xi_g, eta_g, zeta_g]), k=[0,1,0])
        N_zeta = spline.DN(np.array([xi_g, eta_g, zeta_g]), k=[0,0,1])
        
        # Projection into the physical space of evaluation point
        xp = N @ xn
        yp = N @ yn
        zp = N @ zn

        # Jacobian 
        J1 = N_xi @ xn
        J4 = N_xi @ yn
        J7 = N_xi @ zn
        J2 = N_eta @ xn
        J5 = N_eta @ yn
        J8 = N_eta @ zn
        J3 = N_zeta @ xn
        J6 = N_zeta @ yn
        J9 = N_zeta @ zn
        

        
        del N, N_xi, N_eta, N_zeta
   
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
                    (zeta_g == np.clip(zeta_g, elem.zeta[0], elem.zeta[1])) 
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

    return xi_g, eta_g, zeta_g