# %% import
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import matplotlib.pyplot as plt
import time

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()
cam = px.Camera([100, 6.95, -5.35, 0])
cam = px.Camera([1, 0, 0, 0])
# %% Set up bspline
a = 0.925
b = 0.79
Xi = np.array([[0.5, b, 2*b],
               [0.5*a, b*a, 2*b*a],
               [0, 0, 0]])
Yi = np.array([[0, 0, 0],
               [0.5*a, b*a, 2*b*a],
               [0.5, b, 2*b]])

ctrlPts = np.array([Xi, Yi])
degree = [2, 2]
kv = np.array([0, 0, 0, 1, 1, 1])
knotVect = [kv, kv]

n = 5
newr = np.linspace(0, 1, n+2)[1:-1]

n = 10
newt = np.linspace(0, 1, n+2)[1:-1]
m = BSplinePatch(ctrlPts, degree, knotVect)
m.KnotInsertion([newt, newr])


# e = m.Init_elem_2D()
# %%

nbg_xi = m.degree[0]+1
nbg_eta = m.degree[1]+1
#nbg_xi = m.degree[0]+1
GaussLegendre = np.polynomial.legendre.leggauss
xiu = np.unique(m.knotVect[0])
etau = np.unique(m.knotVect[1])


Gauss_xi = GaussLegendre(nbg_xi)
Gauss_eta = GaussLegendre(nbg_eta)

nbf = m.Get_nbf()

e_xi = np.unique(m.knotVect[0])
ne_xi = e_xi.shape[0]-1
e_eta = np.unique(m.knotVect[1])
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

P = m.Get_P()

""" Spatial derivatives """

N, _, _, detJ = m.ShapeFunctions(xi, eta, P=P)

x = N @ P[:, 0]
y = N @ P[:, 1]

dudx, dudy, dvdx, dvdy = cam.dPdX(x, y)
detJP = dudx * dvdy - dvdx * dudy
w = np.kron(wg_eta, wg_xi)

A = sum(w*np.abs(detJP)*np.abs(detJ)*mes_xi*mes_eta/4)


# %% loop way

nbg_xi = 1
nbg_eta = 1
GaussLegendre = np.polynomial.legendre.leggauss
Gauss_xi = GaussLegendre(nbg_xi)
Gauss_eta = GaussLegendre(nbg_eta)

xi_g = Gauss_xi[0]
wg_xi = Gauss_xi[1]
eta_g = Gauss_eta[0]
wg_eta = Gauss_eta[1]
P = m.Get_P()
# px.PlotMeshImage(f, m, cam)


def compute_elem_area(m, cam, elem):
    """
    Compute element area in pixel using a gauss quadrature

    Parameters
    ----------
    m : bspline mesh
        DESCRIPTION.
    cam : camera model
        DESCRIPTION.
    elem : element 2D
        DESCRIPTION.

    Returns
    -------
    float.
    """

    # Initialiazing Gauss Quadrature
    nbg_xi = 1
    nbg_eta = 1
    GaussLegendre = np.polynomial.legendre.leggauss
    Gauss_xi = GaussLegendre(nbg_xi)
    Gauss_eta = GaussLegendre(nbg_eta)

    xi_g = Gauss_xi[0]
    wg_xi = Gauss_xi[1]
    eta_g = Gauss_eta[0]
    wg_eta = Gauss_eta[1]

    # Going from reference space to parametric space
    xi = elem.xi[0] + 0.5 * (xi_g + 1) * elem.mes_xi
    eta = elem.eta[0] + 0.5 * (eta_g + 1) * elem.mes_eta
    detJGauss = elem.mes_xi * elem.mes_eta / 4

    # Get control  points
    P = m.Get_P()

    # Evaluating basis function on evaluation points
    phi, _, _, detJB = m.ShapeFunctions(xi, eta)

    # Going from parametric space to physical space
    x = phi @ P[:, 0]
    y = phi @ P[:, 1]

    # Computing determinant of the camera model
    dudx, dudy, dvdx, dvdy = cam.dPdX(x, y)
    detJP = dudx * dvdy - dvdx * dudy

    # Computing the area with gauss quadrature
    w = np.kron(wg_xi, wg_eta)
    area = np.sum(w * np.abs(detJB) * np.abs(detJP) * detJGauss)

    return area


XI = []
ETA = []
eta_elem = np.array([0, 0])
A = 0
# Loop on patch's element
for key in m.e:

    elem = m.e[key]
    # print(f"\nElement {elem.num} :\n")

    xi = elem.xi
    eta = elem.eta

    # Get element area
    area = compute_elem_area(m, cam, elem)
    A += area
    # Inflate the area to ensure to have enough evaluation point
    # area *= 1.2

    if not np.allclose(eta, eta_elem):
        N_eta = int(np.floor(np.sqrt(area)))
        N_xi = N_eta
        eta_elem = eta
        xi_g = np.linspace(xi[0], xi[1], N_xi)
        eta_g = np.linspace(eta[0], eta[1], N_eta)
        XI += xi_g.tolist()
        ETA += eta_g.tolist()

    else:
        
        N_xi = int(np.floor(area / N_eta))
        xi_g = np.linspace(xi[0], xi[1], N_xi)
        XI += xi_g.tolist()


"""

XI = np.unique(np.array(XI))
ETA = np.unique(np.array(ETA))
spline = m.Get_spline()
N = spline.DN([XI, ETA], k=[0,0])

x = N @ P[:, 0]
y = N @ P[:, 1]

u, v = cam.P(x, y)

px.PlotMeshImage(f, m, cam)
plt.scatter(v, u)
"""

print(A)
# %% Inverse by elem

P = m.Get_P()
spline = m.Get_spline()
px.PlotMeshImage(f, m, cam)
for key in m.e:
    
    elem = m.e[key]
    print(f"\n ELEMENT : {elem.num}\n")
    xi = elem.xi
    eta = elem.eta
    
    area = compute_elem_area(m, cam, elem)
    area *= 4
    
    Neval = int(np.floor(np.sqrt(area)))
    print(f"NEVAL : {Neval}")
    
    xi_eval = np.linspace(xi[0], xi[1], Neval)
    eta_eval = np.linspace(eta[0], eta[1], Neval)
    
    N, _, _, _ = m.ShapeFunctions(xi_eval, eta_eval)
    
    x = N @ P[:, 0]
    y = N @ P[:, 1]
    
    u, v = cam.P(x, y)
    
    u = np.round(u).astype('uint16')
    v = np.round(v).astype('uint16')
    # Getting rid of the duplicate points
    shape_max = max(f.pix.shape)
    rep = np.where(u<shape_max)[0]
    u = u[rep]
    v = v[rep]
    
    
    pix = f.pix * 0
    pix[u, v] = 1
    pixel = np.where(pix)
    # plt.scatter(pixel[1].astype(float), pixel[0].astype(float))
    
    # Going from image space to physical space by inversing camera model
    xg, yg = cam.PinvNL(pixel[0].astype(float), 
                      pixel[1].astype(float))
    

    xi_g, eta_g = m.InverseBSplineMapping(xg, yg, elem=elem)
    # print(xi_g)
    
    select = (xi_g > xi[0]) & (xi_g < xi[1]) &\
             (eta_g > eta[0]) & (eta_g < eta[1])
    
    xi_g = xi_g[select]
    eta_g = eta_g[select]
    
    phi = spline.DN(np.array([xi_g, eta_g]))
    
    xt = phi @ P[:, 0]
    yt = phi @ P[:, 1]
    
    ut, vt = cam.P(xt, yt)
    
    plt.scatter(vt, ut)
    
    
# %%
m.Connectivity()
m.DICIntegrationPixel(m, cam)

u, v = cam.P(m.pgx, m.pgy)
px.PlotMeshImage(f, m, cam)
plt.scatter(v, u)


U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res = px.Correlate(f, g, m, cam, U0=U)

m.Plot(U, alpha=0.5)
m.Plot(U=3*U)
