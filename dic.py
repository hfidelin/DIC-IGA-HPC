# %% import
import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import bsplyne as bs
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon, LineString, Point
import time

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()
cam = px.Camera([100, 6.95, -5.35, 0])

# %% Set up bspline
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
m.KnotInsertion([newt, newr])


# %% Convex Hull
hull = ConvexHull(m.n)


scale_percentage = 5
center = np.mean(hull.points, axis=0)
new_points = center + (1 + scale_percentage / 100) * (hull.points - center)
new_hull = ConvexHull(new_points)


px.PlotMeshImage(f, m, cam)
for simplex in new_hull.simplices:
    u, v = cam.P(new_points[simplex, 0], new_points[simplex, 1])
    plt.plot(v, u, 'r-')                     
                      
                      

for simplex in hull.simplices:
    u, v = cam.P(m.n[simplex, 0], m.n[simplex, 1])
    plt.plot(v, u, 'g-')
    

# %% Box

eps = 1
def bounding_box(cam, points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)
    
    
    u1, v1 = cam.P(bot_left_x, bot_left_y)
    u2, v2 = cam.P(top_right_x, top_right_y)
    
    return u1, v1, u2, v2


u1, v1, u2, v2 = bounding_box(cam, m.n)

width = v2 - v1 
height = u2 - u1


u1 += eps
v1 -= eps

width += 2*eps
height -= 2*eps

u1 = int(u1); v1 = int(v1); u2 = int(u2); v2 = int(u2)
width = int(width); height = int(height)

rect = Rectangle((v1, u1), width, height, linewidth=1, edgecolor='r', facecolor='none')


fig, ax = plt.subplots()
f.Plot()
u, v = cam.P(m.n[:, 0], m.n[:, 1])
m.Plot(n=np.c_[v, u], edgecolor="y", alpha=0.6)
new_rect = rect
ax.add_patch(new_rect)

X = np.arange(v1+1, v1+width)
Y = np.arange(u1-width+1, u1)


XX, YY = np.meshgrid(X, Y)
center_pix = np.column_stack((XX.ravel(), YY.ravel()))
#plt.scatter(center_pix[:, 0], center_pix[:, 1], label='pixel center')

nbg_xi = 100
nbg_eta = 100
pxi = 1.0 / nbg_xi
peta = 1.0 / nbg_eta
xi_g = np.linspace(pxi, 1-pxi, nbg_xi)
eta_g = np.linspace(peta, 1-peta, nbg_eta)
phi, _, _, detJ = m.ShapeFunctions(xi_g, eta_g)
P = m.Get_P()
pgx = phi @ P[:, 0] 
pgy = phi @ P[:, 1]
u1, v1 = cam.P(pgx, pgy)
px.PlotMeshImage(f, m, cam)
plt.scatter(v1, u1, c='g', label='integration point')
plt.legend()

# %% Je tente un truc
neval = [30, 30]
Pxm = m.n[:, 0] 
Pym = m.n[:, 1] 
xi = np.linspace(
    m.knotVect[0][m.degree[0]], m.knotVect[0][-m.degree[0]], neval[0])
eta = np.linspace(
    m.knotVect[1][m.degree[1]], m.knotVect[1][-m.degree[1]], neval[1])

# Iso parameters for the elemnts
xiu = np.unique(m.knotVect[0])
etau = np.unique(m.knotVect[1])

# Basis functions
spline = bs.BSpline(m.degree, m.knotVect)

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


plt.scatter(xe1[0, :], ye1[0, :], color='r')
plt.scatter(xe1[-1, :], ye1[-1, :], color='r')
plt.scatter(xe2[:, 0], ye2[:, 0], color='r') 
plt.scatter(xe2[:, -1], ye2[:, -1], color='r') 


xi_first = []
xi_last = []
for i in range(xe1.shape[1]):
    xi_first.append([xe1[0, i], ye1[0, i]])
    xi_last.append([xe1[-1, i], ye1[-1, i]])


eta_first = []
eta_last = []
for i in range(xe2.shape[0]):
    eta_first.append([xe2[i, 0], ye2[i, 0]])
    eta_last.append([xe2[i, -1], ye2[i, -1]])
    
coord = np.vstack((xi_first, eta_last, xi_last[::-1], eta_first[::-1]))

polygon = Polygon(coord)
plt.clf()
x, y = polygon.exterior.xy
plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.fill(x, y, alpha=0.2, color='green')  

"""    

xi_first = np.array(xi_first)
xi_last = np.array(xi_last)
eta_first = np.array(eta_first)
eta_last = np.array(eta_last)


plt.scatter(xi_first[:, 0], xi_first[:, 1], c='k')
plt.scatter(eta_first[:, 0], eta_first[:, 1], c='k')
plt.scatter(xi_last[:, 0], xi_last[:, 1], c='k')
plt.scatter(eta_last[:, 0], eta_last[:, 1], c='k')


u1 = np.array(u1)
u2 = np.array(u2)
v1 = np.array(v1)
v2 = np.array(v2)

point_x = np.vstack((u1, v1))
point_y = np.vstack((u2, v2))
"""



# %% boucle

pix_close = []
for i in range(len(X)):
    for j in range(len(Y)):
        print((i,j))
        pix = np.array([X[i], Y[j]])
        dist = np.min(cdist(np.atleast_2d(pix), np.array([v, u]).T))
        if dist > 1:
            continue
        else :
            pix_close.append(pix)

pix_close = np.array(pix_close)


rect = Rectangle((v1, u1), width, height, linewidth=1, edgecolor='r', facecolor='none')


fig, ax = plt.subplots()
f.Plot()
u_p, v_p = cam.P(m.n[:, 0], m.n[:, 1])
m.Plot(n=np.c_[v_p, u_p], edgecolor="y", alpha=0.6)
new_rect = rect
ax.add_patch(new_rect)

X = np.arange(v1+1, v1+width)
Y = np.arange(u1-width+1, u1)
plt.scatter(pix_close[:, 0], pix_close[:, 1], c='purple', label='inside pixel')
plt.scatter(v, u, c='g', label='integration point')
plt.legend()

# %% Inverse Mapping


xg = 0 * pix_close[:, 0]
yg = 0 * pix_close[:, 1]
xn, yn = cam.P(m.n[:, 0], m.n[:, 1])
res = 1

for k in range(7):
    print(f"Itération numéro : {k}")
    phi = m.spline.DN([xg, yg], k=[0, 0])
    N_r = m.spline.DN([xg, yg], k=[1, 0])  
    N_s = m.spline.DN([xg, yg], k=[0, 1])
    print(phi.shape)
    dxdr = np.dot(N_r, xn)
    dydr = np.dot(N_r, yn)
    dxds = np.dot(N_s, xn)
    dyds = np.dot(N_s, yn)
    detJ = dxdr * dyds - dydr * dxds
    invJ = np.array([dyds / detJ, -dxds / detJ,
                     -dydr / detJ, dxdr / detJ]).T
    xp = np.dot(phi, xn)
    yp = np.dot(phi, yn)
    dxg = invJ[:, 0] * (pix_close[:, 0] - xp) + invJ[:, 1] * (pix_close[:, 1] - yp)
    dyg = invJ[:, 2] * (pix_close[:, 0] - xp) + invJ[:, 3] * (pix_close[:, 1] - yp)
    res = np.dot(dxg, dxg) + np.dot(dyg, dyg)
    xg = xg + dxg
    yg = yg + dyg
    if res < 1.0e-6:
        break

# %% Integration

m.Connectivity()
m.DIntegration()

u,v = cam.P(m.pgx, m.pgy)
plt.plot(u, v, 'k.')
# %% Correlate

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res = px.Correlate(f, g, m, cam, U0=U)


m.Plot(U, alpha=0.5)
m.Plot(U=3*U)

px.PlotMeshImage(g, m, cam, U)
