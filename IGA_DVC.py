import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import pickle
import bsplyne as bs
import matplotlib.pyplot as plt

"""
spline, ctrl_pts = pickle.load(open("/home-local/fidelin/Téléchargements/pickle_magma", "rb"))
degrees = spline.getDegrees()
knots = spline.getKnots()
m = BSplinePatch(ctrl_pts, degrees, knots)
m.Plot()


f = px.Volume("Binary_Image.tiff").Load()
f.Plot()
cam = px.CameraVol([47.5, 4., 3.31578947, 3.31578947, 0., 0., 0.])

px.PlotMeshImage3d(f, m, cam)
spline.Plot()
"""

# %% Test integration

center = np.array([0, 0, 0])
orientation = np.array([0, 0, 1])

cube, ctrlPts = bs.new_cube(center, orientation, 1)

VecDegrees = cube.getDegrees()
VecKnots = cube.getKnots()

m = BSplinePatch(ctrlPts, VecDegrees, VecKnots)
#m.Plot()

xi = np.linspace(0, 1, 5)
eta = np.linspace(0, 1, 5)
zeta = np.linspace(0, 1, 5)

#phi2D, dphidx2D, dphidy2D, detJ = m.ShapeFunctionsAtGridPoints(xi, eta)
phi, dphidx, dphidy, dphidz, detJ = m.ShapeFunctionsAtGridPoints(xi, eta, zeta)


test = dphidx.A[:,0].reshape((5,5,5))
#print(test[0, :, 0])