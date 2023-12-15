import numpy as np
from bspline_patch_bsplyne import BSplinePatch
import pyxel as px
import pickle

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
    