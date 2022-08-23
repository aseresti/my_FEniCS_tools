''' This script solves the 1-dimensional only spatial PDE
describing the Advection-Diffusion. The BCs and Diffusivity and 
Velocity values are defined based on [Arzani, A., et al. (2021)]  '''



from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import math

D = Constant(0.01)
vel = Constant(1)

nx = 10000
mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)

u_L = Expression('1', degree = 0)
tol = 1E-14

def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, tol)

bc_L = DirichletBC(V, u_L, boundary_L)

u_R = Expression('0.1', degree = 0)

def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, tol)

bc_R = DirichletBC(V, u_R, boundary_R)

bcs = [bc_L, bc_R]

u = TrialFunction(V)
v = TestFunction(V)


a = D*u.dx(0)*v.dx(0)*dx + vel*u.dx(0)*v*dx
L = Constant(0.0)*v*dx
uh = Function(V)

solve(a == L, uh, bcs)

plot(uh,'b+')
plt.show()

