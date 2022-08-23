''' This script solves the 1-dimensional temporal-spatial PDE
describing the Advection-Diffusion. The Velocity is pulsatile:
a half-period sin function during systole and zero during diastole '''

from fenics import *
import math
import numpy as np
import matplotlib.pyplot as plt

T = 1.
num_steps = 50
dt = T / num_steps
systole_pr = int(0.3*num_steps)

w_systole = []
for i in range(systole_pr):
	w_systole.append(math.sin(math.pi*i/systole_pr))
w_diastole = np.zeros(num_steps - systole_pr)
w_systole = np.array(w_systole)
timeseries_ = np.concatenate((w_systole, w_diastole))
timeseries_w = timeseries_.tolist()

D = Constant(0.01)
nx = 100
mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, 'P', 1)
W = VectorFunctionSpace(mesh, 'P', 2)

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

u_n = interpolate(u_L, V)
u_n = interpolate(u_R, V)

u = TrialFunction(V)
v = TestFunction(V)
w = Function(W)

F = u*v*dx + dt*D*u.dx(0)*v.dx(0)*dx + dt*dot(w,grad(u))*v*dx - (u_n + dt*Constant(0.))*v*dx
a, L = lhs(F), rhs(F)


u = Function(V)
t = 0

for n in range(num_steps):
	t += dt
	w.vector()[:] = timeseries_w[n]
	solve(a == L, u, bcs)
	u_n.assign(u)
	plot(u,'b+')
	plt.show()






