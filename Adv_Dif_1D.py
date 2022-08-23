''' This script solves the 1-dimensional temporal-spatial PDE
describing the Advection-Diffusion. The Velocity is constant in
all of the time steps '''


from fenics import *
import matplotlib.pyplot as plt

T = 2.
num_steps = 50
dt = T / num_steps

vel = Constant(1.)
D = Constant(0.01)

nx = 100
mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, 'P', 1)


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

F = u*v*dx + dt*D*u.dx(0)*v.dx(0)*dx + dt*vel*u.dx(0)*v*dx - (u_n + dt*Constant(0.))*v*dx
a, L = lhs(F), rhs(F)


u = Function(V)
t = 0

for n in range(num_steps):
	t += dt
	solve(a == L, u, bcs)

	u_n.assign(u)
	plot(u,'b+')
	plt.show()






