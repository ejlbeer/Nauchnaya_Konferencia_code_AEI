from dolfin import *
import numpy as np

L = 4.0
W = 1.0
H = 1.0
Nx, Ny, Nz = 80, 20, 20
mesh = BoxMesh(Point(0.0, -W/2.0, -H), Point(L,  W/2.0, 0.0), Nx, Ny, Nz)
x = mesh.coordinates()[:, 0]
y = mesh.coordinates()[:, 1]
z = mesh.coordinates()[:, 2]
xmid = L/2.0
mesh.coordinates()[:, 0] = xmid + (x - xmid)*np.abs((x - xmid)/xmid)**0.5
mesh.coordinates()[:, 1] = y*np.abs(y/(W/2.0))**0.5
mesh.coordinates()[:, 2] = -H + (z + H)**1.5/(H**0.5)
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)

class TopIndentor(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 0.0)
class LeftBoundSupport(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[2], -H) and abs(x[0] - 1.0) <= 0.15)
class RightBoundSupport(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[2], -H) and abs(x[0] - (L-1.0)) <= 0.15)

TopIndentor().mark(facets, 1)
LeftBoundSupport().mark(facets, 2)
RightBoundSupport().mark(facets, 3)
ds = Measure("ds", domain=mesh, subdomain_data=facets)

V = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)
u = Function(V, name="Displacement")
du = TrialFunction(V)
v = TestFunction(V)
gap = Function(V2, name="Gap")
p = Function(V0, name="ContactPressure")

E = Constant(10.0)
nu = Constant(0.3)

mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

def eps(u):
    return sym(grad(u))

def sigma(u):
return (lmbda*tr(eps(u))*Identity(3) + 2.0*mu*eps(u))

def ppos(x):
    return (x + abs(x))/2.0

bc_left_z = DirichletBC(V.sub(2), Constant(0.0), facets, 2)
bc_right_z = DirichletBC(V.sub(2), Constant(0.0), facets, 3)
bc_fix_x = DirichletBC(V.sub(0), Constant(0.0), facets, 2)
bc = [bc_left_z, bc_right_z, bc_fix_x]
R = 1.0
x0 = L/2.0
y0 = 0.0
d = 0.0
obstacle = Expression("-d + (pow(x[0]-x0,2) + pow(x[1]-y0,2))/(2.0*R)", d=d, x0=x0, y0=y0, R=R, degree=2)

pen = Constant(1e4)
F = inner(sigma(u), eps(v))*dx + pen * v[2] * ppos(u[2] - obstacle) * ds(1)
J = derivative(F, u, du)
problem = NonlinearVariationalProblem(F, u, bc, J)

solver = NonlinearVariationalSolver(problem)

param = solver.parameters["newton_solver"]

param["relative_tolerance"] = 1e-4
param["absolute_tolerance"] = 1e-5
param["maximum_iterations"] = 40

param["linear_solver"] = "cg"
param["preconditioner"] = "ilu"

file = XDMFFile("three_point_bending_3D_true.xdmf")

file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True

dmax = 0.4
nsteps = 5

for i, dval in enumerate(np.linspace(0.0, dmax, nsteps)):
    print(f"Step {i+1}/{nsteps}, d = {dval:.4f}")
    obstacle.d = dval
    solver.solve()
    p.assign(project(-sigma(u)[2,2], V0))
    gap.assign(project(obstacle - u[2], V2))
    file.write(u, dval)
    file.write(gap, dval)
    file.write(p, dval)
