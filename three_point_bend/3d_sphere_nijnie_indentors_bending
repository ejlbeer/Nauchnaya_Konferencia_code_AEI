from dolfin import *
import numpy as np
L = 6.0
W = 1.0
H = 1.0
Nx, Ny, Nz = 80, 24, 24
mesh = BoxMesh(Point(0.0, -W/2.0, -H), Point(L,  W/2.0,  0.0), Nx, Ny, Nz)

x = mesh.coordinates()[:, 0]
y = mesh.coordinates()[:, 1]
z = mesh.coordinates()[:, 2]
xmid = L / 2.0
mesh.coordinates()[:, 0] = xmid + (x - xmid) * np.abs((x - xmid)/xmid)**0.5
mesh.coordinates()[:, 1] = y * np.abs(y/(W/2.0))**0.5
mesh.coordinates()[:, 2] = -H + (z + H)**1.5 / (H**0.5)

facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1,0)

class TopContact(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 0.0)
class LeftSupport(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[2], -H) and abs(x[0] - 1.0) <= 0.15)
class RightSupport(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[2], -H) and abs(x[0] - (L - 1.0)) <= 0.15)

TopContact().mark(facets, 1)
LeftSupport().mark(facets, 2)
RightSupport().mark(facets, 3)

ds = Measure("ds", domain=mesh, subdomain_data=facets)

E = Constant(10.0)
nu = Constant(0.30)

mu = E / (2.0 * (1.0 + nu))

lmbda = (E * nu ((1.0 + nu) * (1.0 - 2.0 * nu)))

def eps(u):
    return sym(grad(u))

def sigma(u):
    return (lmbda * tr(eps(u)) * Identity(3) + 2.0 * mu * eps(u))

def ppos(x):
    return (x + abs(x)) / 2.0

V  = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)

u  = Function(V, name="Displacement")

du = TrialFunction(V)
v  = TestFunction(V)

gap = Function(V2, name="Gap")
p   = Function(V0, name="ContactPressure")

R_top = 1.5
x0 = L / 2.0
y0 = 0.0
top_obstacle = Expression("-d + (pow(x[0] - x0, 2) + pow(x[1] - y0, 2))/(2*R)", d=0.0, x0=x0, y0=y0, R=R_top,degree=2)

R_support = 1.5
margin = 0.25
xs_left  = margin + 0.8
xs_right = L - (margin + 0.8)
ys_sup = 0.0
R_support = 0.8

left_support = Expression("-(pow(x[0] - xs, 2) + pow(x[1] - ys, 2)) / (2.0*R)", xs=xs_left, ys=ys_sup, R=R_support, degree=2)

right_support = Expression("-(pow(x[0] - xs, 2) + pow(x[1] - ys, 2)) / (2.0*R)", xs=xs_right, ys=ys_sup, R=R_support, degree=2)

class PinPoint(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 0.0) and near(x[1], 0.0) and near(x[2], -H))

pin = PinPoint()

bc_pin_x = DirichletBC(V.sub(0), Constant(0.0), pin,method="pointwise")

bc_pin_y = DirichletBC(V.sub(1), Constant(0.0), pin,method="pointwise")

bc = [bc_pin_x, bc_pin_y]

pen = Constant(5e3)

F = inner(sigma(u), eps(v))*dx

F += (pen * v[2] * ppos(u[2] - top_obstacle) * ds(1))
F += (- pen * v[2] * ppos(left_support - u[2]) * ds(2))
F += (- pen * v[2] * ppos(right_support - u[2]) * ds(3))

J = derivative(F, u, du)

problem = NonlinearVariationalProblem(F, u, bc, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters["newton_solver"]

prm["relative_tolerance"] = 1e-6
prm["absolute_tolerance"] = 1e-8
prm["maximum_iterations"] = 40

prm["linear_solver"] = "gmres"
prm["preconditioner"] = "hypre_amg"

prm["relaxation_parameter"] = 0.7

file = XDMFFile("three_point_bending_spheres_3D.xdmf")

file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True

dmax = 0.4
nsteps = 4

for i, dval in enumerate(np.linspace(0.0, dmax, nsteps)):
    print(f"\nStep {i+1}/{nsteps}")
    print(f"d = {dval:.5f}")
    top_obstacle.d = dval

    try:
        solver.solve()
    except RuntimeError:
        print("Newton failed.")
        break
    p.assign(project(-sigma(u)[2,2], V0))
    gap.assign(project(top_obstacle - u[2], V2))
    file.write(u, dval)
    file.write(gap, dval)
    file.write(p, dval)
