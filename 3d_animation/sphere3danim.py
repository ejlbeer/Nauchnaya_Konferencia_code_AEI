from dolfin import *
import numpy as np
Nx, Ny, Nz = 30, 30, 15
mesh = UnitCubeMesh(Nx, Ny, Nz)
mesh.coordinates()[:, 0] = mesh.coordinates()[:, 0]**2
mesh.coordinates()[:, 1] = mesh.coordinates()[:, 1]**2
mesh.coordinates()[:, 2] = -mesh.coordinates()[:, 2]**2
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 0.0) and on_boundary
def symmetry_x(x, on_boundary):
    return near(x[0], 0.0) and on_boundary
def symmetry_y(x, on_boundary):
    return near(x[1], 0.0) and on_boundary
def bottom(x, on_boundary):
    return near(x[2], -1.0) and on_boundary

facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
Top().mark(facets, 1)
ds = Measure("ds", subdomain_data=facets)
R = 0.5
dmax = R
nsteps = 10

obstacle = Expression(
    "-d + (pow(x[0],2) + pow(x[1],2))/(2*R)",
    d=0.0, R=R, degree=2
)

V = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)

u = Function(V, name="Displacement")
du = TrialFunction(V)
v = TestFunction(V)
gap = Function(V2, name="Gap")
p = Function(V0, name="Contact pressure")
bc = [
    DirichletBC(V, Constant((0.0, 0.0, 0.0)), bottom),
    DirichletBC(V.sub(0), Constant(0.0), symmetry_x),
    DirichletBC(V.sub(1), Constant(0.0), symmetry_y)
]
E = Constant(10.0)
nu = Constant(0.3)
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
def eps(u):
    return sym(grad(u))
def sigma(u):
    return lmbda*tr(eps(u))*Identity(3) + 2*mu*eps(u)
def ppos(x):
    return (x + abs(x)) / 2.0

pen = Constant(1e3)
form = inner(sigma(u), eps(v))*dx \
     + pen * v[2] * ppos(u[2] - obstacle) * ds(1)
J = derivative(form, u, du)
problem = NonlinearVariationalProblem(form, u, bc, J)
solver = NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6
solver.parameters["newton_solver"]["maximum_iterations"] = 50
solver.parameters["newton_solver"]["linear_solver"] = "gmres"
solver.parameters["newton_solver"]["preconditioner"] = "hypre_amg"
solver.parameters["newton_solver"]["relaxation_parameter"] = 0.5
file = XDMFFile("contact_3D_animation.xdmf")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True
for i, dval in enumerate(np.linspace(0, dmax, nsteps)):
    print(f"Step {i+1}/{nsteps}, d = {dval:.4f}")
    obstacle.d = dval
    try:
        solver.solve()
    except RuntimeError:
        print("Reducing step...")
        obstacle.d = 0.5 * (prev_d + dval)
        solver.solve()
    p.assign(-project(sigma(u)[2,2], V0))
    gap.assign(project(obstacle - u[2], V2))
    file.write(u, dval)
    file.write(gap, dval)
    file.write(p, dval)
