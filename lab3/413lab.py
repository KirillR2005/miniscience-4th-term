# The modules that will be used are imported:

#This code computes the heat equation solution with initial temperature distribution T = T_0/(1 + (r - r_0)^2)

import importlib.util

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py.PETSc import ScalarType  # type: ignore
else:
    print("This demo requires petsc4py.")
    exit(0)

from mpi4py import MPI
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Define temporal parameters
t_0 = 0 
t_end = 0.6 
num_steps = 100
dt = t_end/num_steps  

# Heat conductivity
kappa = 1

msh = mesh.create_box(MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 5.0, 2.0)), (20, 50, 20), cell_type=mesh.CellType.tetrahedron)
V = fem.functionspace(msh, ("Lagrange", 1))

def initial_condition(x, a = 1, b = 1, c = 1):
    return 1/(1 + ((x[0]-1.0)**2 + (x[1]-1.0)**2 + (x[2]-1.0)**2))

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Create boundary condition
fdim = msh.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

#xdmf file for visualization
xdmf = io.XDMFFile(msh.comm, "result/heat_diffusion4.xdmf", "w")
xdmf.write_mesh(msh)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t_0)

# Next, the variational problem is defined:

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = fem.Constant(msh, PETSc.ScalarType(0.1))
a = kappa * u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

# The variational problem is assembled into a linear system using
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

#integrate the solution in time

t = t_0
for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf.write_function(uh, t)
xdmf.close()
