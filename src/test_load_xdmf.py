# Need the h5py module
import h5py
from dolfin import *

mesh = Mesh("../meshes/3rdventricle-refine1.xml.gz")
# Create a Function to read the solution into
V = VectorFunctionSpace(mesh, "CG", 1)
u0 = Function(V)

# Open the result file for reading
fl = h5py.File("results_refine_NS3rdVentricle_IPCS_1/Velocity/Velocity.h5", "r")

# Choose the first time step
vec = fl["/VisualisationVector/10"]

# Scalar FunctionSpace Q is required for mapping vertices to dofs 
Q = FunctionSpace(mesh, 'CG', 1)
q = Function(Q)
v3d = vertex_to_dof_map(Q)
# Now map vertexfunction to the V function space
for i in range(3):
    q.vector()[v3d] = vec[:, i]
    assign(u0.sub(i), q)

plot(u0, mesh=mesh)
interactive()

