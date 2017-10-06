from cbcpost.fieldbases.Field import Field
from cbcpost.fieldbases.MetaField import MetaField
from itertools import ifilter, imap, izip, chain
from itertools import count as C_O_U_N_T
from collections import defaultdict, namedtuple
from math import sqrt
import dolfin as df
import numpy as np
import random

class CellWithParticles(df.Cell):
    '''
    Dolfin cell holding a set of particles which are keys in the
    lp_collection dictionary.
    '''
    def __init__(self, lp_collection, cell_id):
        mesh = lp_collection.mesh
        df.Cell.__init__(self, mesh, cell_id)
        # NOTE: the choice for set here is to allow for removing left particles
        # by difference 
        self.particles = set([])
        # Make cell aware of its neighbors
        tdim = lp_collection.dim
        neighbors = sum((vertex.entities(tdim).tolist() for vertex in df.vertices(self)), [])
        neighbors = set(neighbors) - set([cell_id])   # Remove self
        self.neighbors = map(lambda neighbor_index: df.Cell(mesh, neighbor_index), neighbors)

    def __add__(self, particle):
        '''Add a particle.'''
        self.particles.add(particle)

    def __len__(self):
        '''Number of particles'''
        return len(self.particles)


def subdomain_count(lpc, subdomains, markers):
    '''
    Given cell function(subdomains) return local and global count of
    particles found in regions marked as markers.
    '''
    assert isinstance(subdomains, df.CellFunctionSizet)

    local_counts = [sum(imap(len, ifilter(lambda cell: subdomains[cell] == marker,
                                          lpc.cells.itervalues())))
                    for marker in markers]
    local_counts = np.array(local_counts, dtype=int)
    global_counts = np.zeros_like(local_counts)
    lpc.comm.Allreduce(local_counts, global_counts)    

    return tuple(local_counts), tuple(global_counts)


def subdomain_seed(lpc, nparticles, subdomains=None, markers=None):
    '''Seed particles in marked subdomain.'''
    # We shall chose the points by selecting a random cell (allowed if it has a
    # proper tag) and inside the cells we make a random point
    mesh = lpc.mesh
    if subdomains is not None and markers is not None:
        allowed = set(c.index()
                      for c in chain(*[df.SubsetIterator(subdomains, m) for m in markers]))
        allowed = list(allowed)
        random_cell = lambda: random.choice(allowed)
    else:
        # Without markers all cells are allowed
        ncells = mesh.topology().size(mesh.topology().dim())

        random_cell = lambda: random.randint(0, ncells-1)
   
    # Random point inside triangle/tet
    # http://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
    dim = mesh.geometry().dim()
    if dim == 2:
        def random_point(A, B, C):
            '''See Milikan's answer.'''
            r1 = sqrt(random.random())
            r2 = random.random()
            return (1-r1)*A + r1*(1-r2)*B + r2*r1*C
    else:
        assert dim == 3
        def random_point(A, B, C, D):
            '''See the thread for generalization'''
            c = np.random.rand(4)
            c = -np.log(c)
            c /= sum(c)
            return A*c[0] + B*c[1] + C*c[2] + D*c[3]

    count = 0
    particles = []
    tag=[]
    lpc.reset_tickets(nparticles*lpc.comm.rank) # 
    while count < nparticles:
        cell = random_cell()
        x = df.Cell(mesh, cell).get_vertex_coordinates().reshape((-1, dim))
        p = random_point(*x)
        particles.append(p) 
	tag.append(int(next(lpc.ticket))) # ID TAG 
        count += 1
  
    particles = np.c_[particles,tag]
    return lpc.add_particles(particles)



class LPCollection(MetaField): # NOT a field 
    '''
    Collection of Lagrangian particles. Particle data is stored in a dictionary 
    and cells which contain them hold reference to the data.
    '''

    def __init__(self, V, property_layout=None, params=None, name="default", label=None, debug=False):
	
        MetaField.__init__(self, "Velocity", params ,name ,label)

     
        # The cell neighbors requires cell-cell connectivity which is here
        # defined as throught cell - vertex - cell connectivity
        mesh = V.mesh()
        self.dim = mesh.geometry().dim()
        assert self.dim == mesh.topology().dim()
        mesh.init(0, self.dim)
        self.mesh = mesh
        
        # Locating particles in cells done by bbox collisions
        self.tree = mesh.bounding_box_tree()
        self.lim = mesh.topology().size_global(self.dim)
        
        # Velocity evaluation is done by restriction the function on a cell. For
        # this we prealocate data
        element = V.dolfin_element()
        
        num_tensor_entries = 1
        for i in range(element.value_rank()): num_tensor_entries *= element.value_dimension(i)

        self.basis_matrix = np.zeros((element.space_dimension(), num_tensor_entries))
        self.element = element

        # Let's decide how to do stepping
        if element.value_rank() == 1:
            self.step = self.__step_vector
            self.coefficients = np.zeros(element.space_dimension())
        else:
            assert element.value_rank() == 0
            self.step = self.__step_scalar
            self.coefficients = np.zeros((self.dim, element.space_dimension()))

        # Particles and cells are stored in dicts and we refer to each by ints.
        # For cell this is the cell index, particles get a ticket from their
        # counter. # NOTE: When particle leaves CPU it is removed from the
        # dictionary in turn particles.keys may develop 'holes'.
        self.particles = {}
        self.cells = {}
        self.ticket = C_O_U_N_T()
	self.counted=True
        self.verbose=False

	
        # Property layout here is the map which maps property name to
        # length of a vector that represents property value. By default
        # particles only store the position
        self.gem =None
        if property_layout is None: 
              property_layout = []
        else:
              self.gem = property_layout[0][0]
	
        property_layout = [('x', self.dim)] + property_layout
        props, sizes = map(list, zip(*property_layout))
        assert all(v > 0 for v in sizes)
        offsets = [0] + sizes
        self.psize = sum(offsets)    # How much to store per particle
        self.offsets = np.cumsum(offsets)
        self.keys = props

        # Finally the particles are send in circle  (prev) -> (this) --> (next)
        comm = mesh.mpi_comm().tompi4py()
        assert comm.size == 1 or comm.size % 2 == 0
        self.next_rank = (comm.rank + 1) % comm.size
        self.prev_rank = (comm.rank - 1) % comm.size
        self.comm = comm

        if debug:
            self.__add_particles_local = add_timer(self.__add_particles_local)
            self.__add_particles_global = add_timer(self.__add_particles_global)
            self.__update = add_timer(self.__update)

    # Most common in API --

    def compute(self,get):
         '''This function is required with the use of cbcpost. For each iteration a call to compute will be made'''   
         u = get("Velocity")
         t1 = get("t")
         t2 = get("t",-1)
         dt=t1 -t2
         self.step(u,dt)
         ''' The function will return itself, and the data will be stored using the class function store ''' 
         # self -> data -> Metafield -> data.store(filename)
         return  self   

    def add_particles(self, particles):
        '''Add new particles to collection.'''
        # How many particles do we start and end up with
        if self.verbose: start = plen(self.comm, particles).gc

        # Figure out which particles cannot be added locally on CPU
        not_found = self.__add_particles_local(particles)
        # Send unfoung particles to other CPUs to see if can be added there
        count = len(not_found)
        count_global = self.comm.allgather(count)
        not_found = self.__add_particles_global(count_global, not_found)

        if self.verbose: 
            missing = plen(self.comm, not_found).gc
            info('Wanted to add %d particles. Found %d.' % (start, start-missing))

   	
    def store(self, name, prop=''): # 
	prop = self.keys[1:][0]
	# NEED to timestep
        '''Save current particle position's (and scalar data) in XDMF file'''
        assert name.endswith('.xdmf')
        f = df.XDMFFile(self.mesh.mpi_comm(), name)
        if prop:
            i = self.keys.index(prop)
            is_scalar = (self.offsets[i+1] - self.offsets[i]) == 1
            assert is_scalar

            f.write(map(df.Point, imap(self.get_x, self.particles.iterkeys())),
                    np.array([self.get_property(p, prop) for p in self.particles.iterkeys()]),
                    df.XDMFFile.Encoding_HDF5)
        else:
            f.write(map(df.Point, imap(self.get_x, self.particles.iterkeys())),
                    df.XDMFFile.Encoding_HDF5)

    # Convenience ---

    def reset_tickets(self,start):
        '''Resets the tickets to an initial value of start'''
        self.ticket = C_O_U_N_T(start)


    def get_x(self, particle):
        '''Return position of particle.'''
        return self.particles[particle][:self.dim]

    def get_property(self, particle, prop):
        '''Get property of particle.'''
        i = self.keys.index(prop)
        f, l = self.offsets[i], self.offsets[i+1]
        select = lambda p: self.particles[p][f:l]

        if isinstance(particle, int): return select(particle)

        return map(select, particle)

    def cell_count(self):
        '''Local and global cell count of cells in collection.'''
        return plen(self.comm, self.cells)

    def particle_count(self):
        '''Local and global particle count of particles in collection.'''
        return plen(self.comm, self.particles)

    def find_cell(self, x):
        '''Find cell which contains x on this CPU, -1 if not found.'''
        point = df.Point(*x)
        c = self.tree.compute_first_entity_collision(point)
        return c if c < self.lim else -1

    # Core ---

    def __step_vector(self, u, dt):
        'Move particles by forward Euler x += u*dt'
        # Update positions of particles
        for c in self.cells.itervalues():
            vertex_coords, orientation = c.get_vertex_coordinates(), c.orientation()
            # Restrict once per cell
            u.restrict(self.coefficients, self.element, c, vertex_coords, c)
            for p in c.particles:
                x = self.get_x(p)
                # Compute velocity at position x
                self.element.evaluate_basis_all(self.basis_matrix, x, vertex_coords, orientation)
                x[:] = x[:] + dt*np.dot(self.coefficients, self.basis_matrix)[:]
        # Update cells/particles
        self.__update()

    def __step_scalar(self, u, dt):
        'Move particles by forward Euler x += u*dt'
        # Update positions of particles
        for c in self.cells.itervalues():
            vertex_coords, orientation = c.get_vertex_coordinates(), c.orientation()
            # Restrict once per cell each components
            for i, ui in enumerate(u):
                ui.restrict(self.coefficients[i], self.element, c, vertex_coords, c)

            for p in c.particles:
                x = self.get_x(p)
                # Compute velocity at position x
                self.element.evaluate_basis_all(self.basis_matrix, x, vertex_coords, orientation)
                for i in range(self.dim):
                    x[i] += dt*np.dot(self.coefficients[i], self.basis_matrix)[:]
        # Update cells/particles
        self.__update(self.verbose)

    def __add_particles_local(self, particles):
        '''Search CPU for cells that have particles.'''
        not_found = []
        for p in particles:
            x = p[:self.dim]

            c = self.find_cell(x)
            if c > -1: 
                # A found particle gets a new unique tag
                tag = next(self.ticket)
                self.particles[tag] = p
                if c not in self.cells: self.cells[c] = CellWithParticles(self, c)
                # and is added to the cell
                self.cells[c] + tag
            else:
                not_found.append(p)
        return not_found
    
    def __add_particles_global(self, count_global, not_found):
        '''Search other CPUs for cells that have particles.'''
        loop = 1
        # NOTE: if all the particles were in the computational domain then the
        # loop should terminate (at most) once the particles not found on some
        # process travel the full circle. Whathever is not found once the loop
        # is over is outside domain.
        while max(count_global) > 0 and loop < self.comm.size:
            loop += 1
            received = np.zeros(count_global[self.prev_rank]*self.psize, dtype=float)
            sent = np.array(not_found).flatten()
            # Send to next and recv from previous. NOTE: module prevents deadlock
            if self.comm.rank % 2:
                self.comm.Send(sent, self.next_rank, self.comm.rank)
                self.comm.Recv(received, self.prev_rank, self.prev_rank)
            else:
                self.comm.Recv(received, self.prev_rank, self.prev_rank)
                self.comm.Send(sent, self.next_rank, self.comm.rank)
            # Find cells on this CPU for received particles
            received = received.reshape((-1, self.psize))
            not_found = self.__add_particles_local(received)
            count = len(not_found)
            count_global = self.comm.allgather(count)
        self.comm.barrier()
        return not_found



    def __update(self):
        '''
        Update particle and cell dictionaries based on new position of
        particles.
        '''

        if self.verbose: start = self.particle_count().gc
        cell_map = defaultdict(list)    # Collect new cells with particles
        empty_cells = []                # Cells to be removed from self.cells
        for c in self.cells.itervalues():
            left = []
            for p in c.particles:
                x = self.get_x(p)
	
                point = df.Point(*x)
                found = c.contains(point)
                # Search only if particle moved outside original cell
                if not found:
                    left.append(p)
                    # Check first neighbor cells
                    for neighbor in c.neighbors:
                        found = neighbor.contains(point)
                        if found:
                            new_cell = neighbor.index()
                            break
                    # Do a completely new search if not found by now, can be -1
                    if not found: new_cell = self.find_cell(x)
                    # Record to map
                    cell_map[new_cell].append(p)
            # Remove from cell the particles that left it
            c.particles.difference_update(set(left))
            if len(c.particles) == 0: empty_cells.append(c.index())
       
        # Remove cells with no particles
        for c in empty_cells: self.cells.pop(c)

        # Add locally found particles
        local_cells = ifilter(lambda x: x != -1, cell_map.iterkeys())
        for c in local_cells:
            if c not in self.cells: self.cells[c] = CellWithParticles(self, c)
            for p in cell_map[c]: self.cells[c] + p

        # Ship particles not found on this CPU to others
        non_local_particles = cell_map.get(-1, [])
        count_global = self.comm.allgather(len(non_local_particles))
        not_found = self.__add_particles_global(count_global,
                                                [self.particles[i] for i in non_local_particles])
        # Finally remove these particles 
        for p in non_local_particles: self.particles.pop(p)

        if self.verbose: 
            stop = self.particle_count().gc
            info('Before update %d, after update %d' % (start, stop))


if __name__ == '__main__':
    from dolfin import UnitSquareMesh, VectorFunctionSpace, info, Timer, interpolate
    from dolfin import Expression, CellFunction, cells, UnitCubeMesh, CompiledSubDomain
    from dolfin import FunctionSpace
    from mpi4py import MPI as pyMPI
    from cbcpost import *
    
    import sys
    nparticles = int(sys.argv[1])


    pp = PostProcessor(dict(casedir="Results", clean_casedir=True))
  
   
    pp.add_field(SolutionField("Velocity",dict(save=True)))
    # TEST0: subdomain seeding
    mesh = UnitCubeMesh(10, 10, 10)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    v = interpolate(Expression(('-(x[1]-0.5)', 'x[0]-0.5', '0'), degree=1), V)

    subdomains = CellFunction('size_t', mesh, 0)
    CompiledSubDomain('x[2] > 0.2').mark(subdomains, 1)

    property_layout = [('Tag', 1)]
    params =  dict(save=True,save_as=["xdmf"],start_time=0)
    
    lpc = LPCollection(V,property_layout,params) # <- 
    subdomain_seed(lpc, nparticles, subdomains, [1])
    pp.add_field(lpc)
  
    t=0
    dt = 0.1
    for i in range(1,10):
        t += dt
	pp.update_all({"Velocity": lambda: v},t,i ) 
        #pp.update_all({"Particles": lambda: lpc}, t,i ) 
	print t
        #lpc.step(v, dt)
     
    pp.finalize_all()










