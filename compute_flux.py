from cbcpost.fieldbases.Field import Field
from dolfin import *
class Fluxrate(Field):
    def __init__(self, tag,params=None, name=None):
        Field.__init__(self,params,"FluxThrough"+str(tag), label=None)
	self.tag=tag


    def before_first_compute(self, get):
        u = get("Velocity")
        V = u.function_space()
	mesh = V.mesh()

 	self.facet2 = FacetFunction('size_t', mesh, 0)

	for f,marker in ([int(f),marker] for f, marker in mesh.domains().markers(2).iteritems()):
		self.facet2[f] = marker


        self.n = FacetNormal(mesh)

	self.ds = Measure('ds', domain=mesh, subdomain_data=self.facet2, subdomain_id=self.tag)

    def compute(self, get):

        u = get('Velocity')
  
        value = assemble(inner(u,self.n)*self.ds )

        return value
