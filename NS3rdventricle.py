#!/usr/bin/env python

from cbcflow import *
from cbcpost import PostProcessor
from dolfin import *

from os import path
import numpy as np


class NS3rdVentricle(NSProblem):
    "3D pipe test problem with known transient analytical solution."

    @classmethod
    def default_params(cls):
        params = NSProblem.default_params()
        params.replace(
            # Time parameters
            T=None,
            dt=8e-5,
            period=0.8,
            num_periods=1.0,
            # Physical parameters
            rho=0.000993,
            mu=0.000676,
	    mesh_file="3rdventricle-refine1.xml.gz",
            )
        params.update(Q=160
            # Spatial parameters
            # Analytical solution parameters
            )
        return params

    def __init__(self, params=None):
        NSProblem.__init__(self, params)

        # Load mesh
        
       	mesh = Mesh("3rdventricle-refine1.xml.gz")

        #mesh = Mesh(self.params.mesh_filename)

        # We know that the mesh contains markers with these id values
        self.wall_boundary_id = 0
        self.aquaduct_boundary_id = 2
        self.right_lateral_ventricle_boundary_id = 1
	self.left_lateral_ventricle_boundary_id = 3

        # Setup analytical solution constants
        Q = self.params.Q
	Q2 = Q*0.5
        self.nu = 0.6828 #self.params.mu / self.params.rho

        # Beta is the Poiseuille pressure drop if the flow rate is stationary Q
        self.beta = 4.0 * self.nu * Q / (pi * 3.0**4)
	self.beta2 =  4.0 * self.nu * Q / (pi * 1.0**4)
        
        if 0:
            print "Using stationary bcs."
            self.Q_coeffs = [(0.0,Q), (1.0, Q)]
            self.Q_coeffs2 = [(0.0,Q2), (1.0, Q2)]
        else:
            print "Using transient bcs."
            P = self.params.period
            tvalues = np.linspace(0.0, P)
            #Qfloor, Qpeak = 1.0, 0.0
            #Qfloor, Qpeak = 0.3, 0.7
            Qfloor, Qpeak = -140.0, 300.0
            Qvalues = Q * (Qfloor + (Qpeak-Qfloor)*np.sin(pi*((P-tvalues)/P)**2)**2)
            Qvalues2 = Q2 * (Qfloor + (Qpeak-Qfloor)*np.sin(pi*((P-tvalues)/P)**2)**2)
            self.Q_coeffs = zip(tvalues, Qvalues)
	    self.Q_coeffs2 = zip(tvalues, Qvalues)
		
        # Store mesh and markers
        self.initialize_geometry(mesh) #, facet_domains=facet_domains)

    def womersly_aquaduct(self, spaces, t):
        # Create womersley objects
        ua = make_womersley_bcs(self.Q_coeffs, self.mesh, self.aquaduct_boundary_id, self.nu, None)
        for uc in ua:
            uc.set_t(t)
        pa = Expression("-beta", beta=1.0,degree=1)
        pa.beta = 0.0 # self.beta # TODO: This is not correct unless stationary...
        return (ua, pa)

    def womersly_rlv(self, spaces, t):
        # Create womersley objects
        ua = make_womersley_bcs(self.Q_coeffs2, self.mesh, self.right_lateral_ventricle_boundary_id , self.nu, None)
        for uc in ua:
            uc.set_t(t)
        pa = Expression("-beta ", beta=1.0,degree=1)
        pa.beta = 0.0 #self.beta2 # TODO: This is not correct unless stationary...
        return (ua, pa)

    def womersly_llv(self, spaces, t):
        # Create womersley objects
        ua = make_womersley_bcs(self.Q_coeffs2, self.mesh, self.left_lateral_ventricle_boundary_id , self.nu, None)
        for uc in ua:
            uc.set_t(t)
        pa = Expression("-beta", beta=1.0,degree=1)
        pa.beta = self.beta2 # TODO: This is not correct unless stationary...
        return (ua, pa)


    def test_fields(self):
        return [Velocity(), Pressure()]

    def test_references(self, spaces, t):
        return self.womersly_aquaduct(spaces, t)

    def initial_conditions(self, spaces, controls):
	c0 =Constant("0.0")
	icu = [c0 ,c0 ,c0]
	icp = c0
        return (icu ,icp)

    def boundary_conditions(self, spaces, u, p, t, controls):
        # Create no-slip bcs
        d = len(u)
        u0 = [Constant(0.0)] * d
        noslip = (u0, self.wall_boundary_id)

	ua, pa = self.womersly_aquaduct( spaces, t)
   	
	inflow  = ( ua, self.aquaduct_boundary_id)
	p_inflow = ( pa, self.aquaduct_boundary_id)


        # Create outflow boundary conditions for velocity
        ul,pl = self.womersly_llv(spaces, t)
        ur,pr = self.womersly_rlv(spaces, t)
   

	ur_outflow = (ur,self.right_lateral_ventricle_boundary_id)
	ul_outflow = (ul,self.left_lateral_ventricle_boundary_id)

	pr_outflow = (pr,self.right_lateral_ventricle_boundary_id)
	pl_outflow = (pl,self.left_lateral_ventricle_boundary_id)

	
        # Return bcs in two lists
        bcu = [noslip,inflow]
        bcp = []
        if 1: # Switch between pressure or dbc at outlet
            bcp += [pl_outflow,pr_outflow]
        else:
            bcu += [ul_outflow,ur_outflow]
        return (bcu, bcp)

    def update(self, spaces, u, p, t, timestep, bcs, observations, controls):
        bcu, bcp = bcs
        uin = bcu[1][0]
        for ucomp in uin:
            ucomp.set_t(t)


def main():
    problem = NS3rdVentricle()
    scheme = IPCS(dict(
        rebuild_prec_frequency = 1,
        u_tent_prec_structure = "same_nonzero_pattern",
        #p_corr_solver_parameters = dict(relative_tolerance=1e-6, absolute_tolerance=1e-6, monitor_convergence=False),
        u_degree=1,
        p_degree=1,
        solver_u_tent=("gmres", "additive_schwarz"),
        #solver_u_tent=("gmres", "hypre_euclid"),
        solver_p=("cg", "amg"),
        low_memory_version = False,
        #solver_u_tent=("mumps",),
        #solver_u_corr=("cg", "additive_schwarz"),
        solver_u_corr = "WeightedGradient",
        isotropic_diffusion=0.0,
        streamline_diffusion=0.2,
        crosswind_diffusion=0.0,
        #theta=-0.5,
        alpha=1.0,
        ))



    casedir = "results_refine_%s_%s_%d" % (problem.shortname(), scheme.shortname(), 1)
    plot_and_save = dict(save=True,stride_timestep=10)
    fields = [
        Pressure(plot_and_save),
        Velocity(plot_and_save),
        ]
    postproc = PostProcessor({"casedir": casedir})
    postproc.add_fields(fields)

    solver = NSSolver(problem, scheme, postproc)
    solver.solve()

if __name__ == "__main__":
    main()
