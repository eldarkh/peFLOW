// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#include <deal.II/base/work_stream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>

#include "../inc/problem.h"
#include "../inc/darcy_data.h"
#include "../inc/darcy_mfe.h"
#include "../inc/utilities.h"

namespace darcy
{
    using namespace dealii;
    using namespace utilities;
    using namespace peflow;

    // DarcyMFE: class constructor
    template <int dim>
    DarcyMFE<dim>::DarcyMFE (const unsigned int degree, ParameterHandler &param)
            :
            DarcyProblem<dim>(degree, param,
                              FESystem<dim>(FE_RaviartThomas<dim>(degree-1), 1, FE_DGQ<dim>(degree-1), 1))
    {}


    // DarcyMFE: make grid and DoFs
    template <int dim>
    void DarcyMFE<dim>::make_grid_and_dofs()
    {
        TimerOutput::Scope t(computing_timer, "Make grid and DOFs");
        system_matrix.clear();

        dof_handler.distribute_dofs(fe);

        DoFRenumbering::component_wise (dof_handler);

        std::vector<types::global_dof_index> dofs_per_component (dim + 1);
        DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
        const unsigned int n_u = dofs_per_component[0],
                n_p = dofs_per_component[dim];


        BlockDynamicSparsityPattern dsp(2, 2);
        dsp.block(0, 0).reinit (n_u, n_u);
        dsp.block(1, 0).reinit (n_p, n_u);
        dsp.block(0, 1).reinit (n_u, n_p);
        dsp.block(1, 1).reinit (n_p, n_p);
        dsp.collect_sizes ();
        DoFTools::make_sparsity_pattern (dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit (sparsity_pattern);
        solution.reinit (2);
        solution.block(0).reinit (n_u);
        solution.block(1).reinit (n_p);
        solution.collect_sizes ();
        system_rhs.reinit (2);
        system_rhs.block(0).reinit (n_u);
        system_rhs.block(1).reinit (n_p);
        system_rhs.collect_sizes ();

        for (typename Triangulation<dim>::active_cell_iterator
                     cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            {
                if ((cell->face(f)->at_boundary())
                    &&
                    (cell->face(f)->center()[1]==1.0))
                    cell->face(f)->set_all_boundary_ids(1);
            }

        {
            const FEValuesExtractors::Vector velocity(0);

            std::map<types::global_dof_index,double> boundary_values;

        }
    }


    // Scratch data for multithreading
    template <int dim>
    DarcyMFE<dim>::CellAssemblyScratchData::
    CellAssemblyScratchData (const FiniteElement<dim> &fe,
                             const Quadrature<dim>    &quadrature,
                             const Quadrature<dim-1>  &face_quadrature,
                             const KInverse<dim> &k_data,
                             Functions::ParsedFunction<dim> *bc_data,
                             Functions::ParsedFunction<dim> *rhs_data)
            :
            fe_values (fe,
                       quadrature,
                       update_values   | update_gradients |
                       update_quadrature_points | update_JxW_values),
            fe_face_values (fe,
                            face_quadrature,
                            update_values     | update_quadrature_points   |
                            update_JxW_values | update_normal_vectors),
            K_inv(k_data),
            bc(bc_data),
            rhs(rhs_data)
    {}


    template <int dim>
    DarcyMFE<dim>::CellAssemblyScratchData::
    CellAssemblyScratchData (const CellAssemblyScratchData &scratch_data)
            :
            fe_values (scratch_data.fe_values.get_fe(),
                       scratch_data.fe_values.get_quadrature(),
                       update_values   | update_gradients |
                       update_quadrature_points | update_JxW_values),
            fe_face_values (scratch_data.fe_face_values.get_fe(),
                            scratch_data.fe_face_values.get_quadrature(),
                            update_values     | update_quadrature_points   |
                            update_JxW_values | update_normal_vectors),
            K_inv(scratch_data.K_inv),
            bc(scratch_data.bc),
            rhs(scratch_data.rhs)
    {}


    // Copy local contributions to global system
    template <int dim>
    void DarcyMFE<dim>::copy_local_to_global (const CellAssemblyCopyData &copy_data)
    {
        for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
        {
            for (unsigned int j=0; j<copy_data.local_dof_indices.size(); ++j)
                system_matrix.add (copy_data.local_dof_indices[i],
                                   copy_data.local_dof_indices[j],
                                   copy_data.cell_matrix(i,j));
            system_rhs(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
        }
    }


    // Function to assemble on a cell
    template <int dim>
    void DarcyMFE<dim>::assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                       CellAssemblyScratchData                                   &scratch_data,
                                                       CellAssemblyCopyData                                      &copy_data)
    {
        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();
        const unsigned int n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();

        copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
        copy_data.cell_rhs.reinit (dofs_per_cell);
        copy_data.local_dof_indices.resize(dofs_per_cell);

        scratch_data.fe_values.reinit (cell);

        // Velocity  and Pressure DoFs vectors
        const FEValuesExtractors::Vector velocity(0);
        const FEValuesExtractors::Scalar pressure (dim);


        std::vector<Tensor<2,dim>>             k_inverse_values (n_q_points);
        scratch_data.K_inv.value_list (scratch_data.fe_values.get_quadrature_points(), k_inverse_values);


        for (unsigned int q=0; q<n_q_points; ++q)
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                const Tensor<1,dim> phi_i_u     = scratch_data.fe_values[velocity].value (i, q);
                const double        div_phi_i_u = scratch_data.fe_values[velocity].divergence (i, q);
                const double        phi_i_p     = scratch_data.fe_values[pressure].value (i, q);
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    const Tensor<1,dim> phi_j_u     = scratch_data.fe_values[velocity].value (j, q);
                    const double        div_phi_j_u = scratch_data.fe_values[velocity].divergence (j, q);
                    const double        phi_j_p     = scratch_data.fe_values[pressure].value (j, q);
                    copy_data.cell_matrix(i,j) += (phi_i_u * k_inverse_values[q] * phi_j_u
                                                   - div_phi_i_u * phi_j_p
                                                   - phi_i_p * div_phi_j_u)
                                                  * scratch_data.fe_values.JxW(q);
                }

                copy_data.cell_rhs(i) += -phi_i_p *scratch_data.rhs->value(scratch_data.fe_values.get_quadrature_points()[q]) 
                                            *scratch_data.fe_values.JxW(q);
            }
        }

        for (unsigned int face_no=0;
             face_no<GeometryInfo<dim>::faces_per_cell;
             ++face_no)
            if ((cell->at_boundary(face_no)) ) // && (cell->face(face_no)->boundary_id() != 1)
            {
                scratch_data.fe_face_values.reinit (cell, face_no);

                for (unsigned int q=0; q<n_face_q_points; ++q)
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        copy_data.cell_rhs(i) += -(scratch_data.fe_face_values[velocity].value (i, q) *
                                                   scratch_data.fe_face_values.normal_vector(q) *
                                                   scratch_data.bc->value(scratch_data.fe_face_values.get_quadrature_points()[q]) *
                                                   scratch_data.fe_face_values.JxW(q));

                    }
            }
        cell->get_dof_indices (copy_data.local_dof_indices);
    }


    template <int dim>
    void DarcyMFE<dim>::assemble_system ()
    {
      Functions::ParsedFunction<dim> *k_inv    = new Functions::ParsedFunction<dim>(dim*dim);
      Functions::ParsedFunction<dim> *bc       = new Functions::ParsedFunction<dim>(1);
      Functions::ParsedFunction<dim> *rhs      = new Functions::ParsedFunction<dim>(1);

      prm.enter_subsection(std::string("permeability ") + Utilities::int_to_string(dim)+std::string("D"));
      k_inv->parse_parameters(prm);
      prm.leave_subsection();

      prm.enter_subsection("BC " + Utilities::int_to_string(dim)+std::string("D"));
      bc->parse_parameters(prm);
      prm.leave_subsection();

      prm.enter_subsection("RHS " + Utilities::int_to_string(dim)+std::string("D"));
      rhs->parse_parameters(prm);
      prm.leave_subsection();

      TimerOutput::Scope t(computing_timer, "Assemble system");
      QGauss<dim> quad(2*(degree+1)+1);
      QGauss<dim-1> face_quad(2*(degree+1)+1);

      KInverse<dim> k_inverse(prm,k_inv);

      WorkStream::run(dof_handler.begin_active(),
                      dof_handler.end(),
                      *this,
                      &DarcyMFE::assemble_system_cell,
                      &DarcyMFE::copy_local_to_global,
                      CellAssemblyScratchData(fe,quad,face_quad, k_inverse, bc, rhs),
                      CellAssemblyCopyData());

      delete k_inv;
      delete rhs;
      delete bc;
    }


    // DarcyMFE: Solve
    template <int dim>
    void DarcyMFE<dim>::solve ()
    {
      TimerOutput::Scope t(computing_timer, "Solve (Direct UMFPACK)");

      SparseDirectUMFPACK  A_direct;
      A_direct.initialize(system_matrix);
      A_direct.vmult (solution, system_rhs);
    }


    // DarcyMFE: run
    template <int dim>
    void DarcyMFE<dim>::run(const unsigned int refine, const unsigned int grid)
    {
      dof_handler.clear();
      triangulation.clear();
      convergence_table.clear();

      for (unsigned int cycle=0; cycle<refine; ++cycle)
        {
            if(cycle == 0)
            {
              if(grid)
              {
                  GridIn<dim> grid_in;
                  grid_in.attach_triangulation (triangulation);
                  std::string mesh_filename ("mesh"+std::to_string(dim)+"d.msh");
                  std::ifstream input_file(mesh_filename);

                  Assert(input_file.is_open(), ExcFileNotOpen(mesh_filename.c_str()));
                  Assert(triangulation.dimension == dim, ExcDimensionMismatch(triangulation.dimension, dim));

                  grid_in.read_msh (input_file);
              }
              else
              {
                GridGenerator::hyper_cube (triangulation, 0, 1);
                triangulation.refine_global(2);
                GridTools::transform(&grid_transform<dim>, triangulation);
              }

              typename Triangulation<dim>::cell_iterator
                      cell = triangulation.begin (),
                      endc = triangulation.end();
              for (; cell!=endc; ++cell)
                for (unsigned int face_number=0;
                     face_number<GeometryInfo<dim>::faces_per_cell;
                     ++face_number)
                    if ((std::fabs(cell->face(face_number)->center()(0) - (1)) < 1e-12)
                        ||
                        (std::fabs(cell->face(face_number)->center()(1) - (1)) < 1e-12))
                        cell->face(face_number)->set_boundary_id (1);
            }
            else
            {
              triangulation.refine_global(1);
            }

            make_grid_and_dofs();
            assemble_system ();
            solve ();
            postprocess();
            compute_errors(cycle);
            output_results (cycle, refine);
            computing_timer.print_summary();
            computing_timer.reset();
        }
    }

  // Explicit instantiation
  template class DarcyMFE<2>;
  template class DarcyMFE<3>;
}


