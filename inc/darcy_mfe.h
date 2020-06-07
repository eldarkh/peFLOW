// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_DARCY_MFE_H
#define PEFLOW_DARCY_MFE_H

#include <deal.II/base/parsed_function.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include "../inc/problem.h"
#include "../inc/utilities.h"
#include "../inc/darcy_data.h"


namespace darcy
{
  using namespace dealii;
  using namespace peflow;

  /*
   * Class implementing the classical Raviart-Thomas mixed finite
   * element method for Darcy problem. The corresponding pressure
   * space is DG_Q(k). For k-th order method, the expected
   * convergence rates are k in all variables. The resulting
   * system is solved via Schur complement approach, using CG with
   * diagonal preconditioner.
   */
  template <int dim>
  class DarcyMFE : public DarcyProblem<dim>
  {
  public:
    using DarcyProblem<dim>::computing_timer;
    using DarcyProblem<dim>::fe;
    using DarcyProblem<dim>::dof_handler;
    using DarcyProblem<dim>::triangulation;
    using DarcyProblem<dim>::prm;
    using DarcyProblem<dim>::degree;
    using DarcyProblem<dim>::solution;
    using DarcyProblem<dim>::convergence_table;
    using DarcyProblem<dim>::postprocess;
    using DarcyProblem<dim>::compute_errors;
    using DarcyProblem<dim>::output_results;
    /*
     * Class constructor takes degree and reference to parameter handle
     * as arguments
     */
    DarcyMFE(const unsigned int degree,
             ParameterHandler &);
    /*
     * Main driver function
     */
    virtual void run(const unsigned int refine, const unsigned int grid);
  private:
    /*
     * Data structure holding the information needed by threads
     * during assembly process
     */
    struct CellAssemblyScratchData
    {
      CellAssemblyScratchData (const FiniteElement<dim> &fe,
                               const Quadrature<dim>    &quadrature,
                               const Quadrature<dim-1>  &face_quadrature,
                               const KInverse<dim> &k_data,
                               Functions::ParsedFunction<dim> *rhs,
                               Functions::ParsedFunction<dim> *bc);
      CellAssemblyScratchData (const CellAssemblyScratchData &scratch_data);
      FEValues<dim>     fe_values;
      FEFaceValues<dim> fe_face_values;
      KInverse<dim>     K_inv;
      Functions::ParsedFunction<dim> *bc;
      Functions::ParsedFunction<dim> *rhs;
    };

    /*
     * Structure to copy data from threads to the main
     */
    struct CellAssemblyCopyData
    {
      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;
    };

    /*
     * Make grid, distribute DoFs and create sparsity pattern
     */
    void make_grid_and_dofs();

    /*
     * Assemble cell matrix and RHS, worker function for each thread
     */
    void assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                               CellAssemblyScratchData                             &scratch,
                               CellAssemblyCopyData                                &copy_data);

    /*
     * Copy data from threads to main
     */
    void copy_local_to_global (const CellAssemblyCopyData &copy_data);

    /*
     * Function to assign each thread to matrix and RHS assembly
     */
    void assemble_system();

    /*
     * Solve the saddle-point type system (CG with preconditioner)
     */
    void solve();

    /*
     * Data structures and internal parameters
     */
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockVector<double>       system_rhs;
  };
}

#endif //PEFLOW_DARCY_MFE_H
