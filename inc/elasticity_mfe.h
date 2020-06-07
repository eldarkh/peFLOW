// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_ELASTICITY_MFE_H
#define PEFLOW_ELASTICITY_MFE_H

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/parsed_function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include "../inc/problem.h"
#include "../inc/utilities.h"
#include "../inc/elasticity_data.h"

namespace elasticity
{
  using namespace dealii;
  using namespace peflow;

  /*
   * Class implementing the mixed finite element method
   * for Linear elasticity model with weakly imposed symmetry.
   * The spaces are BDM(k) - DG_P(k-1) - DG_P(k-1), hence k > 0.
   * For k-th order method, the expected convergence rates are k
   * in all variables. The resulting system is solved directly.
   */
  template <int dim>
  class MixedElasticityProblem : public ElasticityProblem<dim>
  {
  public:
    using ElasticityProblem<dim>::computing_timer;
    using ElasticityProblem<dim>::fe;
    using ElasticityProblem<dim>::dof_handler;
    using ElasticityProblem<dim>::triangulation;
    using ElasticityProblem<dim>::prm;
    using ElasticityProblem<dim>::degree;
    using ElasticityProblem<dim>::total_dim;
    using ElasticityProblem<dim>::solution;
    using ElasticityProblem<dim>::convergence_table;
    using ElasticityProblem<dim>::compute_errors;
    using ElasticityProblem<dim>::output_results;

    /*
     * Class constructor takes degree and reference to parameter handle
     * as arguments
     */
    MixedElasticityProblem(const unsigned int deg,
                           ParameterHandler &param);
    /*
     * Main driver function
     */
    virtual void run(const unsigned int refine, const unsigned int grid = 0);
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
                               const LameCoefficients<dim> &lame, 
                               Functions::ParsedFunction<dim> *bc,
                               Functions::ParsedFunction<dim> *rhs);
      CellAssemblyScratchData (const CellAssemblyScratchData &scratch_data);
      FEValues<dim>     fe_values;
      FEFaceValues<dim> fe_face_values;
      LameCoefficients<dim> lame;
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
     * Solve the saddle-point type system (Direct UMFPACK)
     */
    void solve();

    /*
     * Data structures and internal parameters
     */
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockVector<double> system_rhs;
  };
}

#endif //PEFLOW_ELASTICITY_MFE_H
