// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------
#ifndef PEFLOW_PROBLEM_H
#define PEFLOW_PROBLEM_H

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/parsed_function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include "../inc/utilities.h"
#include "../inc/darcy_data.h"

/*
 * Interface class with only one pure virtual function void run().
 */
namespace peflow
{
  using namespace dealii;
  using namespace utilities;

  /*
   * Interface for all problems
   */
  template <int dim>
  class Problem
  {
  public:
    virtual ~Problem() {}
    virtual void run (const unsigned int refine, const unsigned int grid) = 0;
  };



  /*
   * Base class for Darcy problem
   */
  template <int dim>
  class DarcyProblem : public Problem<dim>
  {
  public:
    virtual void run (const unsigned int refine, const unsigned int grid) = 0;

  protected:
    DarcyProblem (const unsigned int degree, ParameterHandler &param, FESystem<dim> fespace)
            :
            prm(param),
            degree(degree),
            dof_handler(triangulation),
            fe(fespace),
            fe_pres_post(degree),
            dof_handler_pres_post(triangulation),
            computing_timer(std::cout, TimerOutput::summary,
                            TimerOutput::wall_times)
    {}
    /*
     * Reference to a parameter handler object that stores parameters,
     * data and the exact solution
     */
    ParameterHandler &prm;

    /*
     * Functions that compute errors and output results and
     * convergence rates
     */
    virtual void compute_errors (const unsigned int cycle);
    virtual void output_results (const unsigned int cycle,  const unsigned int refine);

    /*
    * Do postprocessing for pressure
    */
    struct PostProcessScratchData;
    virtual void postprocessing_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                              PostProcessScratchData                               &scratch_data,
                              unsigned int &);
    virtual void postprocess ();

    /*
     * Data structures and internal parameters
     */
    const unsigned int degree;
    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;
    FESystem<dim>      fe;

    FE_DGQ<dim>          fe_pres_post;
    DoFHandler<dim>      dof_handler_pres_post;
    Vector<double>       solution_pres_post;

    BlockVector<double> solution;

    /*
     * Convergence table and wall-time timer objects
     */
    ConvergenceTable convergence_table;
    TimerOutput      computing_timer;
  };



  /*
   * Base class for Elasticity problem
   */
  template <int dim>
  class ElasticityProblem : public Problem<dim>
  {
  public:
    virtual void run (const unsigned int refine, const unsigned int grid) = 0;

  protected:
    ElasticityProblem (const unsigned int degree, ParameterHandler &param, FESystem<dim> fespace)
            :
            prm(param),
            degree(degree),
            total_dim(dim*dim + dim + static_cast<int>(0.5*dim*(dim-1))),
            dof_handler(triangulation),
            fe(fespace),
            computing_timer(std::cout, TimerOutput::summary,
                            TimerOutput::wall_times)
    {}

    /*
     * Reference to a parameter handler object that stores parameters,
     * data and the exact solution
     */
    ParameterHandler &prm;

    /*
     * Functions that compute errors and output results and
     * convergence rates
     */
    virtual void compute_errors (const unsigned int cycle);
    virtual void output_results (const unsigned int cycle,  const unsigned int refine);


    /*
     * Data structures and internal parameters
     */
    const unsigned int  degree;
    const int           total_dim;
    Triangulation<dim>  triangulation;
    DoFHandler<dim>     dof_handler;
    FESystem<dim>       fe;
    BlockVector<double> solution;

    /*
     * Convergence table and wall-time timer objects
     */
    ConvergenceTable convergence_table;
    TimerOutput      computing_timer;

    bool b_scaled_rotation;
  };
}


#endif //PEFLOW_PROBLEM_H
