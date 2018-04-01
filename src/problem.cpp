// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2018 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include "../inc/darcy_data.h"
#include "../inc/elasticity_data.h"
#include "../inc/utilities.h"
#include "../inc/problem.h"

namespace peflow
{
  using namespace dealii;
  using namespace peflow;

  // Postprocessing
  template <int dim>
  struct DarcyProblem<dim>::PostProcessScratchData
  {
    FEValues<dim> fe_values;
    FEValues<dim> fe_values_post;

    std::vector<double> p_values;
    std::vector<Tensor<1,dim> > p_gradients;

    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    Vector<double> cell_sol;
    darcy::KInverse<dim> K_inv;

    PostProcessScratchData(const FiniteElement<dim> &fe,
                           const FiniteElement<dim> &fe_post,
                           const QGauss<dim>   &quadrature_formula,
                           const UpdateFlags flags,
                           const UpdateFlags post_flags,
                           const darcy::KInverse<dim> &k_data)
            :
            fe_values(fe, quadrature_formula, flags),
            fe_values_post (fe_post, quadrature_formula, post_flags),
            p_values (quadrature_formula.size()),
            p_gradients (quadrature_formula.size()),
            cell_matrix (fe_post.dofs_per_cell, fe_post.dofs_per_cell),
            cell_rhs (fe_post.dofs_per_cell),
            cell_sol (fe_post.dofs_per_cell),
            K_inv(k_data)
    {}

    PostProcessScratchData(const PostProcessScratchData &sd)
            :
            fe_values (sd.fe_values.get_fe(),
                       sd.fe_values.get_quadrature(),
                       sd.fe_values.get_update_flags()),
            fe_values_post (sd.fe_values_post.get_fe(),
                            sd.fe_values_post.get_quadrature(),
                            sd.fe_values_post.get_update_flags()),
            p_values (sd.p_values),
            p_gradients (sd.p_gradients),
            cell_matrix (sd.cell_matrix),
            cell_rhs (sd.cell_rhs),
            cell_sol (sd.cell_sol),
            K_inv(sd.K_inv)
    {}
  };

  template <int dim>
  void
  DarcyProblem<dim>::postprocessing_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                          PostProcessScratchData &scratch,
                                          unsigned int &)
  {
    typename DoFHandler<dim>::active_cell_iterator
            loc_cell (&triangulation,
                      cell->level(),
                      cell->index(),
                      &dof_handler);

    scratch.fe_values.reinit (loc_cell);
    scratch.fe_values_post.reinit(cell);

    FEValuesExtractors::Vector fluxes(0);
    FEValuesExtractors::Scalar scalar(dim);
    const unsigned int n_q_points = scratch.fe_values_post.get_quadrature().size();
    const unsigned int dofs_per_cell = scratch.fe_values_post.dofs_per_cell;

    std::vector<Tensor<2,dim>>             k_inverse_values (n_q_points);
    scratch.K_inv.value_list (scratch.fe_values_post.get_quadrature_points(), k_inverse_values);

    scratch.fe_values[scalar].get_function_values(solution, scratch.p_values);
    scratch.fe_values[fluxes].get_function_values(solution, scratch.p_gradients);
    double sum = 0;

    for (unsigned int i=1; i<dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        sum = 0;
        for (unsigned int q=0; q<n_q_points; ++q)
          sum += (scratch.fe_values_post.shape_grad(i,q) * scratch.fe_values_post.shape_grad(j,q))
                 * scratch.fe_values_post.JxW(q);
        scratch.cell_matrix(i,j) = sum;
      }

      sum = 0;
      for (unsigned int q=0; q<n_q_points; ++q)
        sum -= (k_inverse_values[q] * scratch.fe_values_post.shape_grad(i,q) * scratch.p_gradients[q])
               * scratch.fe_values_post.JxW(q);
      scratch.cell_rhs(i) = sum;
    }

    for (unsigned int j=0; j<dofs_per_cell; ++j)
    {
      sum = 0;
      for (unsigned int q=0; q<n_q_points; ++q)
        sum += scratch.fe_values_post.shape_value(j,q) * scratch.fe_values_post.JxW(q);
      scratch.cell_matrix(0,j) = sum;
    }
    {
      sum = 0;
      for (unsigned int q=0; q<n_q_points; ++q)
        sum += scratch.p_values[q] * scratch.fe_values_post.JxW(q);
      scratch.cell_rhs(0) = sum;
    }

    scratch.cell_matrix.gauss_jordan();
    scratch.cell_matrix.vmult(scratch.cell_sol, scratch.cell_rhs);
    cell->distribute_local_to_global(scratch.cell_sol, solution_pres_post);
  }

  template <int dim>
  void
  DarcyProblem<dim>::postprocess()
  {
    TimerOutput::Scope t(computing_timer, "Postprocess pressure");

    Functions::ParsedFunction<dim> *k_inv = new Functions::ParsedFunction<dim>(dim*dim);

    prm.enter_subsection(std::string("permeability ") + Utilities::int_to_string(dim)+std::string("D"));
    k_inv->parse_parameters(prm);
    prm.leave_subsection();

    darcy::KInverse<dim> k_inverse(prm,k_inv);

    dof_handler_pres_post.distribute_dofs(fe_pres_post);
    solution_pres_post.reinit(dof_handler_pres_post.n_dofs());

    const QGauss<dim> quadrature_formula(fe_pres_post.degree + 3);
    const UpdateFlags post_flags(update_values | update_gradients | update_quadrature_points |
                                 update_JxW_values);
    const UpdateFlags flags(update_values);
    PostProcessScratchData scratch(fe, fe_pres_post,
                                   quadrature_formula,
                                   flags, post_flags,
                                   k_inverse);

    WorkStream::run(dof_handler_pres_post.begin_active(),
                    dof_handler_pres_post.end(),
                    std::bind(&DarcyProblem<dim>::postprocessing_cell,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3),
                    std::function<void(const unsigned int &)>(),
                    scratch,
                    0U);

    delete k_inv;
  }

  // DarcyProblem: Compute errors
  template <int dim>
  void DarcyProblem<dim>::compute_errors(const unsigned cycle)
  {
    TimerOutput::Scope t(computing_timer, "Compute errors");

    {
      std::string filename = "PostPressure-";
      filename += ('0' + cycle);
      Assert (cycle < 10, ExcInternalError());
      filename += ".vtk";
      std::ofstream output (filename.c_str());
      DataOut<dim> data_out;
      data_out.attach_dof_handler (dof_handler_pres_post);
      data_out.add_data_vector (solution_pres_post, "pressurePost");
      data_out.build_patches ();
      data_out.write_vtk (output);
    }

    const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);

    darcy::PressureBoundaryValues<dim> pres;
    darcy::ExactSolution<dim> exact_solution(prm);
    prm.enter_subsection(std::string("Exact solution ")+ Utilities::int_to_string(dim)+std::string("D"));
    exact_solution.exact_solution_val_data.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Exact gradient ")+ Utilities::int_to_string(dim)+std::string("D"));
    exact_solution.exact_solution_grad_val_data.parse_parameters(prm);
    prm.leave_subsection();

    // Vectors to temporarily store cellwise errros
    Vector<double> cellwise_errors (triangulation.n_active_cells());
    Vector<double> cellwise_norms (triangulation.n_active_cells());

    // Vectors to temporarily store cellwise componentwise div errors
    Vector<double> cellwise_div_errors (triangulation.n_active_cells());
    Vector<double> cellwise_div_norms (triangulation.n_active_cells());

    // Define quadrature points to compute errors at
    QTrapez<1>      q_trapez;
    QIterated<dim>  quadrature(q_trapez,degree+3);

    // This is used to show superconvergence at midcells
    QGauss<dim>   quadrature_super(degree);

    // Since we want to compute the relative norm
    BlockVector<double> zerozeros(1, solution.size());
    Vector<double> zeros(solution_pres_post.size());


    // Pressure error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_norm = cellwise_norms.l2_norm();

    // Pressure error and norm at midcells
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature_super,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_mid_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature_super,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_mid_norm = cellwise_norms.l2_norm();

    // Velocity L2 error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    const double u_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);

    const double u_l2_norm = cellwise_norms.l2_norm();

    // Velocity Hdiv error and seminorm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_div_errors, quadrature,
                                       VectorTools::Hdiv_seminorm,
                                       &velocity_mask);
    const double u_hd_error = cellwise_div_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_div_norms, quadrature,
                                       VectorTools::Hdiv_seminorm,
                                       &velocity_mask);
    const double u_hd_norm = cellwise_div_norms.l2_norm();

    // Postprocessed pressure solution
    VectorTools::integrate_difference (dof_handler_pres_post, solution_pres_post, pres,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm);
    const double u_post_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler_pres_post, zeros, pres,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm);
    const double u_post_norm = cellwise_norms.l2_norm();

    // Assemble convergence table
    const unsigned int n_active_cells=triangulation.n_active_cells();
    const unsigned int n_dofs=dof_handler.n_dofs();

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("Velocity,L2", u_l2_error/u_l2_norm);
    convergence_table.add_value("Velocity,Hdiv", u_hd_error/u_hd_norm);
    convergence_table.add_value("Pressure,L2", p_l2_error/p_l2_norm);
    convergence_table.add_value("Pressure,L2mid", p_l2_mid_error/p_l2_mid_norm);
    convergence_table.add_value("Pressure,L2post", u_post_error/u_post_norm);
  }


  // DarcyProblem: Output results
  template <int dim>
  void DarcyProblem<dim>::output_results(const unsigned int cycle, const unsigned int refine)
  {
    TimerOutput::Scope t(computing_timer, "Output results");

    std::vector<std::string> solution_names;
    std::string rhs_name = "rhs";

    switch(dim)
    {
      case 2:
        solution_names.push_back ("u1");
        solution_names.push_back ("u2");
        solution_names.push_back ("p");
        break;
      case 3:
        solution_names.push_back ("u1");
        solution_names.push_back ("u2");
        solution_names.push_back ("u3");
        solution_names.push_back ("p");
        break;
      default:
      Assert(false, ExcNotImplemented());
    }


    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, solution_names, DataOut<dim>::type_dof_data);

    data_out.build_patches ();

    std::ofstream output ("solution" + std::to_string(dim) + "d-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk (output);

    convergence_table.set_precision("Velocity,L2", 3);
    convergence_table.set_precision("Velocity,Hdiv", 3);
    convergence_table.set_precision("Pressure,L2", 3);
    convergence_table.set_precision("Pressure,L2mid", 3);
    convergence_table.set_precision("Pressure,L2post", 3);
    convergence_table.set_scientific("Velocity,L2", true);
    convergence_table.set_scientific("Velocity,Hdiv", true);
    convergence_table.set_scientific("Pressure,L2", true);
    convergence_table.set_scientific("Pressure,L2mid", true);
    convergence_table.set_scientific("Pressure,L2post", true);
    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("Velocity,L2", "$ \\|\\u - \\u_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Velocity,Hdiv", "$ \\|\\nabla\\cdot(\\u - \\u_h)\\|_{L^2} $");
    convergence_table.set_tex_caption("Pressure,L2", "$ \\|p - p_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Pressure,L2mid", "$ \\|Qp - p_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Pressure,L2post", "$ \\|p^* - p_h\\|_{L^2} $");
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");

    convergence_table.evaluate_convergence_rates("Velocity,L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Velocity,Hdiv", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Pressure,L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Pressure,L2mid", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Pressure,L2post", ConvergenceTable::reduction_rate_log2);

    std::ofstream error_table_file("error" + std::to_string(dim) + "d.tex");

    if (cycle == refine-1){
      convergence_table.write_text(std::cout);
      convergence_table.write_tex(error_table_file);
    }
  }


  // ElasticityProblem: Compute errors
  template <int dim>
  void ElasticityProblem<dim>::compute_errors(const unsigned cycle)
  {
    TimerOutput::Scope t(computing_timer, "Compute errors");

    const ComponentSelectFunction<dim> rotation_mask(dim*dim+dim, total_dim);
    const ComponentSelectFunction<dim> displacement_mask(std::make_pair(dim*dim,dim*dim+dim), total_dim);
    const ComponentSelectFunction<dim> stress_mask(std::make_pair(0,dim*dim), total_dim);

    elasticity::ExactSolution<dim> exact_solution(prm);
    prm.enter_subsection(std::string("Exact solution ")+ Utilities::int_to_string(dim)+std::string("D"));
    exact_solution.exact_solution_val_data.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Exact gradient ")+ Utilities::int_to_string(dim)+std::string("D"));
    exact_solution.exact_solution_grad_val_data.parse_parameters(prm);
    prm.leave_subsection();


    // Vectors to temporarily store cellwise errros
    Vector<double> cellwise_errors (triangulation.n_active_cells());
    Vector<double> cellwise_norms (triangulation.n_active_cells());

    // Vectors to temporarily store cellwise componentwise div errors
    Vector<double> cellwise_div_errors (triangulation.n_active_cells());
    Vector<double> cellwise_div_norms (triangulation.n_active_cells());

    // Define quadrature points to compute errors at
    QTrapez<1>      q_trapez;
    QIterated<dim>  quadrature(q_trapez,degree+2);

    // This is used to show superconvergence at midcells
    QGauss<dim>   quadrature_super(1);

    // Since we want to compute the relative norm
    BlockVector<double> zerozeros(1, solution.size());

    // Rotation error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &rotation_mask);
    const double p_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &rotation_mask);
    const double p_l2_norm = cellwise_norms.l2_norm();

    // Displacement error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double u_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double u_l2_norm = cellwise_norms.l2_norm();

    // Displacement error and norm at midcells
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature_super,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double u_l2_mid_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature_super,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double u_l2_mid_norm = cellwise_norms.l2_norm();

    // Stress L2 error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &stress_mask);
    const double s_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &stress_mask);

    const double s_l2_norm = cellwise_norms.l2_norm();

    // Stress Hdiv seminorm
    cellwise_errors = 0;
    cellwise_norms = 0;
    for (int i=0; i<dim; ++i){
      const ComponentSelectFunction<dim> stress_component_mask (std::make_pair(i*dim,(i+1)*dim), total_dim);

      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_div_errors, quadrature,
                                         VectorTools::Hdiv_seminorm,
                                         &stress_component_mask);
      cellwise_errors += cellwise_div_errors;

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_div_norms, quadrature,
                                         VectorTools::Hdiv_seminorm,
                                         &stress_component_mask);
      cellwise_norms += cellwise_div_norms;
    }

    const double s_hd_error = cellwise_errors.l2_norm();
    const double s_hd_norm = cellwise_norms.l2_norm();

    // Assemble convergence table
    const unsigned int n_active_cells=triangulation.n_active_cells();
    const unsigned int n_dofs=dof_handler.n_dofs();

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("Stress,L2", s_l2_error/s_l2_norm);
    convergence_table.add_value("Stress,Hdiv", s_hd_error/s_hd_norm); //sx_hd_error/sx_hd_norm
    convergence_table.add_value("Displ,L2", u_l2_error/u_l2_norm);
    convergence_table.add_value("Displ,L2mid", u_l2_mid_error/u_l2_mid_norm);
    convergence_table.add_value("Rotat,L2", p_l2_error/p_l2_norm);
  }


  // ElasticityProblem: Output results
  template <int dim>
  void ElasticityProblem<dim>::output_results(const unsigned int cycle, const unsigned int refine)
  {
    TimerOutput::Scope t(computing_timer, "Output results");
    std::vector<std::string> solution_names;
    std::string rhs_name = "rhs";

    switch(dim)
    {
      case 2:
        solution_names.insert(solution_names.end(), {"s11","s12","s21","s22","u","v","p"});
        break;

      case 3:
        solution_names.insert(solution_names.end(),
                              {"s11","s12","s13","s21","s22","s23","s31","s32","s33","u","v","w","p1","p2","p3"});
        break;

      default:
      Assert(false, ExcNotImplemented());
    }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(total_dim-1, DataComponentInterpretation::component_is_part_of_vector);

    switch (dim)
    {
      case 2:
        data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);
        break;

      case 3:
        data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
        break;

      default:
      Assert(false, ExcNotImplemented());
        break;
    }

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);

    data_out.build_patches ();

    std::ofstream output ("solution" + std::to_string(dim) + "d-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk (output);

    convergence_table.set_precision("Stress,L2", 3);
    convergence_table.set_precision("Stress,Hdiv", 3);
    convergence_table.set_precision("Displ,L2", 3);
    convergence_table.set_precision("Displ,L2mid", 3);
    convergence_table.set_precision("Rotat,L2", 3);
    convergence_table.set_scientific("Stress,L2", true);
    convergence_table.set_scientific("Stress,Hdiv", true);
    convergence_table.set_scientific("Displ,L2", true);
    convergence_table.set_scientific("Displ,L2mid", true);
    convergence_table.set_scientific("Rotat,L2", true);
    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("Stress,L2", "$ \\|\\sigma - \\sigma_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Stress,Hdiv", "$ \\|\\nabla\\cdot(\\sigma - \\sigma_h)\\|_{L^2} $");
    convergence_table.set_tex_caption("Displ,L2", "$ \\|u - u_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Displ,L2mid", "$ \\|Qu - u_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Rotat,L2", "$ \\|p - p_h\\|_{L^2} $");
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");

    convergence_table.evaluate_convergence_rates("Stress,L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Stress,Hdiv", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Displ,L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Displ,L2mid", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Rotat,L2", ConvergenceTable::reduction_rate_log2);

    std::ofstream error_table_file("error" + std::to_string(dim) + "d.tex");

    if (cycle == refine-1){
      convergence_table.write_text(std::cout);
      convergence_table.write_tex(error_table_file);
    }
  }


  // Explicit instantiations
  template class DarcyProblem<2>;
  template class DarcyProblem<3>;
  template class ElasticityProblem<2>;
  template class ElasticityProblem<3>;
}
