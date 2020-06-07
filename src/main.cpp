// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2018 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#include <deal.II/base/multithread_info.h>
#include "../inc/problem.h"
#include "../inc/biot_mfe.h"
#include "../inc/biot_parameter_reader.h"
#include "../inc/darcy_parameter_reader.h"
#include "../inc/elasticity_parameter_reader.h"
#include "../inc/darcy_mfe.h"
#include "../inc/darcy_mfmfe.h"
#include "../inc/elasticity_mfe.h"
#include "../inc/elasticity_msmfe.h"

#include <memory>
#include <iostream>

/*
 * Main driver function. Chooses the model and dimension.
 * Then reads parameters from the file. The parameter files
 * should be in the same directory as CMakeLists.txt
 */
int main()
{
  try {
    using namespace dealii;
    using namespace peflow;
    using namespace darcy;
    using namespace elasticity;
    using namespace biot;

    MultithreadInfo::set_thread_limit(8);

    std::ofstream log_file("log_file.txt");
    deallog.attach(log_file);
    deallog.depth_file(2);

    unsigned int model, dim;
    std::cout << "=========================================" <<std::endl;
    std::cout << "Choose the model to run: " << std::endl;
    std::cout << "  1: Mixed Darcy Problem" <<std::endl;
    std::cout << "  2: Multipoint Mixed Darcy Problem" <<std::endl;
    std::cout << "  3: Mixed Elasticity Problem" <<std::endl;
    std::cout << "  4: Multipoint Mixed Elasticity Problem" <<std::endl;
    std::cout << "  5: Mixed Biot Problem" <<std::endl;

    std::cin >> model;
    std::cout << "=========================================" <<std::endl;
    std::cout << "Specify dimension: " << std::endl;
    std::cin >> dim;

    ParameterHandler prm;
    std::unique_ptr<Problem<2>> problem2d;
    std::unique_ptr<Problem<3>> problem3d;

    if(model == 1)
    {
      DarcyParameterReader   param(prm);
      param.read_parameters("parameters_darcy.prm");

      // Get parameters
      const unsigned int degree = prm.get_integer("degree");
      const unsigned int grid  = prm.get_integer("grid_flag");
      const unsigned int refinements = prm.get_integer("refinements");

      switch(dim)
      {
        case 2:
          std::cout << "Mixed Darcy, 2D case: " << std::endl;
          problem2d = std::make_unique<DarcyMFE<2>>(degree, prm);
          problem2d->run(refinements, grid);
          break;
        case 3:
          std::cout << "Mixed Darcy, 3D case: " << std::endl;
          problem3d = std::make_unique<DarcyMFE<3>>(degree, prm);
          problem3d->run(refinements, grid);
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    }
    else if(model == 2)
    {
      DarcyParameterReader param(prm);
      param.read_parameters("parameters_darcy.prm");

      // Get parameters
      const unsigned int degree = prm.get_integer("degree");
      const unsigned int grid  = prm.get_integer("grid_flag");
      const unsigned int refinements = prm.get_integer("refinements");

      switch(dim)
      {
        case 2:
          std::cout << "Multipoint Mixed Darcy, 2D case: " << std::endl;
          problem2d = std::make_unique<MultipointMixedDarcyProblem<2>>(degree, prm);
          problem2d->run(refinements, grid);
          break;
        case 3:
          std::cout << "Multipoint Mixed Darcy, 3D case: " << std::endl;
          problem3d = std::make_unique<MultipointMixedDarcyProblem<3>>(degree, prm);
          problem3d->run(refinements, grid);
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    }
    else if(model == 3)
    {
      ElasticityParameterReader param(prm);
      param.read_parameters("parameters_elasticity.prm");

      // Get parameters
      const unsigned int degree = prm.get_integer("degree");
      const unsigned int grid  = prm.get_integer("grid_flag");
      const unsigned int refinements = prm.get_integer("refinements");

      switch(dim)
      {
        case 2:
          std::cout << "Mixed Linear Elasticity, 2D case: " << std::endl;
          problem2d = std::make_unique<MixedElasticityProblem<2>>(degree, prm);
          problem2d->run(refinements, grid);
          break;
        case 3:
          std::cout << "Mixed Linear Elasticity, 3D case: " << std::endl;
          problem3d = std::make_unique<MixedElasticityProblem<3>>(degree, prm);
          problem3d->run(refinements, grid);
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    }
    else if(model == 4)
    {
      ElasticityParameterReader param(prm);
      param.read_parameters("parameters_elasticity_dsc.prm");

      // Get parameters
      const unsigned int degree = prm.get_integer("degree");
      const unsigned int grid  = prm.get_integer("grid_flag");
      const unsigned int refinements = prm.get_integer("refinements");

      switch(dim)
      {
        case 2:
        {
          std::cout << "Multipoint Mixed Linear Elasticity, 2D case: " << std::endl;
          
          // ther rest needs to be done this way too, don't remember what the plab was
          // with abstract base and ptrs
          MultipointMixedElasticityProblem<2> msmfe2d(degree, prm);
          msmfe2d.set_scaled_rotation(true);
          msmfe2d.run(refinements, grid);
          break;
        }
        case 3:
          std::cout << "Multipoint Mixed Linear Elasticity, 3D case: " << std::endl;
          problem3d = std::make_unique<MultipointMixedElasticityProblem<3>>(degree, prm);
          problem3d->run(refinements, grid);
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    }
    else if(model == 5)
    {
      BiotParameterReader param(prm);
      param.read_parameters("parameters_biot.prm");

      // Get parameters
      const unsigned int degree = prm.get_integer("degree");
      const unsigned int grid = prm.get_integer("grid_flag");
      const unsigned int refinements = prm.get_integer("refinements");
      const double time_step = prm.get_double("time_step");
      const unsigned int num_time_steps = prm.get_integer("num_time_steps");

      switch(dim)
      {
        case 2:
          std::cout << "Mixed Biot, 2D case: " << std::endl;
          problem2d = std::make_unique<MixedBiotProblem<2>>(degree, prm, time_step, num_time_steps);
          problem2d->run(refinements, grid);
          break;
        case 3:
          std::cout << "Mixed Biot, 3D case: " << std::endl;
          Assert(false, ExcNotImplemented());
          //NOTE: it is implemented but not tested...
          problem3d = std::make_unique<MixedBiotProblem<3>>(degree, prm, time_step, num_time_steps);
          problem3d->run(refinements, grid);
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    }

  } catch (std::exception &exc) {
    std::cerr << std::endl << std:: endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  } catch(...) {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }

  return 0;
}
