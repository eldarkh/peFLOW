// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2020 Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_UTILITIES_H
#define PEFLOW_UTILITIES_H

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <random>
#include <unordered_map>
#include <set>

namespace utilities
{
  using namespace dealii;

  template <class T> 
  int sgn(T x) {
    return (T(0) < x) - (x < T(0));
  }

  // For some reason this is needed in newer deal.II versions...
  template <int dim>
  inline void round_point(Point<dim> &p)
  {
    for (unsigned int d=0; d<dim; ++d)
    {
      if(p[d] < 0)
        p[d] = std::ceil(1.e8*p[d] - 0.5)/1.e8;

      p[d] = std::floor(1.e8*p[d] + 0.5)/1.e8;
    }
  }

  // Hash function for points as keys
  template <int dim>
  struct hash_points
  {
    size_t operator()(const Point<dim> &p) const
    {
      size_t h1 = std::hash<double>()(round(p[0]));
      size_t h2 = std::hash<double>()(round(p[1]));
      size_t h3;
      if (dim == 3)
        h3 = std::hash<double>()(round(p[2]));

      switch (dim)
      {
        case 2:
          return (h1 ^ h2);
        case 3:
          return (h1 ^ (h2 << 1)) ^ h3;
        default:
          Assert(false, ExcNotImplemented());
      }
    }
  };

  // Function to check if points are equal
  template <int dim>
  struct points_equal{
    bool operator()( const Point<dim>& lhs, const Point<dim>& rhs ) const{
      switch (dim) {
        case 2:
          return (lhs[0] == rhs[0]) && (lhs[1] == rhs[1]);
        case 3:
          return (lhs[0] == rhs[0]) && (lhs[1] == rhs[1]) && (lhs[2] == rhs[2]);
        default:
          Assert(false, ExcNotImplemented());
      }
    }
  };

  // Aliases for multipoint data structures
  template <int dim>
  using MapPointMatrix = std::unordered_map<Point<dim>, std::map<std::pair<types::global_dof_index,types::global_dof_index>, double>, hash_points<dim>, points_equal<dim>>;

  template <int dim>
  using MapPointVector = std::unordered_map<Point<dim>, std::map<types::global_dof_index, double>, hash_points<dim>, points_equal<dim>>;

  template <int dim>
  using MapPointSet = std::unordered_map<Point<dim>, std::set<types::global_dof_index>, hash_points<dim>, points_equal<dim>>;


  // Function to transform the grid given a map
  template <int dim>
  Point<dim> grid_transform (const Point<dim> &p)
  {
    switch (dim)
    {
      case 2:
        return Point<dim>(p[0], p[1]);
      case 3:
        return Point<dim>(p[0] + 0.03*cos(3*M_PI*p[0])*cos(3*M_PI*p[1])*cos(3*M_PI*p[2]),
                          p[1] - 0.04*cos(3*M_PI*p[0])*cos(3*M_PI*p[1])*cos(3*M_PI*p[2]),
                          p[2] + 0.05*cos(3*M_PI*p[0])*cos(3*M_PI*p[1])*cos(3*M_PI*p[2]));
      default:
        Assert(false, ExcNotImplemented());
    }
  }

  // Function to transform the grid given an h2 regular map
  template <int dim>
  Point<dim> grid_transform_h2 (const Point<dim> &p)
  {
    switch (dim)
    {
      case 2:
        return Point<dim>(p[0] + 0.1*sin(2.0*M_PI*p[0])*sin(2.0*M_PI*p[1]),
                          p[1] + 0.1*sin(2.0*M_PI*p[0])*sin(2.0*M_PI*p[1]));
      case 3:
        // Not really h2 regular
        return Point<dim>(p[0] + 0.03*cos(3*M_PI*p[0])*cos(3*M_PI*p[1])*cos(3*M_PI*p[2]),
                          p[1] - 0.04*cos(3*M_PI*p[0])*cos(3*M_PI*p[1])*cos(3*M_PI*p[2]),
                          p[2] + 0.05*cos(3*M_PI*p[0])*cos(3*M_PI*p[1])*cos(3*M_PI*p[2]));
      default:
        Assert(false, ExcNotImplemented());
    }
  }

  // Function to randomly perturb the grid h^k regularly
  template <int dim>
  Point<dim> grid_transform_unit_hk (const Point<dim> &p, 
                                     std::mt19937* const rand_engine_ptr, 
                                     const double k, 
                                     const double h)
  {
    std::uniform_real_distribution<> dist(-h/3.5, h/3.5); // divide by 3 here to keep quad shape and not cross borders

    switch (dim)
    {
      case 2:
      {
        double x, y;

        double rv1 = dist(*rand_engine_ptr);
        double rv2 = dist(*rand_engine_ptr);

        x = (p[0] < 1e-12 || p[0] > 1.0-1e-12) ? p[0] : p[0] + sgn(rv1) * std::pow(std::fabs(rv1), k);
        y = (p[1] < 1e-12 || p[1] > 1.0-1e-12) ? p[1] : p[1] + sgn(rv2) * std::pow(std::fabs(rv2), k);

        return Point<dim>(x,y);
      }
      case 3:
      {
        double x,y,z;

        double rv1 = dist(*rand_engine_ptr);
        double rv2 = dist(*rand_engine_ptr);
        double rv3 = dist(*rand_engine_ptr);

        x = (p[0] < 1e-12 || p[0] > 1.0-1e-12) ? p[0] : p[0] + sgn(rv1) * std::pow(std::fabs(rv1), k);
        y = (p[1] < 1e-12 || p[1] > 1.0-1e-12) ? p[1] : p[1] + sgn(rv2) * std::pow(std::fabs(rv2), k);
        z = (p[2] < 1e-12 || p[2] > 1.0-1e-12) ? p[2] : p[2] + sgn(rv3) * std::pow(std::fabs(rv3), k);

        return Point<dim>(x,y,z);
      }
      default:
        Assert(false, ExcNotImplemented());
    }
  }

  // Create rank 2 tensor from rank 1 tensors
  template <int dim>
  Tensor<2,dim> make_tensor(const std::vector<Tensor<1,dim> > &vec)
  {
    Tensor<2,dim> res;

    for(int row=0;row<dim;++row)
      for(int col=0;col<dim;++col)
        res[row][col] = vec[row][col];

    return res;
  }

  
  // Create asymmetry tensor from a rank-2 tensor
  template <int dim>
  Tensor<1,static_cast<int>(0.5*dim*(dim-1))> make_asymmetry_tensor(const Tensor<2,dim> &mat)
  {
    const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));
    Tensor<1,rotation_dim> res;
    
    switch(dim)
    {
      case 2:
        res[0] = mat[0][1] - mat[1][0];
        break;
      case 3:
        res[0] = mat[2][1] - mat[1][2];
        res[1] = mat[0][2] - mat[2][0];
        res[2] = mat[1][0] - mat[0][1];
        break;
      default:
        Assert(false, ExcNotImplemented());
    }

    return res;
  }


  // Create asymmetry tensor from a vector or rank-1 tensors
  template <int dim>
  Tensor<1,static_cast<int>(0.5*dim*(dim-1))> make_asymmetry_tensor(const std::vector<Tensor<1,dim>> &vec)
  {
    const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));
    Tensor<2,dim> mat = make_tensor(vec);
    Tensor<1,rotation_dim> res = make_asymmetry_tensor<dim>(mat);

    return res;
  }

  
  // Create compliance tensor for stress variables
  template <int dim>
  Tensor<2,dim> compliance_tensor_stress(const std::vector<Tensor<1,dim>> &vec, const double mu, const double lambda)
  {
    Tensor<2,dim> res;
    double trace=0.;

    for (unsigned int i=0;i<dim;++i)
      trace += vec[i][i];

    for (unsigned int row=0;row<dim;++row)
      for (unsigned int col=0;col<dim;++col)
        if (row == col)
          res[row][col] = 1/(2*mu)*(vec[row][col] - lambda/(2*mu + dim*lambda)*trace);
        else
          res[row][col] = 1/(2*mu)*(vec[row][col]);

    return res;
  }

  // Create compliance tensor for pressure variables
  template <int dim>
  Tensor<2,dim> compliance_tensor_pressure(const double &pres, const double mu, const double lambda)
  {
    Tensor<2,dim> res;

    for(unsigned int i=0;i<dim;i++)
      res[i][i] = pres/(2*mu+dim*lambda);

    return res;
  }


  // Invert an SPD matrix using CG
  inline void invert_spd(const FullMatrix<double> &A, FullMatrix<double> &X)
  {
    SolverControl solver_control (1000, 1e-12);
    SolverCG<> solver (solver_control);

    size_t n = A.n();

    if(n==1){
      X = IdentityMatrix(n);
    } else {
      Vector<double> b(n);
      Vector<double> x(n);
      FullMatrix<double> res(n,n);
      for(unsigned int i=0;i<n;++i)
      {
        b = 0;
        b[i] = 1;

        solver.solve (A, x, b, PreconditionIdentity());

        for(size_t row=0;row<n;++row)
          if (fabs(x[row]) > 1.e-12)
            res(row,i) = x[row];
      }

      X.copy_from(res);
    }
  }
}

#endif //PEFLOW_UTILITIES_H

