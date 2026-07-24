#include "IpOptionsList.hpp"
#include "IpReSolveSolverInterface.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace Ipopt;

namespace
{

bool check_solution(
   const std::string& name,
   const Number* actual,
   const Number* expected,
   Index n,
   Number tolerance = 1e-8)
{
   for( Index i = 0; i < n; ++i )
   {
      if( std::abs(actual[i] - expected[i]) > tolerance )
      {
         std::cerr << "FAIL: " << name
                   << " at index " << i
                   << ": expected " << expected[i]
                   << ", got " << actual[i] << '\n';
         return false;
      }
   }

   return true;
}

bool solve_succeeded(
   const std::string& name,
   ESymSolverStatus status)
{
   if( status != SYMSOLVER_SUCCESS )
   {
      std::cerr << "FAIL: " << name
                << " returned solver status " << status << '\n';
      return false;
   }

   return true;
}

bool configure_solver(
   ReSolveSolverInterface& solver,
   const std::string& method,
   const char* label)
{
   OptionsList options;

   if( !options.SetNumericValue("resolve_tol", 1e-3)
       || !options.SetIntegerValue("resolve_ordering", 0)
       || !options.SetIntegerValue("resolve_n_skip_refactoring", 1)
       || !options.SetStringValue("resolve_halt_if_singular", "no")
       || !options.SetStringValue("resolve_method", method)
       || !options.SetNumericValue("resolve_rcond_val", 1e-128)
       || !options.SetStringValue("resolve_use_rcond", "no") )
   {
      std::cerr << "FAIL: " << label << " options could not be set\n";
      return false;
   }

   if( !solver.InitializeImpl(options, "") )
   {
      std::cerr << "FAIL: " << label << " InitializeImpl failed\n";
      return false;
   }

   return true;
}

bool run_method(
   const std::string& method,
   const char* label)
{
   constexpr Index dim = 3;
   constexpr Index nnz = 7;

   // Full CSR, zero-based indexing:
   //
   // [ * * 0 ]
   // [ * * * ]
   // [ 0 * * ]
   const Index ia[dim + 1] = {0, 2, 5, 7};
   const Index ja[nnz] = {0, 1, 0, 1, 2, 1, 2};

   const Number A1[nnz] = {
      4.0, 1.0,
      1.0, 3.0, 1.0,
      1.0, 2.0
   };

   const Number A2[nnz] = {
      5.0, 1.0,
      1.0, 4.0, 1.0,
      1.0, 3.0
   };

   const Number x1[dim] = {1.0, 2.0, 3.0};
   const Number x2[dim] = {-1.0, 0.0, 2.0};

   const Number b1_A1[dim] = {6.0, 10.0, 8.0};
   const Number b2_A1[dim] = {-4.0, 1.0, 4.0};

   const Number b1_A2[dim] = {7.0, 12.0, 11.0};
   const Number b2_A2[dim] = {-5.0, 1.0, 6.0};

   ReSolveSolverInterface solver;

   if( !configure_solver(solver, method, label) )
   {
      return false;
   }

   if( solver.MatrixFormat()
       != SparseSymLinearSolverInterface::CSR_Full_Format_0_Offset )
   {
      std::cerr << "FAIL: " << label
                << " returned an unexpected matrix format\n";
      return false;
   }

   if( !solve_succeeded(
          std::string(label) + " InitializeStructure",
          solver.InitializeStructure(dim, nnz, ia, ja)) )
   {
      return false;
   }

   Number* values = solver.GetValuesArrayPtr();
   if( values == nullptr )
   {
      std::cerr << "FAIL: " << label
                << " GetValuesArrayPtr returned null\n";
      return false;
   }

   // nrhs = 1 baseline.
   std::copy(A1, A1 + nnz, values);

   Number rhs_single[dim];
   std::copy(b1_A1, b1_A1 + dim, rhs_single);

   if( !solve_succeeded(
          std::string(label) + " nrhs=1",
          solver.MultiSolve(true, ia, ja, 1, rhs_single, false, 0)) )
   {
      return false;
   }

   if( !check_solution(
          std::string(label) + " nrhs=1",
          rhs_single,
          x1,
          dim) )
   {
      return false;
   }

   // nrhs = 2 using the existing factorization.
   Number rhs_multiple[2 * dim] = {
      b1_A1[0], b1_A1[1], b1_A1[2],
      b2_A1[0], b2_A1[1], b2_A1[2]
   };

   if( !solve_succeeded(
          std::string(label) + " nrhs=2",
          solver.MultiSolve(false, ia, ja, 2, rhs_multiple, false, 0)) )
   {
      return false;
   }

   if( !check_solution(
          std::string(label) + " nrhs=2 rhs 1",
          rhs_multiple,
          x1,
          dim)
       || !check_solution(
          std::string(label) + " nrhs=2 rhs 2",
          rhs_multiple + dim,
          x2,
          dim) )
   {
      return false;
   }

   // Change numerical values while preserving the sparsity structure.
   std::copy(A2, A2 + nnz, values);

   Number rhs_updated[2 * dim] = {
      b1_A2[0], b1_A2[1], b1_A2[2],
      b2_A2[0], b2_A2[1], b2_A2[2]
   };

   if( !solve_succeeded(
          std::string(label) + " matrix update",
          solver.MultiSolve(true, ia, ja, 2, rhs_updated, false, 0)) )
   {
      return false;
   }

   if( !check_solution(
          std::string(label) + " matrix update rhs 1",
          rhs_updated,
          x1,
          dim)
       || !check_solution(
          std::string(label) + " matrix update rhs 2",
          rhs_updated + dim,
          x2,
          dim) )
   {
      return false;
   }

   // Reuse the updated factorization.
   Number rhs_repeat[dim];
   std::copy(b1_A2, b1_A2 + dim, rhs_repeat);

   if( !solve_succeeded(
          std::string(label) + " repeated solve",
          solver.MultiSolve(false, ia, ja, 1, rhs_repeat, false, 0)) )
   {
      return false;
   }

   if( !check_solution(
          std::string(label) + " repeated solve",
          rhs_repeat,
          x1,
          dim) )
   {
      return false;
   }

   if( solver.IncreaseQuality() )
   {
      std::cerr << "FAIL: " << label
                << " IncreaseQuality should return false\n";
      return false;
   }

   return true;
}

} // namespace

int main()
{
   std::cout << "Testing ReSolve Solver Interface...\n";

   if( !run_method(resolve_klu, "KLU") )
   {
      return 1;
   }

#ifdef RESOLVE_USE_GPU
# ifdef RESOLVE_USE_CUDA
   if( !run_method(resolve_glu, "GLU") )
   {
      return 1;
   }
# endif

   if( !run_method(resolve_rf, "RF") )
   {
      return 1;
   }

   if( !run_method(resolve_rf_fgmres, "RF-FGMRES") )
   {
      return 1;
   }
#endif

   std::cout << "    Test passed!\n";

   return 0;
}
