// Copyright (C) 2023 ORNL and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Slaven Peles, Maksudul Alam

#include "IpoptConfig.h"
#include "IpReSolveSolverInterface.hpp"

#include <cmath>
#include <iostream>


namespace Ipopt
{
#if IPOPT_VERBOSITY > 0
static const Index dbg_verbosity = 0;
#endif

ReSolveSolverInterface::ReSolveSolverInterface() : 
   val_(NULL)
{
   DBG_START_METH("ReSolveSolverInterface::ReSolveSolverInterface()", dbg_verbosity);
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
  resolve_KLU_ = new ReSolve::LinSolverDirectKLU();
}

ReSolveSolverInterface::~ReSolveSolverInterface()
{
   DBG_START_METH("ReSolveSolverInterface::~ReSolveSolverInterface()", dbg_verbosity);
   delete[] val_;
}

void ReSolveSolverInterface::RegisterOptions(
   SmartPtr<RegisteredOptions> roptions
)
{   
   roptions->AddNumberOption(
      "resolve_tol",
      "Partial pivoting tolerance",
      0.001, 
      "If the diagonal entry has a magnitude greater than or equal to tol times the largest magnitude of entries in the pivot column, then the diagonal entry is chosen.",
      false);

   roptions->AddIntegerOption(
      "resolve_ordering",
      "Which fill-reducing ordering to use",
      0,
      "0 for AMD, 1 for COLAMD, 2 for a user-provided permutation P and Q (or a natural ordering if P and Q are NULL), or 3 for the user order function.",
      false);

   roptions->AddIntegerOption(
      "resolve_btf",
      "Use BTF",
      1,
      "if nonzero, then BTF is used to permute the input matrix into block upper triangular form.",
      false);

   roptions->AddIntegerOption(
      "resolve_scale",
      "Whether or not the matrix should be scaled",
      2,
      "If scale < 0, then no scaling is performed and the input matrix is not checked for errors. If scale >= 0, the input matrix is check for errors. If scale=0, then no scaling is performed. If scale=1, then each row of A is divided by the sum of the absolute values in that row. If scale=2, then each row of A is divided by the maximum absolute value in that row. Default: 2.",
      false);

   roptions->AddBoolOption(
      "resolve_halt_if_singular",
      "how to handle a singular matrix",
      false,
      "FALSE: keep going, TRUE: stop quickly.",
      false);
}

bool ReSolveSolverInterface::InitializeImpl(
   const OptionsList& options,
   const std::string& prefix
)
{
   printf("InitializeImpl! --ReSolve. This uses ReSolve Matrix Definitions.\n");
   
   /*
   Number tol;
   options.GetNumericValue("resolve_tol", tol, prefix);

   Index order_method;
   options.GetIntegerValue("resolve_ordering", order_method, prefix);

   Index btf;
   options.GetIntegerValue("resolve_btf", btf, prefix);

   Index scale;
   options.GetIntegerValue("resolve_scale", scale, prefix);

   bool halt_if_singular;
   options.GetBoolValue("resolve_halt_if_singular", halt_if_singular, prefix);
  */
   resolve_KLU_->setupParameters(1, 0.1, false);

   return true;
}

ESymSolverStatus ReSolveSolverInterface::MultiSolve(
   bool         new_matrix,
   const Index* ia,
   const Index* ja,
   Index        nrhs,
   Number*      rhs_vals,
   bool         check_NegEVals,
   Index        numberOfNegEVals
)
{
   DBG_START_METH("ReSolveSolverInterface::MultiSolve", dbg_verbosity);

#if 1
   printf("ReSolve MultiSolve Called\n");
#endif

#if 1
   if(factorize_){
      // perform the factorization
#if 1
      printf("klu_factor Getting Called\n");
#endif
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().Start();
      }
      printf("Factorize Steps 1\n");
	  int status  = resolve_KLU_->factorize();

      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().End();
      }

      if( status != 0 )
      {
         DBG_PRINT((1, "FACTORIZATION FAILED!\n"));
         return SYMSOLVER_FATAL_ERROR;  // Matrix singular or error occurred
      }

      factorize_ = false;
   }

   if( pivtol_changed_ )
   {
      DBG_PRINT((1, "Pivot tolerance has changed.\n"));
      pivtol_changed_ = false;
      // If the pivot tolerance has been changed but the matrix is not
      // new, we have to request the values for the matrix again to do
      // the factorization again.
      if( !new_matrix )
      {
         DBG_PRINT((1, "Ask caller to call again.\n"));
         refactorize_ = true;
         return SYMSOLVER_CALL_AGAIN;
      }
   }
#endif

   // check if a factorization has to be done
   DBG_PRINT((1, "new_matrix = %d\n", new_matrix));
   if( !first_iteration_ && (new_matrix || refactorize_) )
   {
      // perform the factorization
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().Start();
      }
	  //resolve_KLU_->refactorize();
	  int status = resolve_KLU_->refactorize();
      std::cout<<"KLU refactorization status: "<<status<<std::endl;      
      refactorize_ = false;
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().End();
      }
   }

   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemBackSolve().Start();
   }
   
   

   // Copy rhs_vals to vec_rhs cuda
   vec_rhs_->update(rhs_vals, "cpu", "cpu");
   int status = resolve_KLU_->solve(vec_rhs_, vec_x_);
   // Copy vec_x cuda to vec_x
   std::cout<<"KLU solve status: "<<status<<std::endl;  
   // copy vec_x to rhs_vals
   std::memcpy(rhs_vals, vec_x_->getData("cpu"), (ndim_) * sizeof(ReSolve::real_type));

   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemBackSolve().End();
   }

#if 0
   printf("ReSolve Solve Done\n");
#endif

    first_iteration_ = false;

   return SYMSOLVER_SUCCESS;
}

Number* ReSolveSolverInterface::GetValuesArrayPtr()
{
   DBG_START_METH("ReSolveSolverInterface::GetValuesArrayPtr", dbg_verbosity);
   DBG_ASSERT(initialized_);

   return A_->getValues("cpu");
}

/** Initialize the local copy of the positions of the nonzero elements */
ESymSolverStatus ReSolveSolverInterface::InitializeStructure(
   Index        dim,
   Index        nonzeros,
   const Index* ia,
   const Index* ja
)
{
   DBG_START_METH("ReSolveSolverInterface::InitializeStructure", dbg_verbosity);

   ESymSolverStatus retval = SYMSOLVER_SUCCESS;
   printf("dim: %d, nonzeros %d\n", dim, nonzeros);

   // Store size for later use
   ndim_ = dim;


  A_ = new ReSolve::MatrixCSR(dim, dim, nonzeros);
   if( val_ != NULL )
   {
      delete[] val_;
   }
   val_ = new Number[nonzeros];
   
  A_->setMatrixData(const_cast<int*>(ia), const_cast<int*>(ja), val_, "cpu");
  
  resolve_KLU_->setup(A_);
  
  vec_rhs_ = new ReSolve::Vector(A_->getNumRows());
  vec_x_ = new ReSolve::Vector(A_->getNumRows());
  
  vec_x_->allocate("cpu");//for KLU
  vec_x_->allocate("cuda");
  
   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
   }
   int status = resolve_KLU_->analyze();
   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemSymbolicFactorization().End();
   }   
   printf("ReSolve - Symbolic Factorization Done\n");
   
   factorize_ = true;
   first_iteration_ = true;


   if (status != 0){
      printf("Symbolic_ factorization crashed with Common_.status = %d \n", status);
      return SYMSOLVER_FATAL_ERROR;
   }

   initialized_ = true;

   return retval;
}



Index ReSolveSolverInterface::NumberOfNegEVals() const
{
   return numneg_;
}

bool ReSolveSolverInterface::IncreaseQuality()
{
   return true;
}

} // namespace Ipopt
