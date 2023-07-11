// Copyright (C) 2023 ORNL and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Slaven Peles, Maksudul Alam

#include "IpoptConfig.h"
#include "IpKLUSolverInterface.hpp"

#include <cmath>
#include <iostream>


namespace Ipopt
{
#if IPOPT_VERBOSITY > 0
static const Index dbg_verbosity = 0;
#endif

KLUSolverInterface::KLUSolverInterface() : 
   val_(NULL)
{
   DBG_START_METH("KLUSolverInterface::KLUSolverInterface()", dbg_verbosity);
}

KLUSolverInterface::~KLUSolverInterface()
{
   DBG_START_METH("KLUSolverInterface::~KLUSolverInterface()", dbg_verbosity);
   delete[] val_;
   klu_free_symbolic(&Symbolic_, &Common_);
   klu_free_numeric(&Numeric_, &Common_);
}

void KLUSolverInterface::RegisterOptions(
   SmartPtr<RegisteredOptions> roptions
)
{   
   roptions->AddNumberOption(
      "klu_tol",
      "Partial pivoting tolerance",
      0.001, 
      "If the diagonal entry has a magnitude greater than or equal to tol times the largest magnitude of entries in the pivot column, then the diagonal entry is chosen.",
      false);

   roptions->AddIntegerOption(
      "klu_ordering",
      "Which fill-reducing ordering to use",
      0,
      "0 for AMD, 1 for COLAMD, 2 for a user-provided permutation P and Q (or a natural ordering if P and Q are NULL), or 3 for the user order function.",
      false);

   roptions->AddIntegerOption(
      "klu_btf",
      "Use BTF",
      1,
      "if nonzero, then BTF is used to permute the input matrix into block upper triangular form.",
      false);

   roptions->AddIntegerOption(
      "klu_scale",
      "Whether or not the matrix should be scaled",
      2,
      "If scale < 0, then no scaling is performed and the input matrix is not checked for errors. If scale >= 0, the input matrix is check for errors. If scale=0, then no scaling is performed. If scale=1, then each row of A is divided by the sum of the absolute values in that row. If scale=2, then each row of A is divided by the maximum absolute value in that row. Default: 2.",
      false);

   roptions->AddBoolOption(
      "klu_halt_if_singular",
      "how to handle a singular matrix",
      false,
      "FALSE: keep going, TRUE: stop quickly.",
      false);
}

bool KLUSolverInterface::InitializeImpl(
   const OptionsList& options,
   const std::string& prefix
)
{
   printf("InitializeImpl! --KLU\n");

   klu_defaults(&Common_);

   Number tol;
   options.GetNumericValue("klu_tol", tol, prefix);
   Common_.tol = tol;
   printf("klu_tol: %f\n", Common_.tol);

   Index order_method;
   options.GetIntegerValue("klu_ordering", order_method, prefix);
   Common_.ordering = order_method;
   printf("klu_ordering: %d\n", Common_.ordering);

   Index btf;
   options.GetIntegerValue("klu_btf", btf, prefix);
   Common_.btf = btf;
   printf("klu_btf: %d\n", Common_.btf);

   Index scale;
   options.GetIntegerValue("klu_scale", scale, prefix);
   Common_.scale = scale;
   printf("klu_scale: %d\n", Common_.scale);

   bool halt_if_singular;
   options.GetBoolValue("klu_halt_if_singular", halt_if_singular, prefix);
   Common_.halt_if_singular = halt_if_singular;
   printf("klu_halt_if_singular: %d\n", Common_.halt_if_singular);

   return true;
}

ESymSolverStatus KLUSolverInterface::MultiSolve(
   bool         new_matrix,
   const Index* ia,
   const Index* ja,
   Index        nrhs,
   Number*      rhs_vals,
   bool         check_NegEVals,
   Index        numberOfNegEVals
)
{
   DBG_START_METH("KLUSolverInterface::MultiSolve", dbg_verbosity);

#if 0
   printf("KLU MultiSolve Called\n");
#endif

#if 1
   if(factorize_){
      // perform the factorization
#if 0
      printf("klu_factor Called\n");
#endif
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().Start();
      }
      Numeric_ = klu_factor(Ap_, Ai_, val_, Symbolic_, &Common_);
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().End();
      }
      if( Numeric_ == nullptr )
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

#if 1
   // check if a factorization has to be done
   DBG_PRINT((1, "new_matrix = %d\n", new_matrix));
   if( new_matrix || refactorize_ )
   {
#if 0
      printf("klu_refactor Called\n");
#endif
      // perform the factorization
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().Start();
      }
      klu_refactor(Ap_, Ai_, val_, Symbolic_, Numeric_, &Common_);
      refactorize_ = false;
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().End();
      }
   }
#endif

   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemBackSolve().Start();
   }
   klu_solve(Symbolic_, Numeric_, ndim_, nrhs, rhs_vals, &Common_);
   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemBackSolve().End();
   }

#if 0
   printf("KLU Solve Done\n");
#endif

   return SYMSOLVER_SUCCESS;
}

Number* KLUSolverInterface::GetValuesArrayPtr()
{
   DBG_START_METH("KLUSolverInterface::GetValuesArrayPtr", dbg_verbosity);
   DBG_ASSERT(initialized_);

   return val_;
}

/** Initialize the local copy of the positions of the nonzero elements */
ESymSolverStatus KLUSolverInterface::InitializeStructure(
   Index        dim,
   Index        nonzeros,
   const Index* ia,
   const Index* ja
)
{
   DBG_START_METH("KLUSolverInterface::InitializeStructure", dbg_verbosity);

   ESymSolverStatus retval = SYMSOLVER_SUCCESS;
   printf("dim: %d\n, nonzeros %d\n", dim, nonzeros);

   // Store size for later use
   ndim_ = dim;

#if 0
   Ap_ = new int[dim+1];
   Ai_ = new int[nonzeros];

   //Copy Structure
   //const_cast<int*> 
   for(int i=0;i<=dim;i++){Ap_[i]=ia[i];}
   for(int i=0;i<nonzeros;i++){Ai_[i]=ja[i];}
#else
   Ap_ = const_cast<int*>(ia);
   Ai_ = const_cast<int*>(ja);
#endif

   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
   }
   Symbolic_ = klu_analyze(ndim_, Ap_, Ai_, &Common_);
   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemSymbolicFactorization().End();
   }   
   printf("KLU - Symbolic Factorization Done\n");
   
   factorize_ = true;

   if (Symbolic_ == nullptr){
      printf("Symbolic_ factorization crashed with Common_.status = %d \n", Common_.status);
      return SYMSOLVER_FATAL_ERROR;
   }

#if 0
   printf("dim: %d\n, nonzeros %d\n", dim, nonzeros);

   for(int i=1; i<=dim; i++){
      printf("ia[%d]=(%d)\n", i, ia[i]);
   }

   for(int i=0; i< nonzeros; i++){
      printf("ja[%d]=(%d)\n", i, ja[i]);
   }

   for(int i=0; i<dim; i++){
      int n = ia[i+1] - ia[i];
	  printf("%3d -- %3d:\t", i, n);
      for(int j=0; j<n; j++){
         printf("[%3d]=%3d\t", ia[i]+j, ja[ia[i]+j]);
      }
      printf("\n------------------------------------\n");
   }
#endif
   // Setup memory for values
   if( val_ != NULL )
   {
      delete[] val_;
   }
   val_ = new Number[nonzeros];

   initialized_ = true;

   return retval;
}



Index KLUSolverInterface::NumberOfNegEVals() const
{
   return numneg_;
}

bool KLUSolverInterface::IncreaseQuality()
{
   return true;
}

} // namespace Ipopt
