// Copyright (C) 2005, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Michael Hagemann               Univ of Basel 2005-10-28
//               original version (based on MA27TSolverInterface.cpp)

#include "IpoptConfig.h"
#include "IpKLUSolverInterface.hpp"

#include <cmath>
#include <iostream>

#ifdef IPOPT_HAS_HSL
#include "CoinHslConfig.h"
#endif


/** MA57 functions from HSL library (symbols resolved at linktime) */

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
}

bool KLUSolverInterface::InitializeImpl(
   const OptionsList& options,
   const std::string& prefix
)
{
   printf("InitializeImpl! --KLU\n");
  
   klu_defaults(&Common_);

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

   printf("KLU MultiSolve Called\n");

   Symbolic_ = klu_analyze(ndim_, Ap_, Ai_, &Common_);
   Numeric_ = klu_factor(Ap_, Ai_, val_, Symbolic_, &Common_);
   klu_solve(Symbolic_, Numeric_, ndim_, nrhs, rhs_vals, &Common_);

   printf("KLU Solve Done\n");

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
   Ap_ = new int[dim+1];
   Ai_ = new int[nonzeros];

   //Copy Structure
   for(int i=0;i<=dim;i++){Ap_[i]=ia[i];}
   for(int i=0;i<nonzeros;i++){Ai_[i]=ja[i];}

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
