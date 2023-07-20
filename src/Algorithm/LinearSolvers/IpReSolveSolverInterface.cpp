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

bool ReSolveSolverInterface::InitializeImpl(
   const OptionsList& options,
   const std::string& prefix
)
{
   printf("InitializeImpl! --ReSolve. This uses ReSolve Matrix Definitions.\n");
   
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
	  resolve_KLU_->factorize();
	  //printf("Factorize Steps 2\n");
	  //ReSolve::Matrix* L = resolve_KLU_->getLFactor();
	  //printf("Factorize Steps 3\n");
      //ReSolve::Matrix* U = resolve_KLU_->getUFactor();
	  //printf("Factorize Steps 4\n");
	  //if (L == nullptr) {printf("ERROR");}

      //ReSolve::index_type* P = resolve_KLU_->getPOrdering();
	  //printf("Factorize Steps 5\n");
      //ReSolve::index_type* Q = resolve_KLU_->getQOrdering();
	  //printf("Factorize Steps 6\n");
      //resolve_GLU_->setup(A_, L, U, P, Q); 
	  //printf("Factorize Steps 7\n");

      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().End();
      }
/*
      if( Numeric_ == nullptr )
      {
         DBG_PRINT((1, "FACTORIZATION FAILED!\n"));
         return SYMSOLVER_FATAL_ERROR;  // Matrix singular or error occurred
      }
*/
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
	  //resolve_KLU_->refactorize();
	  int status = resolve_KLU_->refactorize();
      std::cout<<"KLU refactorization status: "<<status<<std::endl;      
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
   
   
	printf("vec_rhs_[cpu]=%p\n", (void *)vec_rhs_->getData("cpu"));
	printf("vec_rhs_[cuda]=%p\n", (void *)vec_rhs_->getData("cuda"));
	printf("vec_x_[cpu]=%p\n", (void *)vec_x_->getData("cpu"));
	printf("vec_x_[cuda]=%p\n", (void *)vec_x_->getData("cuda"));   
	printf("rhs_vals=%p\n", (void *)rhs_vals);   
   
 
   printf("BEFORE: \n");  
   for(int i=0; i<10; i++){
        printf("%d=%f\t", i, rhs_vals[i]);
   }
   printf("\n");
   

  //vec_rhs[0] = 1;
  //vec_rhs[1] = 2;
   
   
   
   // Copy rhs_vals to vec_rhs cuda
   vec_rhs_->update(rhs_vals, "cpu", "cpu");
   int status = resolve_KLU_->solve(vec_rhs_, vec_x_);
   // Copy vec_x cuda to vec_x
   std::cout<<"GLU solve status: "<<status<<std::endl;  
   //vec_x_->update(vec_x_->getData("cuda"), "cuda", "cpu");
   // copy vec_x to rhs_vals
   std::memcpy(rhs_vals, vec_x_->getData("cpu"), (ndim_) * sizeof(ReSolve::real_type));

   printf("AFTER: \n");  
   for(int i=0; i<10; i++){
        printf("%d=%f\t", i, rhs_vals[i]);
   }
   printf("\n");

   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemBackSolve().End();
   }

#if 0
   printf("ReSolve Solve Done\n");
#endif

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
   resolve_KLU_->analyze();
   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemSymbolicFactorization().End();
   }   
   printf("ReSolve - Symbolic Factorization Done\n");
   
   factorize_ = true;

/*
   if (Symbolic_ == nullptr){
      printf("Symbolic_ factorization crashed with Common_.status = %d \n", Common_.status);
      return SYMSOLVER_FATAL_ERROR;
   }
*/
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
