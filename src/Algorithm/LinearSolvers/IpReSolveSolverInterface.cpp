// Copyright (C) 2023 ORNL and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Slaven Peles, Maksudul Alam

#include "IpReSolveSolverInterface.hpp"
#include "IpoptConfig.h"

#include <cmath>
#include <iostream>

namespace Ipopt
{
#if IPOPT_VERBOSITY > 0
static const Index dbg_verbosity = 0;
#endif

ReSolveSolverInterface::ReSolveSolverInterface() : val_(NULL)
{
  DBG_START_METH("ReSolveSolverInterface::ReSolveSolverInterface()", dbg_verbosity);
  rcond_val_ = 1e-128;
  factor_by_t_ = 2;

#if RESOLVE_WITH_CUDA
  printf("Resolve with CUDA\n");
#elif RESOLVE_WITH_HIP
  printf("Resolve with HIP\n");
#else
  printf("Resolve with CPU. CUDA or HIP Unavailable\n");
#endif
}

ReSolveSolverInterface::~ReSolveSolverInterface()
{
  DBG_START_METH("ReSolveSolverInterface::~ReSolveSolverInterface()", dbg_verbosity);
  delete[] val_;
  val_ = nullptr;

#if RESOLVE_WITH_CUDA
  delete workspace_CUDA_;
  delete matrix_handler_;
  delete vector_handler_;
  delete resolve_KLU_;
  if ( method_ == resolve_glu) 
  {
    delete resolve_GLU_;
  }
  else if ( method_ == resolve_rf) 
  {
    delete resolve_Rf_;
  }
  else if ( method_ == resolve_rf_fgmres) 
  {
    delete resolve_Rf_;
    delete GS_;
    delete resolve_FGMRES_;
  }
#endif

#if RESOLVE_WITH_HIP
  delete workspace_HIP_;
  delete matrix_handler_;
  delete vector_handler_;
  delete resolve_KLU_;
  if (method_ == resolve_rf)
  {
    delete resolve_Rf_;
  }
  else if (method_ == resolve_rf_fgmres)
  {
    delete resolve_Rf_;
    delete GS_;
    delete resolve_FGMRES_;
  }
#endif

  delete vec_rhs_;
  delete vec_x_;
  delete A_;
}

void ReSolveSolverInterface::RegisterOptions(SmartPtr<RegisteredOptions> roptions)
{
  std::vector<std::string> options;
  std::vector<std::string> descrs;

  options.push_back(resolve_klu);
  descrs.push_back("Use KLU");
#if RESOLVE_WITH_CUDA
  options.push_back(resolve_glu);
  descrs.push_back("Use GLU");

  options.push_back(resolve_rf);
  descrs.push_back("Use RF");

  options.push_back(resolve_rf_fgmres);
  descrs.push_back("Use FGMRES");
#endif

#if RESOLVE_WITH_HIP
  options.push_back(resolve_rf);
  descrs.push_back("Use RF");

  options.push_back(resolve_rf_fgmres);
  descrs.push_back("Use FGMRES");
#endif

  roptions->AddStringOption("resolve_method",                               //
                            "Indicates which linear solver should be used", //
                            "klu",                                          //
                            options,                                        //
                            descrs,                                         //
                            "This is experimental and does not work well.", //
                            true);

  roptions->AddNumberOption("resolve_tol",                //
                            "Partial pivoting tolerance", //
                            0.1,                          // Default is 0.1 in ReSolve
                            "If the diagonal entry has a magnitude greater than or equal to tol times the largest "
                            "magnitude of entries in the pivot column, then the diagonal entry is chosen.",
                            false);

  roptions->AddIntegerOption("resolve_ordering",                    //
                             "Which fill-reducing ordering to use", //
                             1,                                     // Default is 1 in ReSolve
                             "0 for AMD, 1 for COLAMD, 2 for a user-provided permutation P and Q (or a natural "
                             "ordering if P and Q are NULL), or 3 for the user order function.",
                             false);

  roptions->AddIntegerOption("resolve_btf",                                                                                //
                             "Use BTF",                                                                                    //
                             1,                                                                                            //
                             "if nonzero, then BTF is used to permute the input matrix into block upper triangular form.", //
                             false);

  roptions->AddIntegerOption("resolve_scale",                              //
                             "Whether or not the matrix should be scaled", //
                             2,                                            //
                             "If scale < 0, then no scaling is performed and the input matrix is not checked for errors. If scale >= 0, the input matrix is check for errors. If scale=0, then no scaling is performed. If scale=1, then each row of A is "
                             "divided by the sum of the absolute values in that row. If scale=2, then each row of A is divided by the maximum absolute value in that row. Default: 2.", //
                             false);

  roptions->AddBoolOption("resolve_halt_if_singular",        //
                          "how to handle a singular matrix", //
                          false,                             // Default is False in ReSolve
                          "FALSE: keep going, TRUE: stop quickly.", false);

  roptions->AddIntegerOption("resolve_n_skip_refactoring",                      //
                             "How many iterations to skip refactoring",         //
                             1,                                                 //
                             "Integer, Start Refactoring after k-th iteration", //
                             false);

  roptions->AddBoolOption("resolve_use_rcond", //
                          "If you use rcond",  //
                          false,               // Default is False in ReSolve
                          "FALSE: don't use rcond, TRUE: use rcond.", false);

  roptions->AddNumberOption("resolve_rcond_val",       //
                            "ReSolve KLU RCond Value", //
                            1e-128,                    //
                            "RCond Value to initiate KLU Factorize Again", false);
}

bool ReSolveSolverInterface::InitializeImpl(const OptionsList& options, const std::string& prefix)
{
  Number tol;
  options.GetNumericValue("resolve_tol", tol, prefix);

  Index order_method;
  options.GetIntegerValue("resolve_ordering", order_method, prefix);

  Index btf;
  options.GetIntegerValue("resolve_btf", btf, prefix);

  Index scale;
  options.GetIntegerValue("resolve_scale", scale, prefix);

  Index n_skip_refactoring;
  options.GetIntegerValue("resolve_n_skip_refactoring", n_skip_refactoring, prefix);
  k_ = n_skip_refactoring;

  bool halt_if_singular;
  options.GetBoolValue("resolve_halt_if_singular", halt_if_singular, prefix);

  std::string method;
  options.GetStringValue("resolve_method", method, prefix);
  method_ = method;

  options.GetNumericValue("resolve_rcond_val", rcond_val_, prefix);
  options.GetBoolValue("resolve_use_rcond", use_rcond_, prefix);

#if RESOLVE_WITH_CUDA
  if (method_ == resolve_glu)
  {
    workspace_CUDA_ = new ReSolve::LinAlgWorkspaceCUDA();
    workspace_CUDA_->initializeHandles();

    matrix_handler_ = new ReSolve::MatrixHandler(workspace_CUDA_);
    vector_handler_ = new ReSolve::VectorHandler(workspace_CUDA_);

    resolve_KLU_ = new ReSolve::LinSolverDirectKLU();
    resolve_GLU_ = new ReSolve::LinSolverDirectCuSolverGLU(workspace_CUDA_);
  }
  else if (method_ == resolve_rf)
  {
    workspace_CUDA_ = new ReSolve::LinAlgWorkspaceCUDA;
    workspace_CUDA_->initializeHandles();

    matrix_handler_ = new ReSolve::MatrixHandler(workspace_CUDA_);
    vector_handler_ = new ReSolve::VectorHandler(workspace_CUDA_);

    resolve_KLU_ = new ReSolve::LinSolverDirectKLU;
    resolve_Rf_ = new ReSolve::LinSolverDirectCuSolverRf();
  }
  else if (method_ == resolve_rf_fgmres)
  {
    workspace_CUDA_ = new ReSolve::LinAlgWorkspaceCUDA;
    workspace_CUDA_->initializeHandles();

    matrix_handler_ = new ReSolve::MatrixHandler(workspace_CUDA_);
    vector_handler_ = new ReSolve::VectorHandler(workspace_CUDA_);

    resolve_KLU_ = new ReSolve::LinSolverDirectKLU;
    resolve_Rf_ = new ReSolve::LinSolverDirectCuSolverRf;
    GS_ = new ReSolve::GramSchmidt(vector_handler_, ReSolve::GramSchmidt::cgs2);
    resolve_FGMRES_ = new ReSolve::LinSolverIterativeFGMRES(matrix_handler_, vector_handler_, GS_);
  }
#endif

#if RESOLVE_WITH_HIP
  if (method_ == resolve_rf)
  {
    workspace_HIP_ = new ReSolve::LinAlgWorkspaceHIP();
    workspace_HIP_->initializeHandles();

    matrix_handler_ = new ReSolve::MatrixHandler(workspace_HIP_);
    vector_handler_ = new ReSolve::VectorHandler(workspace_HIP_);

    resolve_KLU_ = new ReSolve::LinSolverDirectKLU();
    resolve_Rf_ = new ReSolve::LinSolverDirectRocSolverRf(workspace_HIP_);
  }
  else if (method_ == resolve_rf_fgmres)
  {
    workspace_HIP_ = new ReSolve::LinAlgWorkspaceHIP();
    workspace_HIP_->initializeHandles();

    matrix_handler_ = new ReSolve::MatrixHandler(workspace_HIP_);
    vector_handler_ = new ReSolve::VectorHandler(workspace_HIP_);

    resolve_KLU_ = new ReSolve::LinSolverDirectKLU();
    resolve_Rf_ = new ReSolve::LinSolverDirectRocSolverRf(workspace_HIP_);

    GS_ = new ReSolve::GramSchmidt(vector_handler_, ReSolve::GramSchmidt::cgs2);
    resolve_FGMRES_ = new ReSolve::LinSolverIterativeFGMRES(matrix_handler_, vector_handler_, GS_);
  }
#endif

  if (method_ == resolve_klu)
  {
    workspace_CPU_ = new ReSolve::LinAlgWorkspaceCpu();
    matrix_handler_ = new ReSolve::MatrixHandler(workspace_CPU_);
    vector_handler_ = new ReSolve::VectorHandler(workspace_CPU_);

    resolve_KLU_ = new ReSolve::LinSolverDirectKLU;
  }

  return true;
}

/** Initialize the local copy of the positions of the nonzero elements */
ESymSolverStatus ReSolveSolverInterface::InitializeStructure(Index dim, Index nonzeros, const Index* ia, const Index* ja)
{
  DBG_START_METH("ReSolveSolverInterface::InitializeStructure", dbg_verbosity);

  ESymSolverStatus retval = SYMSOLVER_SUCCESS;
  printf("dim: %d, nonzeros %d\n", dim, nonzeros);

  // Store size for later use
  ndim_ = dim;
  nonzeros_ = nonzeros;
  printf("Using Refactorization after %d iterations\n\n", k_);

  A_ = new ReSolve::matrix::Csr(dim, dim, nonzeros);
  if (val_ != NULL)
  {
    delete[] val_;
  }
  val_ = new Number[nonzeros];

  A_->setMatrixData(const_cast<int*>(ia), const_cast<int*>(ja), val_, ReSolve::memory::HOST);
  resolve_KLU_->setup(A_);

  vec_rhs_ = new ReSolve::vector::Vector(A_->getNumRows());
  vec_x_ = new ReSolve::vector::Vector(A_->getNumRows());

  vec_x_->allocate(ReSolve::memory::HOST); // for KLU
  // vec_x_->allocate(ReSolve::memory::DEVICE);

  factorize_ = true;
  n_iteration_ = 0;

  initialized_ = true;
  pivtol_changed_ = false;

  return retval;
}

ESymSolverStatus ReSolveSolverInterface::MultiSolve(bool new_matrix, const Index* ia, const Index* ja, Index nrhs, Number* rhs_vals, bool check_NegEVals, Index numberOfNegEVals)
{
  DBG_START_METH("ReSolveSolverInterface::MultiSolve", dbg_verbosity);

  int status;
  int status_refactor = 0;

  bool full_factor_done = false;

  // Get Data from CPU and update the A Matrix
  A_->updateData(A_->getRowData(ReSolve::memory::HOST), A_->getColData(ReSolve::memory::HOST), A_->getValues(ReSolve::memory::HOST), ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  // FACTORIZE

  // Every factor_by_t_ iteration do a Factorization!!!
  if (n_iteration_ % factor_by_t_ == 0)
  {
    // factorize_ = true;
    // re_factorize_ = true;
  }

  if (factorize_ && (new_matrix || re_factorize_))
  {
    // printf("Iteration: %d: Performing KLU Factorization\n", n_iteration_);

    // Symbolic Factorization
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
    }
    status = resolve_KLU_->analyze();
    if (status != 0)
    {
      printf("Symbolic_ factorization crashed with Common_.status = %d \n", status);
      printf("%s:0 Singular\n", __func__);
      return SYMSOLVER_SINGULAR;
    }
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemSymbolicFactorization().End();
    }

    //  perform the factorization
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().Start();
    }

    // First Factorization is always done by KLU
    std::cout << "%" << n_iteration_ << "%" << "FULL FACTORIZATIOM" << std::endl;
    status = resolve_KLU_->factorize();
    full_factor_done = true;

    if (status != 0)
    {
      DBG_PRINT((1, "FACTORIZATION FAILED!\n"));
      printf("%s: Singular\n", __func__);
      return SYMSOLVER_SINGULAR; // Matrix singular or error occurred
    }
    // printf("Iteration: %d: Done KLU Factorization\n", n_iteration_);

    // GLU can be setup as early as possible
    if (n_iteration_ == k_ - 1)
    {
#if RESOLVE_WITH_CUDA
      if (method_ == resolve_glu)
      {
        printf("Iteration: %d: Setting Up GLU\n", n_iteration_);

        ReSolve::matrix::Sparse* L = resolve_KLU_->getLFactor();
        ReSolve::matrix::Sparse* U = resolve_KLU_->getUFactor();
        if (L == nullptr)
        {
          printf("ERROR");
        }
        ReSolve::index_type* P = resolve_KLU_->getPOrdering();
        ReSolve::index_type* Q = resolve_KLU_->getQOrdering();
        resolve_GLU_->setup(A_, L, U, P, Q);
      }
#endif
    }

    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().End();
    }

    // Iteration is 0 indexed. If the n_iteration_ is > 0
    // Stop doing factorization after iteration _k
    factorize_ = (n_iteration_ >= (k_ - 1)) ? false : true;
    re_factorize_ = false;
    // printf("Iteration: %d: Ending Factorization Section\n", n_iteration_);
  }

  if (pivtol_changed_)
  {
    DBG_PRINT((1, "Pivot tolerance has changed.\n"));
    pivtol_changed_ = false;
    // If the pivot tolerance has been changed but the matrix is not
    // new, we have to request the values for the matrix again to do
    // the factorization again.
    if (!new_matrix)
    {
      DBG_PRINT((1, "Ask caller to call again.\n"));
      factorize_ = true;
      return SYMSOLVER_CALL_AGAIN;
    }
  }

  // REFACTORIZE
  // check if a re-factorization has to be done
  DBG_PRINT((1, "new_matrix = %d\n", new_matrix));
  if (!full_factor_done && n_iteration_ >= k_ && (new_matrix || re_factorize_))
  {
    // perform the factorization
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().Start();
    }

#if RESOLVE_WITH_CUDA
    // Actual Refactorize
    if (method_ == resolve_glu)
    {
      status = resolve_GLU_->refactorize();
      if (status != 0)
      {
        std::cout << "CUSOLVER GLU refactorization status: " << status << std::endl;
      }
    }
    else if (method_ == resolve_rf)
    {
      status_refactor = resolve_Rf_->refactorize();
      if (status != 0)
      {
        std::cout << "CUSOLVER RF refactorization status: " << status_refactor << std::endl;
      }
    }
    else if (method_ == resolve_rf_fgmres)
    {
      status = resolve_Rf_->refactorize();
      if (status != 0)
      {
        std::cout << "CUSOLVER RF refactorization status: " << status << std::endl;
      }
    }
#endif

#if RESOLVE_WITH_HIP
    if (method_ == resolve_rf)
    {
      int status = resolve_Rf_->refactorize();
      if (status != 0)
      {
        std::cout << "ROCOLVER RF refactorization status: " << status << std::endl;
      }
    }
    else if (method_ == resolve_rf_fgmres)
    {
      int status = resolve_Rf_->refactorize();
      if (status != 0)
      {
        std::cout << "ROCSOLVER RF refactorization status: " << status << std::endl;
      }
    }
#endif

    if (method_ == resolve_klu)
    {
      std::cout << "%" << n_iteration_ << "%" << "RE-FACTORIZATIOM" << std::endl;
      status = resolve_KLU_->refactorize();
      if (status != 0)
      {
        std::cout << "KLU refactorization status: " << status << std::endl;
      }
    }
    re_factorize_ = false;
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().End();
    }
  } // End Refactorize

  // SOLVE
  if (HaveIpData())
  {
    IpData().TimingStats().LinearSystemBackSolve().Start();
  }

  // First k Iterations, only do KLU
  if (n_iteration_ < k_)
  {
    // USE KLU for up to k_ iterations
    {
      if (use_rcond_)
      {
        Number rcond_val = resolve_KLU_->getMatrixConditionNumber();
        printf("RCond: %12.8e\n", rcond_val);
        if (rcond_val < rcond_val_)
        {
          if (full_factor_done)
          {
            printf("%s:1 Singular\n", __func__);
            return SYMSOLVER_SINGULAR;
          }
          else
          {
            // refactor effectively failed -- need to call again
            // and do full factorization
            factorize_ = true;
            re_factorize_ = true;
            printf("%s:1 Need to do full factorization again.\n", __func__);
            DBG_PRINT((1, "Ask caller to call again.\n"))
            return SYMSOLVER_CALL_AGAIN;
          }
        }
      }

      // Copy rhs_vals to vec_rhs
      vec_rhs_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::HOST);
      status = resolve_KLU_->solve(vec_rhs_, vec_x_);
      if (status != 0)
      {
        std::cout << "KLU solve status: " << status << std::endl;
      }
    }

    // Setup RF here
    if (n_iteration_ == (k_ - 1))
    {
#if RESOLVE_WITH_CUDA
      if (method_ == resolve_rf)
      {
        printf("Iteration: %d: Setting up %s\n", n_iteration_, method_.c_str());
        ReSolve::matrix::Csc* L_csc = (ReSolve::matrix::Csc*)resolve_KLU_->getLFactor();
        ReSolve::matrix::Csc* U_csc = (ReSolve::matrix::Csc*)resolve_KLU_->getUFactor();
        ReSolve::matrix::Csr* L = new ReSolve::matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
        ReSolve::matrix::Csr* U = new ReSolve::matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
        matrix_handler_->csc2csr(L_csc, L, ReSolve::memory::DEVICE);
        matrix_handler_->csc2csr(U_csc, U, ReSolve::memory::DEVICE);
        if (L == nullptr)
        {
          printf("ERROR");
        }
        ReSolve::index_type* P = resolve_KLU_->getPOrdering();
        ReSolve::index_type* Q = resolve_KLU_->getQOrdering();
        resolve_Rf_->setup(A_, L, U, P, Q);

        delete L;
        delete U;
      }
      else if (method_ == resolve_rf_fgmres)
      {
        printf("Iteration: %d: Setting up %s\n", n_iteration_, method_.c_str());

        ReSolve::matrix::Csc* L_csc = (ReSolve::matrix::Csc*)resolve_KLU_->getLFactor();
        ReSolve::matrix::Csc* U_csc = (ReSolve::matrix::Csc*)resolve_KLU_->getUFactor();
        ReSolve::matrix::Csr* L = new ReSolve::matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
        ReSolve::matrix::Csr* U = new ReSolve::matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
        matrix_handler_->csc2csr(L_csc, L, ReSolve::memory::DEVICE);
        matrix_handler_->csc2csr(U_csc, U, ReSolve::memory::DEVICE);
        if (L == nullptr)
        {
          printf("ERROR");
        }
        ReSolve::index_type* P = resolve_KLU_->getPOrdering();
        ReSolve::index_type* Q = resolve_KLU_->getQOrdering();
        resolve_Rf_->setup(A_, L, U, P, Q);
        resolve_FGMRES_->setup(A_);
        resolve_FGMRES_->setupPreconditioner("CuSolverRf", resolve_Rf_);
        // _resolve_FGMRES->resetMatrix(A_);

        delete L;
        delete U;
      }
#endif

#if RESOLVE_WITH_HIP
      if (method_ == resolve_rf)
      {
        ReSolve::matrix::Csc* L = (ReSolve::matrix::Csc*)resolve_KLU_->getLFactor();
        ReSolve::matrix::Csc* U = (ReSolve::matrix::Csc*)resolve_KLU_->getUFactor();
        ReSolve::index_type* P = resolve_KLU_->getPOrdering();
        ReSolve::index_type* Q = resolve_KLU_->getQOrdering();
        vec_rhs_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
        resolve_Rf_->setup(A_, L, U, P, Q, vec_rhs_);
      }
      else if (method_ == resolve_rf_fgmres)
      {
        ReSolve::matrix::Csc* L = (ReSolve::matrix::Csc*)resolve_KLU_->getLFactor();
        ReSolve::matrix::Csc* U = (ReSolve::matrix::Csc*)resolve_KLU_->getUFactor();
        ReSolve::index_type* P = resolve_KLU_->getPOrdering();
        ReSolve::index_type* Q = resolve_KLU_->getQOrdering();
        vec_rhs_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
        resolve_Rf_->setSolveMode(1);
        resolve_Rf_->setup(A_, L, U, P, Q, vec_rhs_);
        std::cout << "about to set FGMRES" << std::endl;
        GS_->setup(A_->getNumRows(), resolve_FGMRES_->getRestart());
        resolve_FGMRES_->setup(A_);
        resolve_FGMRES_->setupPreconditioner("LU", resolve_Rf_);
      }
#endif
    }
  }
  else // After k iteration only solve
  {

    // Every 10 iteration setup again with full factorization? Already factorization done.

#if RESOLVE_WITH_CUDA
    if (method_ == resolve_glu)
    {
      // Copy rhs_vals to vec_rhs cuda
      vec_rhs_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      status = resolve_GLU_->solve(vec_rhs_, vec_x_);
      if (status != 0)
      {
        std::cout << "GLU solve status: " << status << std::endl;
      }
      // Copy vec_x cuda to vec_x in cpu
      vec_x_->update(vec_x_->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::HOST);
    }
    else if (method_ == resolve_rf)
    {
      // Copy rhs_vals to vec_rhs cuda
      vec_rhs_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      status = resolve_Rf_->solve(vec_rhs_, vec_x_);
      if (status != 0)
      {
        std::cout << "RF solve status: " << status << std::endl;
      }
      // Copy vec_x cuda to vec_x in cpu
      vec_x_->update(vec_x_->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::HOST);

      // Copy original RHS values to _vec_r
      // vec_r_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      matrix_handler_->setValuesChanged(true, ReSolve::memory::DEVICE);
    }
    else if (method_ == resolve_rf_fgmres)
    {
      // Copy rhs_vals to vec_rhs cuda
      vec_rhs_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      int status = resolve_Rf_->solve(vec_rhs_, vec_x_);
      if (status != 0)
      {
        std::cout << "RF solve status: " << status << std::endl;
      }

      resolve_FGMRES_->resetMatrix(A_);
      status = resolve_FGMRES_->solve(vec_rhs_, vec_x_);
      if (status != 0)
      {
        std::cout << "RF_FGMRES solve status: " << status << std::endl;
      }
      // Copy vec_x cuda to vec_x in cpu
      vec_x_->update(vec_x_->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::HOST);
    }
#endif

#if RESOLVE_WITH_HIP
    if (method_ == resolve_rf)
    {
      // Copy rhs_vals to vec_rhs cuda
      vec_rhs_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      int status = resolve_Rf_->solve(vec_rhs_, vec_x_);
      if (status != 0)
      {
        std::cout << "RF solve status: " << status << std::endl;
      }
      // Copy vec_x cuda to vec_x in cpu
      vec_x_->update(vec_x_->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::HOST);
    }
    else if (method_ == resolve_rf_fgmres)
    {
      // Copy rhs_vals to vec_rhs cuda
      vec_rhs_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      int status = resolve_Rf_->solve(vec_rhs_, vec_x_);
      if (status != 0)
      {
        std::cout << "RF solve status: " << status << std::endl;
      }

      resolve_FGMRES_->resetMatrix(A_);
      vec_rhs_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      status = resolve_FGMRES_->solve(vec_rhs_, vec_x_);
      if (status != 0)
      {
        std::cout << "RF_FGMRES solve status: " << status << std::endl;
      }
      // Copy vec_x cuda to vec_x in cpu
      vec_x_->update(vec_x_->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::HOST);
    }
#endif

    if (method_ == resolve_klu)
    {
      // Solve using KLU
      if (use_rcond_)
      {
        Number rcond_val = resolve_KLU_->getMatrixConditionNumber();
        printf("RCond: %12.8e\n", rcond_val);
        if (rcond_val < rcond_val_)
        {
          if (full_factor_done)
          {
            printf("%s:2 Singular\n", __func__);
            return SYMSOLVER_SINGULAR;
          }
          else
          {
            // refactor effectively failed -- need to call again
            // and do full factorization
            factorize_ = true;
            re_factorize_ = true;
            printf("Need to do full factorization again.\n");
            DBG_PRINT((1, "Ask caller to call again.\n"))
            return SYMSOLVER_CALL_AGAIN;
          }
        }
      }

      // Copy rhs_vals to vec_rhs cuda
      vec_rhs_->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::HOST);
      int status = resolve_KLU_->solve(vec_rhs_, vec_x_);
      if (status != 0)
      {
        std::cout << "KLU solve status: " << status << std::endl;
      }
    }
  }

  // copy vec_x to rhs_vals
  memcpy(rhs_vals, vec_x_->getData(ReSolve::memory::HOST), (ndim_) * sizeof(ReSolve::real_type));

  if (HaveIpData())
  {
    IpData().TimingStats().LinearSystemBackSolve().End();
  }

  n_iteration_ += 1;

  return SYMSOLVER_SUCCESS;
}

Number* ReSolveSolverInterface::GetValuesArrayPtr()
{
  DBG_START_METH("ReSolveSolverInterface::GetValuesArrayPtr", dbg_verbosity);
  DBG_ASSERT(_initialized);

  return A_->getValues(ReSolve::memory::HOST);
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
