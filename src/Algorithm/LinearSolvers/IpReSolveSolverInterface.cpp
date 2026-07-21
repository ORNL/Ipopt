// Copyright (C) 2023 ORNL and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Slaven Peles, Maksudul Alam

#include "IpReSolveSolverInterface.hpp"
#include "IpoptConfig.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

namespace Ipopt
{
#if IPOPT_VERBOSITY > 0
static const Index dbg_verbosity = 0;
#endif

static_assert(
   std::is_same<Number, ReSolve::real_type>::value,
   "Ipopt and ReSolve must use the same floating-point type.");

static_assert(
   std::is_same<Index, ReSolve::index_type>::value,
   "Ipopt and ReSolve must use the same index type.");

ReSolveSolverInterface::ReSolveSolverInterface()
  : nonzeros_(0),
    initialized_(false),
    initialized_resolve_(false),
    ndim_(0),
    val_(NULL),
    numneg_(0),
    pivtol_changed_(false),
    re_factorize_(false),
    factorize_(false),
    method_(resolve_klu),
    n_iteration_(0),
    k_(1),
    pivot_tol_(0.1),
    ordering_(1),
    halt_if_singular_(false),
    rcond_val_(1e-128),
    use_rcond_(false),
    factor_by_t_(2),
    resolve_KLU_(NULL),
    workspace_CPU_(NULL),
    A_(NULL),
    vec_rhs_(NULL),
    vec_x_(NULL),
#ifdef RESOLVE_USE_GPU
    workspace_GPU_(NULL),
    resolve_Rf_(NULL),
# ifdef RESOLVE_USE_CUDA
    resolve_GLU_(NULL),
# endif
    GS_(NULL),
    resolve_preconditioner_(NULL),
    resolve_FGMRES_(NULL),
#endif
    matrix_handler_(NULL),
    vector_handler_(NULL)
{
  DBG_START_METH("ReSolveSolverInterface::ReSolveSolverInterface()", dbg_verbosity);
#ifdef RESOLVE_USE_GPU
  printf("Resolve with GPU.\n");
# ifdef RESOLVE_USE_CUDA
  printf("Resolve with CUDA.\n");
# else
  printf("Resolve with HIP.\n");
# endif
#else
  printf("Resolve with CPU. CUDA or HIP Unavailable.\n");
#endif
}

ReSolveSolverInterface::~ReSolveSolverInterface()
{
  DBG_START_METH("ReSolveSolverInterface::~ReSolveSolverInterface()", dbg_verbosity);
#ifdef RESOLVE_USE_GPU
  delete resolve_FGMRES_;
  delete resolve_preconditioner_;
  delete GS_;

# ifdef RESOLVE_USE_CUDA
  delete resolve_GLU_;
# endif
  delete resolve_Rf_;
#endif

  delete resolve_KLU_;

  delete matrix_handler_;
  delete vector_handler_;

#ifdef RESOLVE_USE_GPU
  delete workspace_GPU_;
#endif
  delete workspace_CPU_;

  delete vec_rhs_;
  delete vec_x_;
  delete A_;
  delete[] val_;
}

void ReSolveSolverInterface::RegisterOptions(SmartPtr<RegisteredOptions> roptions)
{
  std::vector<std::string> options;
  std::vector<std::string> descrs;

  options.push_back(resolve_klu);
  descrs.push_back("Use KLU");

#ifdef RESOLVE_USE_GPU
# ifdef RESOLVE_USE_CUDA
  options.push_back(resolve_glu);
  descrs.push_back("Use GLU");
# endif
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
                             "Number of initial KLU iterations before switching to the selected refactorization method. Must be at least 1.", //
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

  // printf("ReSolveSolverInterface::InitializeImpl is Called\n");

  options.GetNumericValue("resolve_tol", pivot_tol_, prefix);

  options.GetIntegerValue("resolve_ordering", ordering_, prefix);

  Index btf;
  options.GetIntegerValue("resolve_btf", btf, prefix);

  Index scale;
  options.GetIntegerValue("resolve_scale", scale, prefix);

  Index n_skip_refactoring;
  options.GetIntegerValue("resolve_n_skip_refactoring", n_skip_refactoring, prefix);

  // ReSolve's GPU refactorization methods require an initial KLU solve
  // to construct the factors and permutations used during setup.
  if (n_skip_refactoring < 1)
  {
    Jnlst().Printf(
      J_ERROR,
      J_LINEAR_ALGEBRA,
      "resolve_n_skip_refactoring must be at least 1.\n");
    return false;
  }

  k_ = n_skip_refactoring;

  options.GetBoolValue("resolve_halt_if_singular", halt_if_singular_, prefix);

  options.GetStringValue("resolve_method", method_, prefix);

  options.GetNumericValue("resolve_rcond_val", rcond_val_, prefix);
  options.GetBoolValue("resolve_use_rcond", use_rcond_, prefix);

  bool method_available = (method_ == resolve_klu);

#ifdef RESOLVE_USE_GPU
  method_available =
    method_available
    || method_ == resolve_rf
    || method_ == resolve_rf_fgmres;

# ifdef RESOLVE_USE_CUDA
  method_available =
    method_available
    || method_ == resolve_glu;
# endif
#endif

  if (!method_available)
  {
    Jnlst().Printf(
      J_ERROR,
      J_LINEAR_ALGEBRA,
      "ReSolve method '%s' is not available for this build.\n",
      method_.c_str());
    return false;
  }

  return true;
}

/** Initialize the local copy of the positions of the nonzero elements */
ESymSolverStatus ReSolveSolverInterface::InitializeStructure(Index dim, Index nonzeros, const Index* ia, const Index* ja)
{
  DBG_START_METH("ReSolveSolverInterface::InitializeStructure", dbg_verbosity);

  ESymSolverStatus retval = SYMSOLVER_SUCCESS;
  printf("ReSolveSolverInterface::InitializeStructure Called: dim: %d, nonzeros %d\n", dim, nonzeros);

  if (!initialized_resolve_)
  {
    resolve_KLU_ = new ReSolve::LinSolverDirectKLU();
    resolve_KLU_->setPivotThreshold(pivot_tol_);
    resolve_KLU_->setOrdering(static_cast<int>(ordering_));
    resolve_KLU_->setHaltIfSingular(halt_if_singular_);

    if (method_ == resolve_klu)
    {
      printf("Resolve::KLU Setup\n");
      workspace_CPU_ = new ReSolve::LinAlgWorkspaceCpu();
      matrix_handler_ = new ReSolve::MatrixHandler(workspace_CPU_);
      vector_handler_ = new ReSolve::VectorHandler(workspace_CPU_);
    }

#ifdef RESOLVE_USE_GPU
    else
    {
      printf("Resolve::GPU Setup\n");

      workspace_GPU_ = new workspace_type();
      workspace_GPU_->initializeHandles();

      matrix_handler_ = new ReSolve::MatrixHandler(workspace_GPU_);
      vector_handler_ = new ReSolve::VectorHandler(workspace_GPU_);

      if (method_ == resolve_rf || method_ == resolve_rf_fgmres)
      {
        printf("Resolve::RF Setup\n");
        resolve_Rf_ = new rf_solver(workspace_GPU_);
      }
# ifdef RESOLVE_USE_CUDA
      else if (method_ == resolve_glu)
      {
        printf("Resolve::GLU Setup\n");
        resolve_GLU_ =
          new ReSolve::LinSolverDirectCuSolverGLU(workspace_GPU_);
      }
# endif

      if (method_ == resolve_rf_fgmres)
      {
        printf("Resolve::FGMRES Setup\n");

        GS_ = new ReSolve::GramSchmidt(
          vector_handler_,
          ReSolve::GramSchmidt::CGS2);

        resolve_FGMRES_ = new ReSolve::LinSolverIterativeFGMRES(
          matrix_handler_,
          vector_handler_,
          GS_);

        resolve_preconditioner_ =
          new ReSolve::PreconditionerLU(resolve_Rf_);

        if (resolve_FGMRES_->setPreconditioner(
              resolve_preconditioner_) != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "Failed to attach the ReSolve RF preconditioner "
            "to FGMRES.\n");
          return SYMSOLVER_FATAL_ERROR;
        }
      }
    }
#endif
    initialized_resolve_ = true;
  }

  // Store size for later use
  ndim_ = dim;
  nonzeros_ = nonzeros;
  printf("Using Refactorization after %d iterations\n\n", k_);

  if (!initialized_)
  {
    A_ = new ReSolve::matrix::Csr(dim, dim, nonzeros);
    delete[] val_;
    val_ = new Number[nonzeros];

    // ReSolve borrows the matrix storage and does not take ownership.
    // Ipopt owns ia/ja, while this interface owns val_.
    if( A_->setDataPointers(const_cast<Index*>(ia), const_cast<Index*>(ja), val_, ReSolve::memory::HOST) != 0)
    {
      Jnlst().Printf(
         J_ERROR,
         J_LINEAR_ALGEBRA,
         "Failed to attach Ipopt matrix storage to ReSolve.\n");
      return SYMSOLVER_FATAL_ERROR;
    }
#ifdef RESOLVE_USE_GPU
     if( method_ != resolve_klu
         && A_->allocateMatrixData(ReSolve::memory::DEVICE) != 0 )
     {
        Jnlst().Printf(
           J_ERROR,
           J_LINEAR_ALGEBRA,
           "Failed to allocate ReSolve matrix device storage.\n");
        return SYMSOLVER_FATAL_ERROR;
     }
#endif

     if( resolve_KLU_->setup(A_) != 0 )
     {
        Jnlst().Printf(
           J_ERROR,
           J_LINEAR_ALGEBRA,
           "Failed to set up the ReSolve KLU solver.\n");
        return SYMSOLVER_FATAL_ERROR;
     }

     vec_rhs_ = new ReSolve::vector::Vector(A_->getNumRows());
     vec_x_ = new ReSolve::vector::Vector(A_->getNumRows());

     if( vec_rhs_->allocate(ReSolve::memory::HOST) != 0
         || vec_x_->allocate(ReSolve::memory::HOST) != 0 )
     {
        Jnlst().Printf(
           J_ERROR,
           J_LINEAR_ALGEBRA,
           "Failed to allocate ReSolve host vectors.\n");
        return SYMSOLVER_FATAL_ERROR;
     }

#ifdef RESOLVE_USE_GPU
     if( method_ != resolve_klu )
     {
        if( vec_rhs_->allocate(ReSolve::memory::DEVICE) != 0
            || vec_x_->allocate(ReSolve::memory::DEVICE) != 0 )
        {
          Jnlst().Printf(
             J_ERROR,
             J_LINEAR_ALGEBRA,
             "Failed to allocate ReSolve device vectors.\n");
          return SYMSOLVER_FATAL_ERROR;
        }
     }
#endif
  }
  factorize_ = true;
  n_iteration_ = 0;

  initialized_ = true;
  pivtol_changed_ = false;

  return retval;
}

ESymSolverStatus ReSolveSolverInterface::MultiSolve(bool new_matrix, const Index* ia, const Index* ja, Index nrhs, Number* rhs_vals, bool check_NegEVals, Index numberOfNegEVals)
{
  DBG_START_METH("ReSolveSolverInterface::MultiSolve", dbg_verbosity);

  std::vector<Number> solution_vals;

  if (nrhs > 1)
  {
    solution_vals.resize(
      static_cast<std::size_t>(nrhs) * static_cast<std::size_t>(ndim_));
  }

  int status;
  int status_refactor = 0;

  bool full_factor_done = false;

  (void)ia;
  (void)ja;

  // Ipopt updates val_ directly, so mark the host matrix current before
  // synchronizing updated values to the GPU.
  if (new_matrix)
  {
     if( A_->setUpdated(ReSolve::memory::HOST) != 0 )
     {
        Jnlst().Printf(
           J_ERROR,
           J_LINEAR_ALGEBRA,
           "Failed to mark the ReSolve host matrix as updated.\n");
        return SYMSOLVER_FATAL_ERROR;
     }

#ifdef RESOLVE_USE_GPU
     if( method_ != resolve_klu
         && A_->syncData(ReSolve::memory::DEVICE) != 0 )
     {
        Jnlst().Printf(
           J_ERROR,
           J_LINEAR_ALGEBRA,
           "Failed to synchronize the ReSolve matrix "
           "to the device.\n");
        return SYMSOLVER_FATAL_ERROR;
     }
#endif
  }

  // FACTORIZE

  // Every factor_by_t_ iteration do a Factorization!!!
  if (n_iteration_ % factor_by_t_ == 0)
  {
    // factorize_ = true;
    // re_factorize_ = true;
  }

  if( n_iteration_ == 0){
    printf("First Iteration: %d: Performing KLU Factorization\n", n_iteration_);
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
  }

  if (factorize_ && (new_matrix || re_factorize_))
  {
    // printf("Iteration: %d: Performing KLU Factorization\n", n_iteration_);

    //  perform the factorization
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().Start();
    }

    // First Factorization is always done by KLU
    // std::cout << "%" << n_iteration_ << "%" << "KLU FULL FACTORIZATION" << std::endl;
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
#ifdef RESOLVE_USE_CUDA
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

# ifdef RESOLVE_USE_CUDA
    // Actual Refactorize
    if (method_ == resolve_glu)
    {
      //std::cout << "%" << n_iteration_ << "%" << "GLU->refactorize()" << std::endl;
      status = resolve_GLU_->refactorize();
      if (status != 0)
      {
        Jnlst().Printf(
          J_ERROR,
          J_LINEAR_ALGEBRA,
          "ReSolve CUDA GLU refactorization failed with status %d.\n",
          status);
        return SYMSOLVER_FATAL_ERROR;
      }
    }
    else if (method_ == resolve_rf || method_ == resolve_rf_fgmres)
    {
      status_refactor = resolve_Rf_->refactorize();
      if (status_refactor != 0)
      {
        Jnlst().Printf(
           J_ERROR,
           J_LINEAR_ALGEBRA,
           "ReSolve CUDA RF refactorization failed with status %d.\n",
           status_refactor);
        return SYMSOLVER_FATAL_ERROR;
      }
    }
# elif defined(RESOLVE_USE_HIP)
    if (method_ == resolve_rf || method_ == resolve_rf_fgmres)
    {
      status = resolve_Rf_->refactorize();
      if (status != 0)
      {
        Jnlst().Printf(
           J_ERROR,
           J_LINEAR_ALGEBRA,
           "ReSolve HIP RF refactorization failed with status %d.\n",
          status);
        return SYMSOLVER_FATAL_ERROR;
      }
    }
#endif

    if (method_ == resolve_klu)
    {
      //std::cout << "%" << n_iteration_ << "%" << "KLU->refactorize()" << std::endl;
      status = resolve_KLU_->refactorize();
      if (status != 0)
      {
        Jnlst().Printf(
          J_ERROR,
          J_LINEAR_ALGEBRA,
          "ReSolve KLU refactorization failed with status %d.\n",
          status);
        return SYMSOLVER_FATAL_ERROR;
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
        //printf("RCond: %12.8e\n", rcond_val);
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

      for (Index irhs = 0; irhs < nrhs; ++irhs)
      {
        Number* rhs = rhs_vals + irhs * ndim_;
        // Copy the current RHS to ReSolve
        if (vec_rhs_->copyFromExternal(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST) != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "Failed to copy rhs data to ReSolve host storage.\n");
          return SYMSOLVER_FATAL_ERROR;
        }
        status = resolve_KLU_->solve(vec_rhs_, vec_x_);
        if (status != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "ReSolve KLU solve failed with status %d.\n",
            status);
          return SYMSOLVER_FATAL_ERROR;
        }
        if (nrhs > 1)
        {
          Number* solution = solution_vals.data() + irhs * ndim_;
          if (vec_x_->copyToExternal(solution, ReSolve::memory::HOST, ReSolve::memory::HOST) != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "Failed to copy the ReSolve solution to temporary storage.\n");
            return SYMSOLVER_FATAL_ERROR;
          }
        }
      }
    }

    // Setup RF here
    if (n_iteration_ == (k_ - 1))
    {
# ifdef RESOLVE_USE_CUDA
      if (method_ == resolve_rf || method_ == resolve_rf_fgmres)
      {
        printf("CUDA: Iteration: %d: Setting up %s\n", n_iteration_, method_.c_str());

        ReSolve::matrix::Sparse* L = resolve_KLU_->getLFactor();
        ReSolve::matrix::Sparse* U = resolve_KLU_->getUFactor();
        ReSolve::index_type* P = resolve_KLU_->getPOrdering();
        ReSolve::index_type* Q = resolve_KLU_->getQOrdering();

        if (L == nullptr || U == nullptr || P == nullptr || Q == nullptr)
        {
          Jnlst().Printf(
             J_ERROR,
             J_LINEAR_ALGEBRA,
             "Failed to obtain KLU factors or permutations for ReSolve CUDA RF setup.\n");
          return SYMSOLVER_FATAL_ERROR;
        }

        status = resolve_Rf_->setup(A_, L, U, P, Q);
        if (status != 0)
        {
          Jnlst().Printf(
             J_ERROR,
             J_LINEAR_ALGEBRA,
             "ReSolve CUDA RF setup failed with status %d.\n",
             status);
          return SYMSOLVER_FATAL_ERROR;
        }
      }

      if (method_ == resolve_rf_fgmres && resolve_FGMRES_->setup(A_) != 0)
      {
        Jnlst().Printf(
           J_ERROR,
           J_LINEAR_ALGEBRA,
           "Failed to set up ReSolve FGMRES.\n");
        return SYMSOLVER_FATAL_ERROR;
      }
# elif defined(RESOLVE_USE_HIP)
      if (method_ == resolve_rf || method_ == resolve_rf_fgmres)
      {
        printf("HIP: Iteration: %d: Setting up %s\n", n_iteration_, method_.c_str());
        ReSolve::matrix::Sparse* L = resolve_KLU_->getLFactor();
        ReSolve::matrix::Sparse* U = resolve_KLU_->getUFactor();
        ReSolve::index_type* P = resolve_KLU_->getPOrdering();
        ReSolve::index_type* Q = resolve_KLU_->getQOrdering();
        if (L == nullptr || U == nullptr || P == nullptr || Q == nullptr)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "Failed to obtain KLU factors or permutations for ReSolve HIP RF setup.\n");
          return SYMSOLVER_FATAL_ERROR;
        }

        if (vec_rhs_->copyFromExternal(
              rhs_vals,
              ReSolve::memory::HOST,
              ReSolve::memory::DEVICE)
            != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "Failed to copy rhs data to ReSolve device storage.\n");
          return SYMSOLVER_FATAL_ERROR;
        }

        status = resolve_Rf_->setup(A_, L, U, P, Q, vec_rhs_);
        if (status != 0)
        {
          Jnlst().Printf(
             J_ERROR,
             J_LINEAR_ALGEBRA,
             "ReSolve HIP RF setup failed with status %d.\n",
             status);
          return SYMSOLVER_FATAL_ERROR;
        }
      }

      if (method_ == resolve_rf_fgmres && resolve_FGMRES_->setup(A_) != 0)
      {
        Jnlst().Printf(
           J_ERROR,
           J_LINEAR_ALGEBRA,
           "Failed to set up ReSolve FGMRES.\n");
        return SYMSOLVER_FATAL_ERROR;
      }
# endif
    }
  }
  else // After k iteration only solve
  {
    // Every 10 iteration setup again with full factorization? Already factorization done.
# ifdef RESOLVE_USE_CUDA
    if (method_ == resolve_glu || method_ == resolve_rf || method_ == resolve_rf_fgmres)
    {
      for (Index irhs = 0; irhs < nrhs; ++irhs)
      {
        Number* rhs = rhs_vals + irhs * ndim_;
        // Copy the current RHS to ReSolve device storage
        if (vec_rhs_->copyFromExternal(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE) != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "Failed to copy rhs data to ReSolve device storage.\n");
          return SYMSOLVER_FATAL_ERROR;
        }

        if (method_ == resolve_glu)
        {
          status = resolve_GLU_->solve(vec_rhs_, vec_x_);
          if (status != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "ReSolve CUDA GLU solve failed with status %d.\n",
              status);
            return SYMSOLVER_FATAL_ERROR;
          }
        }
        else if (method_ == resolve_rf)
        {
          status = resolve_Rf_->solve(vec_rhs_, vec_x_);
          if (status != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "ReSolve CUDA RF solve failed with status %d.\n",
              status);
            return SYMSOLVER_FATAL_ERROR;
          }
        }
        else if (method_ == resolve_rf_fgmres)
        {
          status = resolve_Rf_->solve(vec_rhs_, vec_x_);
          if (status != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "ReSolve CUDA RF initial solve failed with status %d.\n",
              status);
            return SYMSOLVER_FATAL_ERROR;
          }

          status = resolve_FGMRES_->resetMatrix(A_);
          if (status != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "Failed to reset the ReSolve CUDA FGMRES matrix "
              "with status %d.\n",
              status);
            return SYMSOLVER_FATAL_ERROR;
          }

          status = resolve_FGMRES_->solve(vec_rhs_, vec_x_);
          if (status != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "ReSolve CUDA FGMRES solve failed with status %d.\n",
              status);
            return SYMSOLVER_FATAL_ERROR;
          }
        }

        // GPU solvers leave the current solution on the device; synchronize it
        // to host memory before copying the solution back to Ipopt.
        if (vec_x_->syncData(ReSolve::memory::HOST) != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "Failed to synchronize the Resolve solution to the host.\n");
          return SYMSOLVER_FATAL_ERROR;
        }
        if (nrhs > 1)
        {
          Number* solution = solution_vals.data() + irhs * ndim_;
          if (vec_x_->copyToExternal(solution, ReSolve::memory::HOST, ReSolve::memory::HOST) != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "Failed to copy the ReSolve solution to temporary storage.\n");
            return SYMSOLVER_FATAL_ERROR;
          }
        }
      }
      matrix_handler_->setValuesChanged(true, ReSolve::memory::DEVICE);
    }
# elif defined(RESOLVE_USE_HIP)
    if (method_ == resolve_rf || method_ == resolve_rf_fgmres)
    {
      for (Index irhs = 0; irhs < nrhs; ++irhs)
      {
        Number* rhs = rhs_vals + irhs * ndim_;
        // Copy the current RHS to ReSolve device storage
        if (vec_rhs_->copyFromExternal(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE) != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "Failed to copy rhs data to ReSolve device storage.\n");
          return SYMSOLVER_FATAL_ERROR;
        }

        status = resolve_Rf_->solve(vec_rhs_, vec_x_);
        if (status != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "ReSolve HIP RF solve failed with status %d.\n",
            status);
          return SYMSOLVER_FATAL_ERROR;
        }

        if (method_ == resolve_rf_fgmres)
        {
          status = resolve_FGMRES_->resetMatrix(A_);
          if (status != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "Failed to reset the ReSolve HIP FGMRES matrix "
              "with status %d.\n",
              status);
            return SYMSOLVER_FATAL_ERROR;
          }

          status = resolve_FGMRES_->solve(vec_rhs_, vec_x_);
          if (status != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "ReSolve HIP FGMRES solve failed with status %d.\n",
              status);
            return SYMSOLVER_FATAL_ERROR;
          }
        }

        // GPU solvers leave the current solution on the device; synchronize it
        // to host memory before copying the solution back to Ipopt.
        if (vec_x_->syncData(ReSolve::memory::HOST) != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "Failed to synchronize the Resolve solution to the host.\n");
          return SYMSOLVER_FATAL_ERROR;
        }
        if (nrhs > 1)
        {
          Number* solution = solution_vals.data() + irhs * ndim_;

          if (vec_x_->copyToExternal(
                solution,
                ReSolve::memory::HOST,
                ReSolve::memory::HOST) != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "Failed to copy the ReSolve solution to temporary storage.\n");
            return SYMSOLVER_FATAL_ERROR;
          }
        }
      }
    }
#endif

    if (method_ == resolve_klu)
    {
      // Solve using KLU
      if (use_rcond_)
      {
        Number rcond_val = resolve_KLU_->getMatrixConditionNumber();
        //printf("RCond: %12.8e\n", rcond_val);
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

      for (Index irhs = 0; irhs < nrhs; ++irhs)
      {
        Number* rhs = rhs_vals + irhs * ndim_;
        // Copy the current RHS to ReSolve host storage
        if (vec_rhs_->copyFromExternal(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST) != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "Failed to copy rhs data to ReSolve host storage.\n");
          return SYMSOLVER_FATAL_ERROR;
        }
        status = resolve_KLU_->solve(vec_rhs_, vec_x_);
        if (status != 0)
        {
          Jnlst().Printf(
            J_ERROR,
            J_LINEAR_ALGEBRA,
            "ReSolve KLU solve failed with status %d.\n",
            status);
          return SYMSOLVER_FATAL_ERROR;
        }
        if (nrhs > 1)
        {
          Number* solution = solution_vals.data() + irhs * ndim_;

          if (vec_x_->copyToExternal(
                solution,
                ReSolve::memory::HOST,
                ReSolve::memory::HOST) != 0)
          {
            Jnlst().Printf(
              J_ERROR,
              J_LINEAR_ALGEBRA,
              "Failed to copy the ReSolve solution to temporary storage.\n");
            return SYMSOLVER_FATAL_ERROR;
          }
        }
      }
    }
  }

  if (nrhs == 1)
  {
    if (vec_x_->copyToExternal(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::HOST) != 0)
    {
      Jnlst().Printf(
        J_ERROR,
        J_LINEAR_ALGEBRA,
        "Failed to copy the Resolve solution to Ipopt.\n");
      return SYMSOLVER_FATAL_ERROR;
    }
  }
  else
  {
    std::copy(
      solution_vals.begin(),
      solution_vals.end(),
      rhs_vals);
  }

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
  DBG_ASSERT(initialized_);

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
