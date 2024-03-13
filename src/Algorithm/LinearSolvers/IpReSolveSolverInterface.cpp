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

ReSolveSolverInterface::ReSolveSolverInterface() : _val(NULL)
{
  DBG_START_METH("ReSolveSolverInterface::ReSolveSolverInterface()", dbg_verbosity);
}

ReSolveSolverInterface::~ReSolveSolverInterface()
{
  DBG_START_METH("ReSolveSolverInterface::~ReSolveSolverInterface()", dbg_verbosity);
  delete[] _val;
}

void ReSolveSolverInterface::RegisterOptions(SmartPtr<RegisteredOptions> roptions)
{
  std::vector<std::string> options;
  std::vector<std::string> descrs;

  options.push_back(resolve_klu);
  descrs.push_back("Use KLU");

  options.push_back(resolve_glu);
  descrs.push_back("Use GLU");

  options.push_back(resolve_rf);
  descrs.push_back("Use RF");

  options.push_back(resolve_rf_fgmres);
  descrs.push_back("Use FGMRES");

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

  // roptions->AddIntegerOption(
  //     "resolve_btf", "Use BTF", 1,
  //     "if nonzero, then BTF is used to permute the input matrix into block upper triangular form.", false);

  // roptions->AddIntegerOption(
  //     "resolve_scale", "Whether or not the matrix should be scaled", 2,
  //     "If scale < 0, then no scaling is performed and the input matrix is not checked for errors. If scale >= 0,
  //     the " "input matrix is check for errors. If scale=0, then no scaling is performed. If scale=1, then each row
  //     of A is " "divided by the sum of the absolute values in that row. If scale=2, then each row of A is divided
  //     by the " "maximum absolute value in that row. Default: 2.", false);

  roptions->AddBoolOption("resolve_halt_if_singular",        //
                          "how to handle a singular matrix", //
                          false,                             // Default is False in ReSolve
                          "FALSE: keep going, TRUE: stop quickly.", false);
}

bool ReSolveSolverInterface::InitializeImpl(const OptionsList& options, const std::string& prefix)
{
  Number tol;
  options.GetNumericValue("resolve_tol", tol, prefix);

  Index order_method;
  options.GetIntegerValue("resolve_ordering", order_method, prefix);

  // Index btf;
  // options.GetIntegerValue("resolve_btf", btf, prefix);

  // Index scale;
  // options.GetIntegerValue("resolve_scale", scale, prefix);

  bool halt_if_singular;
  options.GetBoolValue("resolve_halt_if_singular", halt_if_singular, prefix);

  std::string method;
  options.GetStringValue("resolve_method", method, prefix);
  _method = method;

  _resolve_KLU = new ReSolve::LinSolverDirectKLU();
  //_resolve_KLU->setupParameters(order_method, tol, halt_if_singular);

  if (_method == resolve_glu)
  {
    _workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA();
    _workspace_CUDA->initializeHandles();
    _resolve_GLU = new ReSolve::LinSolverDirectCuSolverGLU(_workspace_CUDA);
  }
  else if (_method == resolve_rf)
  {
    _workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA();
    _workspace_CUDA->initializeHandles();
    _resolve_Rf = new ReSolve::LinSolverDirectCuSolverRf();
    _matrix_handler = new ReSolve::MatrixHandler(_workspace_CUDA);
    _vector_handler = new ReSolve::VectorHandler(_workspace_CUDA);
  }
  else if (_method == resolve_rf_fgmres)
  {
    _workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA();
    _workspace_CUDA->initializeHandles();
    _resolve_Rf = new ReSolve::LinSolverDirectCuSolverRf();
    _matrix_handler = new ReSolve::MatrixHandler(_workspace_CUDA);
    _vector_handler = new ReSolve::VectorHandler(_workspace_CUDA);
    _GS = new ReSolve::GramSchmidt(_vector_handler, ReSolve::GramSchmidt::cgs2);
    _resolve_FGMRES = new ReSolve::LinSolverIterativeFGMRES(_matrix_handler, _vector_handler, _GS);
  }

  return true;
}

/** Initialize the local copy of the positions of the nonzero elements */
ESymSolverStatus ReSolveSolverInterface::InitializeStructure(Index dim, Index nonzeros, const Index* ia, const Index* ja)
{
  DBG_START_METH("ReSolveSolverInterface::InitializeStructure", dbg_verbosity);

  ESymSolverStatus retval = SYMSOLVER_SUCCESS;
  // printf("dim: %d, nonzeros %d\n", dim, nonzeros);

  // Store size for later use
  _ndim = dim;
  _nonzeros = nonzeros;

  _A = new ReSolve::matrix::Csr(dim, dim, nonzeros);
  if (_val != NULL)
  {
    delete[] _val;
  }
  _val = new Number[nonzeros];

  _A->setMatrixData(const_cast<int*>(ia), const_cast<int*>(ja), _val, ReSolve::memory::HOST);

  _resolve_KLU->setup(_A);

  _vec_rhs = new ReSolve::vector::Vector(_A->getNumRows());
  _vec_x = new ReSolve::vector::Vector(_A->getNumRows());

  _vec_x->allocate(ReSolve::memory::HOST); // for KLU
  _vec_x->allocate(ReSolve::memory::DEVICE);

  if (HaveIpData())
  {
    IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
  }
  int status = _resolve_KLU->analyze();
  if (HaveIpData())
  {
    IpData().TimingStats().LinearSystemSymbolicFactorization().End();
  }

  _factorize = true;
  _first_iteration = true;
  _n_iteration = 0;

  if (status != 0)
  {
    printf("Symbolic_ factorization crashed with Common_.status = %d \n", status);
    return SYMSOLVER_FATAL_ERROR;
  }

  _initialized = true;

  return retval;
}

ESymSolverStatus ReSolveSolverInterface::MultiSolve(bool new_matrix, const Index* ia, const Index* ja, Index nrhs, Number* rhs_vals, bool check_NegEVals, Index numberOfNegEVals)
{
  DBG_START_METH("ReSolveSolverInterface::MultiSolve", dbg_verbosity);

  _A->updateData(_A->getRowData(ReSolve::memory::HOST), _A->getColData(ReSolve::memory::HOST), _A->getValues(ReSolve::memory::HOST), ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  if (_factorize)
  {
    // perform the factorization
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().Start();
    }

    // First Factorization is always done by KLU
    int status = _resolve_KLU->factorize();
    if (status != 0)
    {
      if (HaveIpData())
      {
        IpData().TimingStats().LinearSystemFactorization().End();
      }

      DBG_PRINT((1, "FACTORIZATION FAILED!\n"));
      return SYMSOLVER_FATAL_ERROR; // Matrix singular or error occurred
    }

    // GLU can be setup as early as possible
    if (_method == resolve_glu)
    {
      ReSolve::matrix::Sparse* L = _resolve_KLU->getLFactor();
      ReSolve::matrix::Sparse* U = _resolve_KLU->getUFactor();
      if (L == nullptr)
      {
        printf("ERROR");
      }
      ReSolve::index_type* P = _resolve_KLU->getPOrdering();
      ReSolve::index_type* Q = _resolve_KLU->getQOrdering();
      _resolve_GLU->setup(_A, L, U, P, Q);

      delete[] P;
      delete[] Q;
      delete L;
      delete U;
    }

    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().End();
    }

    _factorize = false;
  }

  if (_pivtol_changed)
  {
    DBG_PRINT((1, "Pivot tolerance has changed.\n"));
    _pivtol_changed = false;
    // If the pivot tolerance has been changed but the matrix is not
    // new, we have to request the values for the matrix again to do
    // the factorization again.
    if (!new_matrix)
    {
      DBG_PRINT((1, "Ask caller to call again.\n"));
      _re_factorize = true;
      return SYMSOLVER_CALL_AGAIN;
    }
  }

  // check if a re-factorization has to be done
  DBG_PRINT((1, "new_matrix = %d\n", new_matrix));
  if (!_first_iteration && (new_matrix || _re_factorize))
  {
    // perform the factorization
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().Start();
    }

    // Actual Refactorize
    if (_method == resolve_glu)
    {
      int status = _resolve_GLU->refactorize();
      if (status != 0)
      {
        std::cout << "CUSOLVER GLU refactorization status: " << status << std::endl;
      }
    }
    else if (_method == resolve_rf)
    {
      int status = _resolve_Rf->refactorize();
      if (status != 0)
      {
        std::cout << "CUSOLVER RF refactorization status: " << status << std::endl;
      }
    }
    else if (_method == resolve_rf_fgmres)
    {
      int status = _resolve_Rf->refactorize();
      if (status != 0)
      {
        std::cout << "CUSOLVER RF refactorization status: " << status << std::endl;
      }
    }
    else
    {
      int status = _resolve_KLU->refactorize();
      if (status != 0)
      {
        std::cout << "KLU refactorization status: " << status << std::endl;
      }
    }
    _re_factorize = false;
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().End();
    }
  }

  if (HaveIpData())
  {
    IpData().TimingStats().LinearSystemBackSolve().Start();
  }

  // First Iteration
  if (_n_iteration == 0)
  {
    if (_method == resolve_glu)
    {
      // Copy rhs_vals to vec_rhs cuda
      _vec_rhs->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      int status = _resolve_GLU->solve(_vec_rhs, _vec_x);
      if (status != 0)
      {
        std::cout << "GLU solve status: " << status << std::endl;
      }
      // Copy vec_x cuda to vec_x in cpu
      _vec_x->update(_vec_x->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::HOST);
    }
    else // USE KLU
    {
      // Copy rhs_vals to vec_rhs cuda
      _vec_rhs->update(rhs_vals, ReSolve::memory::DEVICE, ReSolve::memory::DEVICE);
      int status = _resolve_KLU->solve(_vec_rhs, _vec_x);
      if (status != 0)
      {
        std::cout << "KLU solve status: " << status << std::endl;
      }
    }

    // Setup RF here
    printf("Iteration: %d: Setting up %s\n", _n_iteration, _method.c_str());
    if (_method == resolve_rf)
    {
      ReSolve::matrix::Csc* L_csc = (ReSolve::matrix::Csc*)_resolve_KLU->getLFactor();
      ReSolve::matrix::Csc* U_csc = (ReSolve::matrix::Csc*)_resolve_KLU->getUFactor();
      ReSolve::matrix::Csr* L = new ReSolve::matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
      ReSolve::matrix::Csr* U = new ReSolve::matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
      _matrix_handler->csc2csr(L_csc, L, ReSolve::memory::DEVICE);
      _matrix_handler->csc2csr(U_csc, U, ReSolve::memory::DEVICE);
      if (L == nullptr)
      {
        printf("ERROR");
      }
      ReSolve::index_type* P = _resolve_KLU->getPOrdering();
      ReSolve::index_type* Q = _resolve_KLU->getQOrdering();
      _resolve_Rf->setup(_A, L, U, P, Q);

      delete[] P;
      delete[] Q;
      delete L;
      delete L_csc;
      delete U;
      delete U_csc;
    }
    else if (_method == resolve_rf_fgmres)
    {
      ReSolve::matrix::Csc* L_csc = (ReSolve::matrix::Csc*)_resolve_KLU->getLFactor();
      ReSolve::matrix::Csc* U_csc = (ReSolve::matrix::Csc*)_resolve_KLU->getUFactor();
      ReSolve::matrix::Csr* L = new ReSolve::matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
      ReSolve::matrix::Csr* U = new ReSolve::matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
      _matrix_handler->csc2csr(L_csc, L, ReSolve::memory::DEVICE);
      _matrix_handler->csc2csr(U_csc, U, ReSolve::memory::DEVICE);
      if (L == nullptr)
      {
        printf("ERROR");
      }
      ReSolve::index_type* P = _resolve_KLU->getPOrdering();
      ReSolve::index_type* Q = _resolve_KLU->getQOrdering();
      _resolve_Rf->setup(_A, L, U, P, Q);
      _resolve_FGMRES->setup(_A);
      _resolve_FGMRES->setupPreconditioner("CuSolverRf", _resolve_Rf);
      // _resolve_FGMRES->resetMatrix(_A);

      delete[] P;
      delete[] Q;
      delete L;
      delete L_csc;
      delete U;
      delete U_csc;
    }
  }
  else // After 1st iteration
  {
    if (_method == resolve_glu)
    {
      // Copy rhs_vals to vec_rhs cuda
      _vec_rhs->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      int status = _resolve_GLU->solve(_vec_rhs, _vec_x);
      if (status != 0)
      {
        std::cout << "GLU solve status: " << status << std::endl;
      }
      // Copy vec_x cuda to vec_x in cpu
      _vec_x->update(_vec_x->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::HOST);
    }
    else if (_method == resolve_rf)
    {
      // Copy rhs_vals to vec_rhs cuda
      _vec_rhs->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      int status = _resolve_Rf->solve(_vec_rhs, _vec_x);
      if (status != 0)
      {
        std::cout << "RF solve status: " << status << std::endl;
      }
      // Copy vec_x cuda to vec_x in cpu
      _vec_x->update(_vec_x->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::HOST);
    }
    else if (_method == resolve_rf_fgmres)
    {
      // Copy rhs_vals to vec_rhs cuda
      _vec_rhs->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      int status = _resolve_Rf->solve(_vec_rhs, _vec_x);
      if (status != 0)
      {
        std::cout << "RF solve status: " << status << std::endl;
      }

      _resolve_FGMRES->resetMatrix(_A);
      status = _resolve_FGMRES->solve(_vec_rhs, _vec_x);
      if (status != 0)
      {
        std::cout << "RF_FGMRES solve status: " << status << std::endl;
      }
      // Copy vec_x cuda to vec_x in cpu
      _vec_x->update(_vec_x->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::HOST);
    }
    else
    {
      // Copy rhs_vals to vec_rhs cuda
      _vec_rhs->update(rhs_vals, ReSolve::memory::HOST, ReSolve::memory::HOST);
      int status = _resolve_KLU->solve(_vec_rhs, _vec_x);
      if (status != 0)
      {
        std::cout << "KLU solve status: " << status << std::endl;
      }
    }
  }

  // copy vec_x to rhs_vals
  memcpy(rhs_vals, _vec_x->getData(ReSolve::memory::HOST), (_ndim) * sizeof(ReSolve::real_type));

  if (HaveIpData())
  {
    IpData().TimingStats().LinearSystemBackSolve().End();
  }

  // Setup Rf at first iteration
  _n_iteration += 1;
  _first_iteration = false;

  return SYMSOLVER_SUCCESS;
}

Number* ReSolveSolverInterface::GetValuesArrayPtr()
{
  DBG_START_METH("ReSolveSolverInterface::GetValuesArrayPtr", dbg_verbosity);
  DBG_ASSERT(_initialized);

  return _A->getValues(ReSolve::memory::HOST);
}

Index ReSolveSolverInterface::NumberOfNegEVals() const
{
  return _numneg;
}

bool ReSolveSolverInterface::IncreaseQuality()
{
  return true;
}

} // namespace Ipopt
