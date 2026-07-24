// Copyright (C) 2023 ORNL and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Slaven Peles, Maksudul Alam

#include "IpKLUSolverInterface.hpp"
#include "IpoptConfig.h"

#include <cmath>
#include <iostream>

namespace Ipopt
{
#if IPOPT_VERBOSITY > 0
static const Index dbg_verbosity = 0;
#endif

KLUSolverInterface::KLUSolverInterface() : _val(NULL), _Symbolic(NULL), _Numeric(NULL)
{
  DBG_START_METH("KLUSolverInterface::KLUSolverInterface()", dbg_verbosity);
  _rcond_val = 1e-128; 
}

KLUSolverInterface::~KLUSolverInterface()
{
  DBG_START_METH("KLUSolverInterface::~KLUSolverInterface()", dbg_verbosity);
  delete[] _val;
  klu_free_symbolic(&_Symbolic, &_Common);
  klu_free_numeric(&_Numeric, &_Common);
}

void KLUSolverInterface::RegisterOptions(SmartPtr<RegisteredOptions> roptions)
{
  roptions->AddNumberOption("klu_tol",                    //
                            "Partial pivoting tolerance", //
                            0.001,                        //
                            "If the diagonal entry has a magnitude greater than or equal to tol times the largest "
                            "magnitude of entries in the pivot column, then the diagonal entry is chosen.",
                            false);

  roptions->AddIntegerOption("klu_ordering",                        //
                             "Which fill-reducing ordering to use", //
                             0,                                     //
                             "0 for AMD, 1 for COLAMD, 2 for a user-provided permutation P and Q (or a natural "
                             "ordering if P and Q are NULL), or 3 for the user order function.",
                             false);

  roptions->AddIntegerOption("klu_btf",                                                                                    //
                             "Use BTF",                                                                                    //
                             1,                                                                                            //
                             "if nonzero, then BTF is used to permute the input matrix into block upper triangular form.", //
                             false);

  roptions->AddIntegerOption("klu_scale",                                  //
                             "Whether or not the matrix should be scaled", //
                             2,                                            //
                             "If scale < 0, then no scaling is performed and the input matrix is not checked for errors. If scale >= 0, the "
                             "input matrix is check for errors. If scale=0, then no scaling is performed. If scale=1, then each row of A is "
                             "divided by the sum of the absolute values in that row. If scale=2, then each row of A is divided by the "
                             "maximum absolute value in that row. Default: 2.",
                             false);

  roptions->AddBoolOption("klu_halt_if_singular",            //
                          "how to handle a singular matrix", //
                          false,                             //
                          "FALSE: keep going, TRUE: stop quickly.", false);

  roptions->AddNumberOption("klu_rcond_val",                    //
                            "KLU RCond Value", //
                            1e-16,                        //
                            "RCond Value to initiate KLU Factorize Again",
                            false);
}

bool KLUSolverInterface::InitializeImpl(const OptionsList& options, const std::string& prefix)
{
  klu_defaults(&_Common);

  Number tol;
  options.GetNumericValue("klu_tol", tol, prefix);
  _Common.tol = tol;
  // printf("klu_tol: %f\n", _Common.tol);

  Index order_method;
  options.GetIntegerValue("klu_ordering", order_method, prefix);
  _Common.ordering = order_method;
  // printf("klu_ordering: %d\n", _Common.ordering);

  Index btf;
  options.GetIntegerValue("klu_btf", btf, prefix);
  _Common.btf = btf;
  // printf("klu_btf: %d\n", _Common.btf);

  Index scale;
  options.GetIntegerValue("klu_scale", scale, prefix);
  _Common.scale = scale;
  // printf("klu_scale: %d\n", _Common.scale);

  bool halt_if_singular;
  options.GetBoolValue("klu_halt_if_singular", halt_if_singular, prefix);
  _Common.halt_if_singular = halt_if_singular;
  // printf("klu_halt_if_singular: %d\n", _Common.halt_if_singular);

  options.GetNumericValue("klu_rcond_val", _rcond_val, prefix);

  return true;
}

ESymSolverStatus KLUSolverInterface::MultiSolve(bool new_matrix, const Index* ia, const Index* ja, Index nrhs, Number* rhs_vals, bool check_NegEVals, Index numberOfNegEVals)
{
  DBG_START_METH("KLUSolverInterface::MultiSolve", dbg_verbosity);

  bool full_factor_done = false;

  if (_factorize)
  {
    // perform the factorization
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().Start();
    }
    klu_free_numeric(&_Numeric, &_Common);
    _Numeric = klu_factor(_Ai, _Aj, _val, _Symbolic, &_Common);
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().End();
    }
    if (_Numeric == nullptr)
    {
      DBG_PRINT((1, "FACTORIZATION FAILED!\n"));
      return SYMSOLVER_FATAL_ERROR; // Matrix singular or error occurred
    }
    _factorize = false;
    full_factor_done = true;
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

  // check if a factorization has to be done
  DBG_PRINT((1, "new_matrix = %d\n", new_matrix));
  if (!_first_iteration && (new_matrix || _re_factorize))
  {
    // perform the factorization
    if (HaveIpData())
    {
      IpData().TimingStats().LinearSystemFactorization().Start();
    }
    klu_refactor(_Ai, _Aj, _val, _Symbolic, _Numeric, &_Common);
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

  // klu_rcond(_Symbolic, _Numeric, &_Common);
  // printf("RCond: %12.8e\n", _Common.rcond);
  // klu_condest(_Ai, _val, _Symbolic, _Numeric, &_Common);
  // printf("Condest: %12.8e\n", _Common.condest);

  klu_rcond(_Symbolic, _Numeric, &_Common);
  if (_Common.rcond < _rcond_val)
  {
    if (full_factor_done)
    {
      return SYMSOLVER_SINGULAR;
    }
    else
    {
      // refactor effectively failed -- need to call again
      // and do full factorization
      _factorize = true;
      printf("RCond: %12.8e\n", _Common.rcond);
      printf("Need to do full factorization again.\n");
      DBG_PRINT((1, "Ask caller to call again.\n"))
      return SYMSOLVER_CALL_AGAIN;
    }
  }

  klu_solve(_Symbolic, _Numeric, _ndim, nrhs, rhs_vals, &_Common);
  // printf("KLU Solve Status: %d\n", sts);

  if (HaveIpData())
  {
    IpData().TimingStats().LinearSystemBackSolve().End();
  }

  _first_iteration = false;

  return SYMSOLVER_SUCCESS;
}

Number* KLUSolverInterface::GetValuesArrayPtr()
{
  DBG_START_METH("KLUSolverInterface::GetValuesArrayPtr", dbg_verbosity);
  DBG_ASSERT(_initialized);

  return _val;
}

/** Initialize the local copy of the positions of the nonzero elements */
ESymSolverStatus KLUSolverInterface::InitializeStructure(Index dim, Index nonzeros, const Index* ia, const Index* ja)
{
  DBG_START_METH("KLUSolverInterface::InitializeStructure", dbg_verbosity);

  ESymSolverStatus retval = SYMSOLVER_SUCCESS;
  printf("dim: %d, nonzeros %d\n", dim, nonzeros);

  // Store size for later use
  _ndim = dim;
  _nonzeros = nonzeros;

  _Ai = const_cast<int*>(ia);
  _Aj = const_cast<int*>(ja);
  // Setup memory for values
  if (_val != NULL)
  {
    delete[] _val;
  }
  _val = new Number[nonzeros];

  if (HaveIpData())
  {
    IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
  }
  _Symbolic = klu_analyze(_ndim, _Ai, _Aj, &_Common);
  if (HaveIpData())
  {
    IpData().TimingStats().LinearSystemSymbolicFactorization().End();
  }

  _factorize = true;
  _first_iteration = true;

  if (_Symbolic == nullptr)
  {
    printf("Symbolic_ factorization crashed with Common_.status = %d \n", _Common.status);
    return SYMSOLVER_FATAL_ERROR;
  }

  _initialized = true;

  return retval;
}

Index KLUSolverInterface::NumberOfNegEVals() const
{
  return _numneg;
}

bool KLUSolverInterface::IncreaseQuality()
{
  return true;
}

} // namespace Ipopt
