// Copyright (C) 2023 ORNL and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Slaven Peles, Maksudul Alam

#ifndef __IPRESOLVESOLVERINTERFACE_HPP__
#define __IPRESOLVESOLVERINTERFACE_HPP__

#include <fstream>
#include <iomanip>
#include <iostream>

#include "IpoptConfig.h"

#if RESOLVE_WITH_CUDA
#include "NVMLHelper.hpp"
#endif

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>

#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#if RESOLVE_WITH_GPU
#include <resolve/LinSolverIterativeFGMRES.hpp>

#if RESOLVE_WITH_CUDA
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
using workspace_type = ReSolve::LinAlgWorkspaceCUDA;
using rf_solver = ReSolve::LinSolverDirectCuSolverRf;

#else
#include <resolve/LinSolverDirectRocSolverRf.hpp>
using workspace_type = ReSolve::LinAlgWorkspaceHIP;
using rf_solver = ReSolve::LinSolverDirectRocSolverRf;
# endif
#endif


#include <sstream>
#include <string>

#include "IpLibraryLoader.hpp"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpTypes.h"
using namespace ReSolve::constants;


namespace Ipopt
{

static const std::string resolve_klu = "klu";

#if RESOLVE_WITH_GPU
#if RESOLVE_WITH_CUDA
static const std::string resolve_glu = "glu";
# endif
static const std::string resolve_rf = "rf";
static const std::string resolve_rf_fgmres = "rf_fgmres";
#endif

/** Interface to the symmetric linear solver ReSolve, derived from
 *  SparseSymLinearSolverInterface.
 */
class ReSolveSolverInterface : public SparseSymLinearSolverInterface
{
public:
  /** @name Constructor/Destructor */
  ///@{
  /** Constructor */
  ReSolveSolverInterface();

  /** Destructor */
  virtual ~ReSolveSolverInterface();
  ///@}

  bool InitializeImpl(const OptionsList& options, const std::string& prefix);

  /** @name Methods for requesting solution of the linear system. */
  ///@{
  virtual ESymSolverStatus InitializeStructure(Index dim, Index nonzeros, const Index* airn, const Index* ajcn);

  virtual Number* GetValuesArrayPtr();

  virtual ESymSolverStatus MultiSolve(bool new_matrix, const Index* airn, const Index* ajcn, Index nrhs, Number* rhs_vals, bool check_NegEVals, Index numberOfNegEVals);

  virtual Index NumberOfNegEVals() const;
  ///@}

  //* @name Options of Linear solver */
  ///@{
  virtual bool IncreaseQuality();

  virtual bool ProvidesInertia() const
  {
    return false;
  }

  EMatrixFormat MatrixFormat() const
  {
    return CSR_Full_Format_0_Offset;
  }
  ///@}

  ///@{
  static void RegisterOptions(SmartPtr<RegisteredOptions> roptions);
  ///@}

private:
  /**@name Default Compiler Generated Methods
   * (Hidden to avoid implicit creation/calling).
   * These methods are not implemented and
   * we do not want the compiler to implement
   * them for us, so we declare them private
   * and do not define them. This ensures that
   * they will not be implicitly created/called. */
  ///@{
  /** Copy Constructor */
  ReSolveSolverInterface(const ReSolveSolverInterface&);

  /** Default Assignment Operator */
  void operator=(const ReSolveSolverInterface&);
  ///@}

  /** Number of nonzeros of the matrix */
  Index nonzeros_;

  bool initialized_;
  bool initialized_resolve_;
  Index ndim_;          ///< Number of dimensions
  Number* val_;         ///< Storage for variables
  Index numneg_;        ///< Number of negative pivots in last factorization
  bool pivtol_changed_; ///< indicates if pivtol has been changed
  bool re_factorize_;
  bool factorize_;
  std::string method_;
  int n_iteration_;

  int k_;
  Number rcond_val_;
  bool use_rcond_;

  int factor_by_t_;

  ReSolve::LinSolverDirectKLU* resolve_KLU_;
  ReSolve::LinAlgWorkspaceCpu* workspace_CPU_;

  ReSolve::matrix::Csr* A_;
  ReSolve::vector::Vector* vec_rhs_;
  ReSolve::vector::Vector* vec_x_;


#if RESOLVE_WITH_GPU
  workspace_type* workspace_GPU_;
  rf_solver* resolve_Rf_;
# if RESOLVE_WITH_CUDA
  ReSolve::LinSolverDirectCuSolverGLU* resolve_GLU_;
# endif
  ReSolve::GramSchmidt* GS_;
  ReSolve::LinSolverIterativeFGMRES* resolve_FGMRES_;
#endif

  ReSolve::MatrixHandler* matrix_handler_;
  ReSolve::VectorHandler* vector_handler_;
};

} // namespace Ipopt
#endif
