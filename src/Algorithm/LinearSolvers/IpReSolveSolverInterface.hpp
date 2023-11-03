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
//#include <resolve/LinSolverDirectCuSolverGLU.hpp>
//#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <sstream>
#include <string>

#include "IpLibraryLoader.hpp"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpTypes.h"

using namespace ReSolve::constants;

namespace Ipopt
{

static const std::string resolve_glu = "glu";
static const std::string resolve_klu = "klu";
static const std::string resolve_rf = "rf";
static const std::string resolve_rf_fgmres = "rf_fgmres";

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
  Index _nonzeros;

  bool _initialized;
  Index _ndim;          ///< Number of dimensions
  Number* _val;         ///< Storage for variables
  Index _numneg;        ///< Number of negative pivots in last factorization
  bool _pivtol_changed; ///< indicates if pivtol has been changed
  bool _re_factorize;
  bool _factorize;
  bool _first_iteration;
  std::string _method;
  int _n_iteration;

  ReSolve::LinSolverDirectKLU* _resolve_KLU;
  ReSolve::matrix::Csr* _A;
  ReSolve::vector::Vector* _vec_rhs;
  ReSolve::vector::Vector* _vec_x;
  ReSolve::LinAlgWorkspaceHIP* _workspace_HIP;
  ReSolve::LinSolverDirectRocSolverRf* _resolve_Rf;
  ReSolve::GramSchmidt* _GS;
  ReSolve::LinSolverIterativeFGMRES* _resolve_FGMRES;

  ReSolve::MatrixHandler* _matrix_handler;
  ReSolve::VectorHandler* _vector_handler;
};

} // namespace Ipopt
#endif
