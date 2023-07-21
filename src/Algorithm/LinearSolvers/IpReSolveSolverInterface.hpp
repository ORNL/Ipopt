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
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/MatrixCOO.hpp>
#include <resolve/MatrixCSC.hpp>
#include <resolve/MatrixCSR.hpp>
#include <resolve/MatrixHandler.hpp>
#include <resolve/Vector.hpp>
#include <resolve/VectorHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <sstream>
#include <string>

#include "IpLibraryLoader.hpp"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpTypes.h"

namespace Ipopt
{
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

    bool InitializeImpl(const OptionsList &options, const std::string &prefix);

    /** @name Methods for requesting solution of the linear system. */
    ///@{
    virtual ESymSolverStatus InitializeStructure(Index dim, Index nonzeros, const Index *airn, const Index *ajcn);

    virtual Number *GetValuesArrayPtr();

    virtual ESymSolverStatus MultiSolve(bool new_matrix, const Index *airn, const Index *ajcn, Index nrhs,
                                        Number *rhs_vals, bool check_NegEVals, Index numberOfNegEVals);

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
    ReSolveSolverInterface(const ReSolveSolverInterface &);

    /** Default Assignment Operator */
    void operator=(const ReSolveSolverInterface &);
    ///@}

    /** Number of nonzeros of the matrix */
    Index nonzeros_;

    bool initialized_;
    Index ndim_;          ///< Number of dimensions
    Number *val_;         ///< Storage for variables
    Index numneg_;        ///< Number of negative pivots in last factorization
    bool pivtol_changed_; ///< indicates if pivtol has been changed
    bool refactorize_;
    bool factorize_;
    bool first_iteration_;

    ReSolve::LinSolverDirectKLU *resolve_KLU_;
    ReSolve::MatrixCSR *A_;
    ReSolve::Vector *vec_rhs_;
    ReSolve::Vector *vec_x_;
    ReSolve::LinAlgWorkspaceCUDA *workspace_CUDA_;
    ReSolve::LinSolverDirectCuSolverGLU *resolve_GLU_;
};

} // namespace Ipopt
#endif
