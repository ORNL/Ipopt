// Copyright (C) 2023 ORNL and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Slaven Peles, Maksudul Alam

#ifndef __IPKLUSOLVERINTERFACE_HPP__
#define __IPKLUSOLVERINTERFACE_HPP__

#include "IpLibraryLoader.hpp"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpTypes.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "klu.h"
namespace Ipopt
{
/** Interface to the symmetric linear solver KLU, derived from
 *  SparseSymLinearSolverInterface.
 */
class KLUSolverInterface : public SparseSymLinearSolverInterface
{
  public:
    /** @name Constructor/Destructor */
    ///@{
    /** Constructor */
    KLUSolverInterface();

    /** Destructor */
    virtual ~KLUSolverInterface();
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
    KLUSolverInterface(const KLUSolverInterface &);

    /** Default Assignment Operator */
    void operator=(const KLUSolverInterface &);
    ///@}

    /** Number of nonzeros of the matrix */
    Index _nonzeros;

    bool _initialized;
    Index _ndim;          ///< Number of dimensions
    Number *_val;         ///< Storage for variables
    Index _numneg;        ///< Number of negative pivots in last factorization
    bool _pivtol_changed; ///< indicates if pivtol has been changed
    bool _re_factorize;
    bool _factorize;
    bool _first_iteration;

    klu_symbolic *_Symbolic;
    klu_numeric *_Numeric;
    klu_common _Common;

    int *_Ai;
    int *_Aj;
};

} // namespace Ipopt
#endif
