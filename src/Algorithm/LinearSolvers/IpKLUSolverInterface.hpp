// Copyright (C) 2023 ORNL and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Slaven Peles, Maksudul Alam

#ifndef __IPKLUSOLVERINTERFACE_HPP__
#define __IPKLUSOLVERINTERFACE_HPP__

#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpLibraryLoader.hpp"
#include "IpTypes.h"

#include "klu.h"
namespace Ipopt
{
/** Interface to the symmetric linear solver KLU, derived from
 *  SparseSymLinearSolverInterface.
 */
class KLUSolverInterface: public SparseSymLinearSolverInterface
{
public:
   /** @name Constructor/Destructor */
   ///@{
   /** Constructor */
   KLUSolverInterface();

   /** Destructor */
   virtual ~KLUSolverInterface();
   ///@}

   bool InitializeImpl(
      const OptionsList& options,
      const std::string& prefix
   );

   /** @name Methods for requesting solution of the linear system. */
   ///@{
   virtual ESymSolverStatus InitializeStructure(
      Index        dim,
      Index        nonzeros,
      const Index* airn,
      const Index* ajcn
   );

   virtual Number* GetValuesArrayPtr();

   virtual ESymSolverStatus MultiSolve(
      bool         new_matrix,
      const Index* airn,
      const Index* ajcn,
      Index        nrhs,
      Number*      rhs_vals,
      bool         check_NegEVals,
      Index        numberOfNegEVals
   );

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
   static void RegisterOptions(
      SmartPtr<RegisteredOptions> roptions
   );
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
   KLUSolverInterface(
      const KLUSolverInterface&
   );

   /** Default Assignment Operator */
   void operator=(
      const KLUSolverInterface&
   );
   ///@}

   /** Number of nonzeros of the matrix */
   Index nonzeros_;

   bool initialized_;
   Index ndim_;    ///< Number of dimensions
   Number* val_; ///< Storage for variables
   Index numneg_;  ///< Number of negative pivots in last factorization
   bool pivtol_changed_; ///< indicates if pivtol has been changed

   klu_symbolic* Symbolic;
   klu_numeric* Numeric;
   klu_common Common;

   int* Ap;
   int* Ai;

};

} // namespace Ipopt
#endif
