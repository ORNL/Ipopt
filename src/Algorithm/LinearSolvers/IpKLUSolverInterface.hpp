// Copyright (C) 2023 ORNL and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Slaven Peles, Maksudul Alam

#ifndef __IPKLUSOLVERINTERFACE_HPP__
#define __IPKLUSOLVERINTERFACE_HPP__

#include <fstream>
#include <iomanip>
#include <iostream>
#include "IpLibraryLoader.hpp"
#include "IpSparseSymLinearSolverInterface.hpp"
#include <sstream>
#include <string>
#include "IpTypes.h"

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
    Index nonzeros_;

    bool initialized_;
    Index ndim_;          ///< Number of dimensions
    Number *val_;         ///< Storage for variables
    Index numneg_;        ///< Number of negative pivots in last factorization
    bool pivtol_changed_; ///< indicates if pivtol has been changed
    bool refactorize_;
    bool factorize_;

    klu_symbolic *Symbolic_;
    klu_numeric *Numeric_;
    klu_common Common_;

    int *Ap_;
    int *Ai_;

    int seq = 0;

    std::string intToString(int value)
    {
        std::stringstream ss;
        ss << std::setw(2) << std::setfill('0') << value;
        return ss.str();
    }

    void write_CSR_matrix_rhs(int *ia, int *ja, double *vals, double *rhs, int ndim, int nonzeros, std::string prefix,
                              int matrix_seq)
    {
        std::string mat_file_name = "matrix_" + prefix + "_" + intToString(matrix_seq) + ".mtx";
        std::string vec_file_name = "rhs_" + prefix + "_" + intToString(matrix_seq) + ".mtx";
        printf("Matrix Filename: %s\n", mat_file_name.c_str());
        printf("Vector Filename: %s\n", vec_file_name.c_str());

        std::ofstream f_mat;
        std::ofstream f_vec;

        f_mat.open(mat_file_name);
        f_vec.open(vec_file_name);

        f_mat << "%%MatrixMarket matrix coordinate real symmetric\n";
        f_mat << "% ID: " << matrix_seq << "\n";
        f_mat << ndim << " " << ndim << " " << nonzeros << "\n";

        f_vec << "%%MatrixMarket matrix array real general\n";
        f_vec << "% ID: " << matrix_seq << "\n";
        f_vec << ndim << " " << 1 << "\n";

        for (int i = 0; i < ndim_; i++)
        {
            int nR = ia[i + 1] - ia[i];
            printf("Number of Items in row %d = %d\n", i, nR);
            for (int j = 0; j < nR; j++)
            {
                int idx = ia[i] + j;
                int c = ja[idx];
                printf("[%d, %d] == %d\n", i, c, idx);
                f_mat << i << " " << c << " " << vals[idx] << "\n";
            }
            f_vec << rhs[i] << "\n";
        }

        f_mat.close();
        f_vec.close();
    }
};

} // namespace Ipopt
#endif
