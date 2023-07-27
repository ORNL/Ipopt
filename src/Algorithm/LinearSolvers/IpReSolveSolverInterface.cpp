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

bool ReSolveSolverInterface::InitializeImpl(const OptionsList &options, const std::string &prefix)
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
    _resolve_KLU->setupParameters(order_method, tol, halt_if_singular);

    if (_method == resolve_glu)
    {
        _workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA();
        _workspace_CUDA->initializeHandles();
        _resolve_GLU = new ReSolve::LinSolverDirectCuSolverGLU(_workspace_CUDA);
    }

    return true;
}

ESymSolverStatus ReSolveSolverInterface::MultiSolve(bool new_matrix, const Index *ia, const Index *ja, Index nrhs,
                                                    Number *rhs_vals, bool check_NegEVals, Index numberOfNegEVals)
{
    DBG_START_METH("ReSolveSolverInterface::MultiSolve", dbg_verbosity);

    _A->updateData(_A->getRowData("cpu"), _A->getColData("cpu"), _A->getValues("cpu"), "cpu", "cuda");

    if (_factorize)
    {
        // perform the factorization
        if (HaveIpData())
        {
            IpData().TimingStats().LinearSystemFactorization().Start();
        }
        int status = _resolve_KLU->factorize();
        if (_method == resolve_glu)
        {
            ReSolve::Matrix *L = _resolve_KLU->getLFactor();
            ReSolve::Matrix *U = _resolve_KLU->getUFactor();
            if (L == nullptr)
            {
                printf("ERROR");
            }
            ReSolve::index_type *P = _resolve_KLU->getPOrdering();
            ReSolve::index_type *Q = _resolve_KLU->getQOrdering();
            _resolve_GLU->setup(_A, L, U, P, Q);
        }

        if (HaveIpData())
        {
            IpData().TimingStats().LinearSystemFactorization().End();
        }

        if (status != 0)
        {
            DBG_PRINT((1, "FACTORIZATION FAILED!\n"));
            return SYMSOLVER_FATAL_ERROR; // Matrix singular or error occurred
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

    // check if a factorization has to be done
    DBG_PRINT((1, "new_matrix = %d\n", new_matrix));
    if (!_first_iteration && (new_matrix || _re_factorize))
    {
        // perform the factorization
        if (HaveIpData())
        {
            IpData().TimingStats().LinearSystemFactorization().Start();
        }
        if (_method == resolve_glu)
        {
            int status = _resolve_GLU->refactorize();
            if (status != 0)
            {
                std::cout << "CUSOLVER GLU refactorization status: " << status << std::endl;
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

    if (_method == resolve_glu)
    {
        // Copy rhs_vals to vec_rhs cuda
        _vec_rhs->update(rhs_vals, "cpu", "cuda");
        int status = _resolve_GLU->solve(_vec_rhs, _vec_x);
        if (status != 0)
        {
            std::cout << "GLU solve status: " << status << std::endl;
        }
        // Copy vec_x cuda to vec_x in cpu
        _vec_x->update(_vec_x->getData("cuda"), "cuda", "cpu");
    }
    else
    {
        // Copy rhs_vals to vec_rhs cuda
        _vec_rhs->update(rhs_vals, "cpu", "cpu");
        int status = _resolve_KLU->solve(_vec_rhs, _vec_x);
        if (status != 0)
        {
            std::cout << "KLU solve status: " << status << std::endl;
        }
    }
    // copy vec_x to rhs_vals
    std::memcpy(rhs_vals, _vec_x->getData("cpu"), (_ndim) * sizeof(ReSolve::real_type));

    if (HaveIpData())
    {
        IpData().TimingStats().LinearSystemBackSolve().End();
    }

    _first_iteration = false;

    return SYMSOLVER_SUCCESS;
}

Number *ReSolveSolverInterface::GetValuesArrayPtr()
{
    DBG_START_METH("ReSolveSolverInterface::GetValuesArrayPtr", dbg_verbosity);
    DBG_ASSERT(_initialized);

    return _A->getValues("cpu");
}

/** Initialize the local copy of the positions of the nonzero elements */
ESymSolverStatus ReSolveSolverInterface::InitializeStructure(Index dim, Index nonzeros, const Index *ia,
                                                             const Index *ja)
{
    DBG_START_METH("ReSolveSolverInterface::InitializeStructure", dbg_verbosity);

    ESymSolverStatus retval = SYMSOLVER_SUCCESS;
    printf("dim: %d, nonzeros %d\n", dim, nonzeros);

    // Store size for later use
    _ndim = dim;
    _nonzeros = nonzeros;

    _A = new ReSolve::MatrixCSR(dim, dim, nonzeros);
    if (_val != NULL)
    {
        delete[] _val;
    }
    _val = new Number[nonzeros];

    _A->setMatrixData(const_cast<int *>(ia), const_cast<int *>(ja), _val, "cpu");

    _resolve_KLU->setup(_A);

    _vec_rhs = new ReSolve::Vector(_A->getNumRows());
    _vec_x = new ReSolve::Vector(_A->getNumRows());

    _vec_x->allocate("cpu"); // for KLU
    _vec_x->allocate("cuda");

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

    if (status != 0)
    {
        printf("Symbolic_ factorization crashed with Common_.status = %d \n", status);
        return SYMSOLVER_FATAL_ERROR;
    }

    _initialized = true;

    return retval;
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
