// Copyright (C) 2023 ORNL and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Slaven Peles, Maksudul Alam

#include "IpReSolveSolverInterface.hpp"

#include <cmath>
#include <iostream>

#include "IpoptConfig.h"

#define USE_GLU 1

namespace Ipopt
{
#if IPOPT_VERBOSITY > 0
static const Index dbg_verbosity = 0;
#endif

ReSolveSolverInterface::ReSolveSolverInterface() : val_(NULL)
{
    DBG_START_METH("ReSolveSolverInterface::ReSolveSolverInterface()", dbg_verbosity);
    resolve_KLU_ = new ReSolve::LinSolverDirectKLU();
#if USE_GLU
    workspace_CUDA_ = new ReSolve::LinAlgWorkspaceCUDA();
    workspace_CUDA_->initializeHandles();
    resolve_GLU_ = new ReSolve::LinSolverDirectCuSolverGLU(workspace_CUDA_);
#endif
}

ReSolveSolverInterface::~ReSolveSolverInterface()
{
    DBG_START_METH("ReSolveSolverInterface::~ReSolveSolverInterface()", dbg_verbosity);
    delete[] val_;
}

void ReSolveSolverInterface::RegisterOptions(SmartPtr<RegisteredOptions> roptions)
{
    printf("ReSolveSolverInterface::%s is called\n", __func__);
}

bool ReSolveSolverInterface::InitializeImpl(const OptionsList &options, const std::string &prefix)
{
    printf("ReSolveSolverInterface::%s is called\n", __func__);

    resolve_KLU_->setupParameters(1, 0.1, false);

    return true;
}

ESymSolverStatus ReSolveSolverInterface::MultiSolve(bool new_matrix, const Index *ia, const Index *ja, Index nrhs,
                                                    Number *rhs_vals, bool check_NegEVals, Index numberOfNegEVals)
{
    DBG_START_METH("ReSolveSolverInterface::MultiSolve", dbg_verbosity);

    A_->updateData(A_->getRowData("cpu"), A_->getColData("cpu"), A_->getValues("cpu"), "cpu", "cuda");

    if (factorize_)
    {
        // perform the factorization
        if (HaveIpData())
        {
            IpData().TimingStats().LinearSystemFactorization().Start();
        }
        printf("Factorize Steps 1\n");
        int status = resolve_KLU_->factorize();
#if USE_GLU
        printf("Factorize Steps 2\n");
        ReSolve::Matrix *L = resolve_KLU_->getLFactor();
        printf("Factorize Steps 3\n");
        ReSolve::Matrix *U = resolve_KLU_->getUFactor();
        printf("Factorize Steps 4\n");
        if (L == nullptr)
        {
            printf("ERROR");
        }

        ReSolve::index_type *P = resolve_KLU_->getPOrdering();
        printf("Factorize Steps 5\n");
        ReSolve::index_type *Q = resolve_KLU_->getQOrdering();
        printf("Factorize Steps 6\n");
        resolve_GLU_->setup(A_, L, U, P, Q);
        printf("Factorize Steps 7\n");
#endif

        if (HaveIpData())
        {
            IpData().TimingStats().LinearSystemFactorization().End();
        }

        if (status != 0)
        {
            DBG_PRINT((1, "FACTORIZATION FAILED!\n"));
            return SYMSOLVER_FATAL_ERROR; // Matrix singular or error occurred
        }

        factorize_ = false;
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
            refactorize_ = true;
            return SYMSOLVER_CALL_AGAIN;
        }
    }

    // check if a factorization has to be done
    DBG_PRINT((1, "new_matrix = %d\n", new_matrix));
    if (!first_iteration_ && (new_matrix || refactorize_))
    {
        // perform the factorization
        if (HaveIpData())
        {
            IpData().TimingStats().LinearSystemFactorization().Start();
        }
#if USE_GLU
        int status = resolve_GLU_->refactorize();
        std::cout << "CUSOLVER GLU refactorization status: " << status << std::endl;
#else
        int status = resolve_KLU_->refactorize();
        std::cout << "KLU refactorization status: " << status << std::endl;
#endif
        refactorize_ = false;
        if (HaveIpData())
        {
            IpData().TimingStats().LinearSystemFactorization().End();
        }
    }

    if (HaveIpData())
    {
        IpData().TimingStats().LinearSystemBackSolve().Start();
    }

#if USE_GLU
    // Copy rhs_vals to vec_rhs cuda
    vec_rhs_->update(rhs_vals, "cpu", "cuda");
    int status = resolve_GLU_->solve(vec_rhs_, vec_x_);
    // Copy vec_x cuda to vec_x
    std::cout << "GLU solve status: " << status << std::endl;
    vec_x_->update(vec_x_->getData("cuda"), "cuda", "cpu");
#else
    // Copy rhs_vals to vec_rhs cuda
    vec_rhs_->update(rhs_vals, "cpu", "cpu");
    int status = resolve_KLU_->solve(vec_rhs_, vec_x_);
    // Copy vec_x cuda to vec_x
    std::cout << "KLU solve status: " << status << std::endl;
#endif
    // copy vec_x to rhs_vals
    std::memcpy(rhs_vals, vec_x_->getData("cpu"), (ndim_) * sizeof(ReSolve::real_type));

    if (HaveIpData())
    {
        IpData().TimingStats().LinearSystemBackSolve().End();
    }

    first_iteration_ = false;

    return SYMSOLVER_SUCCESS;
}

Number *ReSolveSolverInterface::GetValuesArrayPtr()
{
    DBG_START_METH("ReSolveSolverInterface::GetValuesArrayPtr", dbg_verbosity);
    DBG_ASSERT(initialized_);

    return A_->getValues("cpu");
}

/** Initialize the local copy of the positions of the nonzero elements */
ESymSolverStatus ReSolveSolverInterface::InitializeStructure(Index dim, Index nonzeros, const Index *ia,
                                                             const Index *ja)
{
    DBG_START_METH("ReSolveSolverInterface::InitializeStructure", dbg_verbosity);

    ESymSolverStatus retval = SYMSOLVER_SUCCESS;
    printf("dim: %d, nonzeros %d\n", dim, nonzeros);

    // Store size for later use
    ndim_ = dim;
    nonzeros_ = nonzeros;

    A_ = new ReSolve::MatrixCSR(dim, dim, nonzeros);
    if (val_ != NULL)
    {
        delete[] val_;
    }
    val_ = new Number[nonzeros];

    A_->setMatrixData(const_cast<int *>(ia), const_cast<int *>(ja), val_, "cpu");

    resolve_KLU_->setup(A_);

    vec_rhs_ = new ReSolve::Vector(A_->getNumRows());
    vec_x_ = new ReSolve::Vector(A_->getNumRows());

    vec_x_->allocate("cpu"); // for KLU
    vec_x_->allocate("cuda");

    if (HaveIpData())
    {
        IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
    }
    int status = resolve_KLU_->analyze();
    if (HaveIpData())
    {
        IpData().TimingStats().LinearSystemSymbolicFactorization().End();
    }

    factorize_ = true;
    first_iteration_ = true;

    if (status != 0)
    {
        printf("Symbolic_ factorization crashed with Common_.status = %d \n", status);
        return SYMSOLVER_FATAL_ERROR;
    }

    initialized_ = true;

    return retval;
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
