#ifndef __TEST_CSR_H__
#define __TEST_CSR_H__

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <vector>

using namespace std;

namespace debug
{

struct CSR
{
  vector<double> val;
  vector<int> col_ind;
  vector<int> row_ptr;
};

class Debug
{
public:
  Debug()
  {
  }

  template <typename T> static T* load_b_data(std::string prefix, int iter, int& n)
  {
    std::fstream file;

    std::stringstream ss;
    ss << prefix << "-" << iter << ".txt";
    std::string filename = ss.str();

    file.open(filename, std::ios::in | std::fstream::binary);
    file.read((char*)&n, sizeof(n));
    T* d = new T[n];

    T p;
    for (int i = 0; i < n; i++)
    {
      file.read((char*)&p, sizeof(T));
      d[i] = p;
    }

    file.close();
    return d;
  }

  template <typename T> static void save_b_data(std::string prefix, int iter, int n, const T* data)
  {
    std::fstream file;
    std::stringstream ss;
    ss << prefix << "-" << iter << ".bin";
    std::string filename = ss.str();

    file.open(filename, std::ios::out | std::fstream::binary);

    file.write((char*)&n, sizeof(n));
    // std::cout.precision(std::numeric_limits<double>::max_digits10 - 1);
    file.write((char*)(data), n * sizeof(T));
    file.close();
  }

  template <typename T> static T* load_data(std::string prefix, int iter, int& n)
  {
    std::fstream file;

    std::stringstream ss;
    ss << prefix << "-" << iter << ".txt";
    std::string filename = ss.str();

    file.open(filename, std::ios::in);
    file >> n;
    T* d = new T[n];

    T p;
    for (int i = 0; i < n; i++)
    {
      file >> p;
      d[i] = p;
    }

    file.close();
    return d;
  }

  template <typename T> static void save_data(std::string prefix, int iter, int n, const T* data)
  {
    std::fstream file;
    std::stringstream ss;
    ss << prefix << "-" << iter << ".txt";
    std::string filename = ss.str();

    file.open(filename, std::ios::out);
    file << n << std::endl;

    // std::cout.precision(std::numeric_limits<double>::max_digits10 - 1);

    for (int i = 0; i < n; i++)
    {
      //   file << std::scientific << data[i] << std::endl;
      file << data[i] << std::endl;
    }
    // file.write((char *)(data), n * sizeof(T));
    file.close();
  }

  static void save_coo_as_mm(std::string prefix, int iter, int dim, int nnz, const int* row_ptr, const int* col_ind, double* val)
  {
    printf("%s\t%d\n", __FILE__, __LINE__);
    int cont = 0;

    std::stringstream ss;
    ss << prefix << "-matrix-" << iter << ".mtx";
    std::string filename = ss.str();
    std::cout << "Filename: " << filename << std::endl;

    std::ofstream mat_file(filename);

    mat_file << "%%MatrixMarket matrix coordinate real general" << std::endl;
    mat_file << "% ID: " << iter << std::endl;
    mat_file << dim << "\t" << dim << "\t" << nnz << std::endl;

    int nnz_c = 0;

    for (int i = 0; i < nnz; i++)
    {
      mat_file << row_ptr[i] << "\t" << col_ind[i] << "\t" << val[i] << std::endl;
    }
    mat_file.close();
    printf("%d, %d==%d\n", dim, nnz_c, nnz);
  }

  static void save_csc_as_csv(std::string prefix, int iter, int dim, int nnz, const int* row_ptr, const int* col_ind, double* val)
  {
    printf("%s\t%d\n", __FILE__, __LINE__);
    int cont = 0;

    std::stringstream ss;
    ss << prefix << "-matrix-" << iter << ".csv";
    std::string filename = ss.str();
    std::cout << "Filename: " << filename << std::endl;

    std::ofstream mat_file(filename);

    int nnz_c = 0;

    for (int i = 1; i <= dim; i++)
    {
      int row_start = row_ptr[i - 1];
      int row_end = row_ptr[i];

      nnz_c += row_end - row_start;
      // printf("%d, %d, %d\n", row_start, row_end, row_end - row_start);

      for (int jj = 0; jj < dim; jj++)
      {
        bool found = false;
        for (int j = row_start; j < row_end; j++)
        {
          if (jj == col_ind[j])
          {
            found = true;
            break;
          }
        }

        if (found)
        {
          mat_file << val[cont] << ",";
          cont++;
        }
        else
        {
          mat_file << "0,";
        }
      }
      mat_file << std::endl;
    }
    mat_file.close();
    printf("%d, %d==%d\n", dim, nnz_c, nnz);
  }

  static void save_csc_as_mm(std::string prefix, int iter, int dim, int nnz, const int* row_ptr, const int* col_ind, double* val)
  {
    printf("%s\t%d\n", __FILE__, __LINE__);
    int cont = 0;

    std::stringstream ss;
    ss << prefix << "-matrix-" << iter << ".mtx";
    std::string filename = ss.str();
    std::cout << "Filename: " << filename << std::endl;

    std::ofstream mat_file(filename);

    mat_file << "%%MatrixMarket matrix coordinate real general" << std::endl;
    mat_file << "% ID: " << iter << std::endl;
    mat_file << dim << "\t" << dim << "\t" << nnz << std::endl;

    int nnz_c = 0;

    for (int i = 1; i <= dim; i++)
    {
      int row_start = row_ptr[i - 1];
      int row_end = row_ptr[i];

      nnz_c += row_end - row_start;
      // printf("%d, %d, %d\n", row_start, row_end, row_end - row_start);

      for (int j = row_start; j < row_end; j++)
      {
        //   printf("%d ", col_ind[j]);
        mat_file << i << "\t" << col_ind[j] + 1 << "\t" << val[cont] << std::endl;
        cont++;
      }
    }
    mat_file.close();
    printf("%d, %d==%d\n", dim, nnz_c, nnz);
  }

  static void save_vec_as_csv(std::string prefix, int iter, int dim, int nnz, const int* row_ptr, const int* col_ind, double* val)
  {
  }

  static void save_vec_as_mm(std::string prefix, int iter, int dim, double* val)
  {
    printf("%s\t%d\n", __FILE__, __LINE__);
    int cont = 0;

    std::stringstream ss;
    ss << prefix << "-rhs-" << iter << ".mtx";
    std::string filename = ss.str();
    std::cout << "Filename: " << filename << std::endl;

    std::ofstream vec_file(filename);

    vec_file << "%%MatrixMarket matrix array real general" << std::endl;
    vec_file << "% ID: " << iter << std::endl;
    vec_file << dim << "\t" << 1 << std::endl;

    int nnz_c = 0;

    for (int i = 0; i < dim; i++)
    {
      vec_file << val[i] << std::endl;
    }
    vec_file.close();
  }
};

} // namespace debug
#endif
