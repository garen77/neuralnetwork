#pragma once

#include <iostream>

namespace linalg {

    enum class MatrixType { Numeric, Matrix };

    template <class T>
    class Matrix;

    template <class T>
    Matrix<T>* operator*(Matrix<T>& a, Matrix<T>& b);

    template <class T>
    class Matrix {
    private:
        int numRows;
        int numCols;
        T** elements;
        MatrixType type;

    public:
        Matrix(MatrixType ty, int nr, int nc);

        T** getElements();
        int getNumRows(void);
        int getNumCols(void);
        MatrixType getType();
        void print(void);

        friend Matrix<T>* operator*(Matrix<T>& a, Matrix<T>& b) {
            int nr = a.getNumRows();
            int nc = b.getNumCols();
            int nca = a.getNumCols();
            int nrb = b.getNumRows();
            if (a.getType() == MatrixType::Numeric && b.getType() == MatrixType::Numeric) {

                linalg::Matrix<T>* res = new linalg::Matrix<T>(MatrixType::Numeric, nr, nc);
                if (nca == nrb) {
                    for (int i = 0; i < nr; i++) {
                        for (int j = 0; j < nc; j++) {
                            double sp = 0.0;
                            for (int jj = 0; jj < nrb; jj++) {
                                sp += a.getElements()[i][jj] * b.getElements()[jj][j];
                            }
                            res->elements[i][j] = sp;
                        }
                    }
                }

                return res;
            }
            else {
                return NULL;
            }
        }


        friend std::ostream& operator<<(std::ostream& strm, const Matrix<T>* m) {
            if (m->type == MatrixType::Matrix) {
                strm << m;
            }
            else {
                strm << "\nMatrix: \n";
                for (int i = 0; i < m->numRows; i++) {
                    for (int j = 0; j < m->numCols; j++) {
                        strm << "[" << i << "][" << j << "] = " << m->elements[i][j] << " ";
                    }
                    strm << "\n";
                }
            }
            
            return strm;
        }
    };

    template <class T>
    Matrix<T>::Matrix(MatrixType ty, int nr, int nc): type(ty), numRows(nr), numCols(nc) {
        
        this->elements = new T * [numRows];
        for (int i = 0; i < numRows; i++) {
            this->elements[i] = new T[numCols];
            for (int j = 0; j < numCols; j++) {
                if (this->type == MatrixType::Numeric) {
                    this->elements[i][j] = 0.0;
                }
            }
        }
    }

    template <class T>
    T** Matrix<T>::getElements() {
        return this->elements;
    }

    template <class T>
    int Matrix<T>::getNumRows() {
        return numRows;
    }

    template <class T>
    int Matrix<T>::getNumCols() {
        return numCols;
    }

    template <class T>
    MatrixType Matrix<T>::getType() {
        return this->type;
    }

    template <class T>
    void Matrix<T>::print(void) {
        std::cout << "\nMatrix: \n";
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                std::cout << "[" << i << "][" << j << "] = " << this->elements[i][j] << " ";
            }
            std::cout << "\n";
        }
    }

}