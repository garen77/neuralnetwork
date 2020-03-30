#pragma once

#include <iostream>

namespace linalg {
    template <typename T>
    class Matrix;

    template <typename T>
    Matrix<T>* operator*(Matrix<T>& a, Matrix<T>& b);

    template <typename T>
    class Matrix {
    private:
        int numRows;
        int numCols;
        T** elements;

    public:
        Matrix(int nr, int nc);

        T** getElements();
        int getNumRows(void);
        int getNumCols(void);

        void print(void);

        friend Matrix<T>* operator*(Matrix<T>& a, Matrix<T>& b) {
            int nr = a.getNumRows();
            int nc = b.getNumCols();
            int nca = a.getNumCols();
            int nrb = b.getNumRows();
            linalg::Matrix<T>* res = new linalg::Matrix<T>(nr, nc);
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
    };

    template <typename T>
    Matrix<T>::Matrix(int nr, int nc) :numRows(nr), numCols(nc) {
        this->elements = new T * [numRows];
        for (int i = 0; i < numRows; i++) {
            this->elements[i] = new T[numCols];
            for (int j = 0; j < numCols; j++) {
                this->elements[i][j] = 0.0;
            }
        }
    }

    template <typename T>
    T** Matrix<T>::getElements() {
        return this->elements;
    }

    template <typename T>
    int Matrix<T>::getNumRows() {
        return numRows;
    }

    template <typename T>
    int Matrix<T>::getNumCols() {
        return numCols;
    }

    template <typename T>
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