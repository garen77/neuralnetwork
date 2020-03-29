#pragma once

#include <vector>

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

    
}

namespace nn {

	class FullyConnected {

	private:
		linalg::Matrix<double>* weights;
		int numInputs, numOutputs;
		linalg::Matrix<double>* input;
		linalg::Matrix<double>* output;

		FullyConnected* previousLayer;

	public:

		FullyConnected(int ni, int no);

		linalg::Matrix<double>* getWeights();
		linalg::Matrix<double>* getInput();
		linalg::Matrix<double>* getOutput();

		linalg::Matrix<double>* feedForward(linalg::Matrix<double>* input);
		void backPropagate(linalg::Matrix<double>& expected, double learningRate);

	};

	class NeuralNet {

	private:
        std::vector<FullyConnected>* layers;
        int numOfLayers;
        
    public:
        NeuralNet(int nl);
        
        void learn(linalg::Matrix<double>& trainingSet);
	};
}
