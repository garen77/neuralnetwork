#include <iostream>
#include <vector>
#include "NeuralNetwork.h"


linalg::Matrix::Matrix(int nr, int nc) :numRows(nr), numCols(nc) {
	this->elements = new double* [numRows];
	for (int i = 0; i < numRows; i++) {
		this->elements[i] = new double[numCols];
		for (int j = 0; j < numCols; j++) {
			this->elements[i][j] = 0.0;
		}
	}
}

double** linalg::Matrix::getElements() {
	return this->elements;
}

int linalg::Matrix::getNumRows() {
	return numRows;
}

int linalg::Matrix::getNumCols() {
	return numCols;
}

linalg::Matrix* linalg::operator*(linalg::Matrix& a, linalg::Matrix& b) {
	int nr = a.getNumRows();
	int nc = b.getNumCols();
	int nca = a.getNumCols();
	int nrb = b.getNumRows();
	linalg::Matrix* res = new linalg::Matrix(nr, nc);
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

void linalg::Matrix::print(void) {
	std::cout << "\nMatrix: \n";
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			std::cout << "[" << i << "][" << j << "] = " << this->elements[i][j] << " ";
		}
		std::cout << "\n";
	}
}

nn::FullyConnected::FullyConnected(int numInputs, int numOutputs) {
	this->W = new double* [numOutputs];
	this->ni = numInputs;
	this->no = numOutputs;
	for (int i = 0; i < numOutputs; i++) {
		this->W[i] = new double [numInputs];
		for (int j = 0; j < numInputs; j++) {
			this->W[i][j] = ((double)rand() / (RAND_MAX));
		}
	}
}

void nn::FullyConnected::train(double** trainIns, double** trainOuts, const int inSize, int outSize, int trainSize, int numEpochs, double rateLearning) {
	for (int e = 0; e < numEpochs; e++) {
		for (int j = 0; j < trainSize; j++) {
			std::vector<double> sampleIn(inSize);
			std::vector<double> sampleOut(outSize);
			for (int i = 0; i < inSize; i++) {
				sampleIn.push_back(trainIns[i][j]);
			}
			for (int i = 0; i < outSize; i++) {
				sampleOut.push_back(trainIns[i][j]);
			}

		}
	}

}