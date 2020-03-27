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

nn::FullyConnected::FullyConnected(int ni, int no):numInputs(ni), numOutputs(no) {
	this->weights = new linalg::Matrix(numOutputs, numInputs);	
	for (int i = 0; i < numOutputs; i++) {
		for (int j = 0; j < numInputs; j++) {
			this->weights->getElements()[i][j] = ((double)rand() / (RAND_MAX));
		}
	}
}

linalg::Matrix* nn::FullyConnected::getWeights() {
	return this->weights;
}

linalg::Matrix* nn::FullyConnected::getInput() {
	return this->input;
}

linalg::Matrix* nn::FullyConnected::getOutput() {
	return this->output;
}

linalg::Matrix* nn::FullyConnected::feedForward(linalg::Matrix* input) {
	this->input = input;
	this->output = (*this->weights) * (*input);
	return this->output;
}

void nn::FullyConnected::backPropagate(linalg::Matrix& expected, double learningRate) {
	int nr = this->numOutputs;
	int nc = this->numInputs;
	linalg::Matrix* weightsMatrix = this->getWeights();
	double diffOut = 0.0;
	for (int i = 0; i < nr; i++) {
		diffOut += expected.getElements()[0][i] - this->output->getElements()[0][i];
	}
	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < nc; j++) {
			double w = weightsMatrix->getElements()[i][j];
			double x = 
			w = w + learningRate * diffOut * this->input->getElements()[0][j];
		}
	}
}