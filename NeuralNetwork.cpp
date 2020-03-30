

/*#include "NeuralNetwork.h"
#include <iostream>



template <typename T>
linalg::Matrix<T>::Matrix(int nr, int nc) :numRows(nr), numCols(nc) {
	this->elements = new T* [numRows];
	for (int i = 0; i < numRows; i++) {
		this->elements[i] = new T[numCols];
		for (int j = 0; j < numCols; j++) {
			this->elements[i][j] = 0.0;
		}
	}
}

template <typename T>
T** linalg::Matrix<T>::getElements() {
	return this->elements;
}

template <typename T>
int linalg::Matrix<T>::getNumRows() {
	return numRows;
}

template <typename T>
int linalg::Matrix<T>::getNumCols() {
	return numCols;
}

template <typename T>
void linalg::Matrix<T>::print(void) {
	std::cout << "\nMatrix: \n";
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			std::cout << "[" << i << "][" << j << "] = " << this->elements[i][j] << " ";
		}
		std::cout << "\n";
	}
}



void nn::FullyConnected::init(int ni, int no) {
    this->numInputs = ni;
    this->numOutputs = no;
    this->weights = new linalg::Matrix<double>(no, ni);
    for (int i = 0; i < no; i++) {
        for (int j = 0; j < ni; j++) {
            this->weights->getElements()[i][j] = ((double)rand() / (RAND_MAX));
        }
    }
}

nn::FullyConnected::FullyConnected(int ni, int no) :numInputs(ni), numOutputs(no) {
    this->weights = new linalg::Matrix<double>(numOutputs, numInputs);
    for (int i = 0; i < numOutputs; i++) {
        for (int j = 0; j < numInputs; j++) {
            this->weights->getElements()[i][j] = ((double)rand() / (RAND_MAX));
        }
    }
}

linalg::Matrix<double>* nn::FullyConnected::getWeights() {
    return this->weights;
}

linalg::Matrix<double>* nn::FullyConnected::getInput() {
    return this->input;
}

linalg::Matrix<double>* nn::FullyConnected::getOutput() {
    return this->output;
}

linalg::Matrix<double>* nn::FullyConnected::feedForward(linalg::Matrix<double>* input) {
    this->input = input;
    this->output = (*this->weights) * (*input);
    return this->getOutput();
}


void nn::FullyConnected::backPropagate(linalg::Matrix<double>& expected, double learningRate) {
    int nr = this->numOutputs;
    int nc = this->numInputs;
    linalg::Matrix<double>* weightsMatrix = this->getWeights();
    double diffOut = 0.0;
    for (int i = 0; i < nr; i++) {
        diffOut += expected.getElements()[0][i] - this->output->getElements()[0][i];
    }
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
            double w = weightsMatrix->getElements()[i][j];
            double x = this->input->getElements()[0][j];
            w = w + learningRate * diffOut * x;
            weightsMatrix->getElements()[i][j] = w;
        }
    }
}


nn::NeuralNet::NeuralNet(int nl) :numOfLayers(nl) {

    this->layers = new std::vector<nn::FullyConnected*>(nl);
    for (int i = 0; i < nl; i++) {
        nn::FullyConnected* fc = new nn::FullyConnected(0, 0);
        this->layers->push_back(fc);
    }

}

void nn::NeuralNet::learn(linalg::Matrix<double>& trainingSet) {
    /*
     training set
     [[x1,y1],[x2,y2],...,[xn,yn]]
     */
    /*int numOfSamples = trainingSet.getNumCols();
    for (int i = 0; i < numOfSamples; i++) {
        // forward phase
        for (nn::FullyConnected* fc : (*this->layers)) {

            fc->feedForward(&trainingSet);

        }
    }
}


*/