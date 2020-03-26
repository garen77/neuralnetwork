#pragma once

namespace linalg {

	class Matrix {
	private:
		int numRows;
		int numCols;
		double** elements;

	public:
		Matrix(int nr, int nc);

		friend Matrix* operator*(Matrix& a, Matrix& b);

		double** getElements();
		int getNumRows(void);
		int getNumCols(void);

		void print(void);
	};
}

namespace nn {

	class FullyConnected {

	private:
		linalg::Matrix* weights;
		int numInputs, numOutputs;

		linalg::Matrix* output;

		FullyConnected* previousLayer;

	public:

		FullyConnected(int ni, int no);

		linalg::Matrix* getWeights();
		linalg::Matrix* getOutput();

		linalg::Matrix* feedForward(linalg::Matrix* input);
		void backPropagate(linalg::Matrix* expected);

	};

	class NeuralNet {

	private:

	};
}