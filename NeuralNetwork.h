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
		double** W;
		int ni, no;

	public:

		FullyConnected(int numInputs, int numOutputs);


		void train(double** trainIns, double** trainOuts, const int inSize, int outSize, int trainSize, int numEpochs, double rateLearning);
	};
}