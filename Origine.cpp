#include "NeuralNetwork.h"
//#include "NeuralNetwork.cpp"

int main(int argc, char* argv[]) {

	linalg::Matrix<double>* m1 = new linalg::Matrix<double>(linalg::MatrixType::Numeric, 2, 2);
	m1->getElements()[0][0] = 1;
	m1->getElements()[0][1] = 2;
	m1->getElements()[1][0] = 3;
	m1->getElements()[1][1] = 1;

	std::cout << m1;
	linalg::Matrix<double>* m2 = new linalg::Matrix<double>(linalg::MatrixType::Numeric, 2, 3);
	m2->getElements()[0][0] = 1;
	m2->getElements()[0][1] = 2;
	m2->getElements()[0][2] = 3;
	m2->getElements()[1][0] = 1;
	m2->getElements()[1][1] = 3;
	m2->getElements()[1][2] = 1;
	std::cout << m2;
	linalg::Matrix<double>* m3 = (*m1) * (*m2);
	std::cout << m3;


	//linalg::Matrix<linalg::Matrix<double>*>* tt = new linalg::Matrix<linalg::Matrix<double>*>(linalg::MatrixType::Matrix, 1, 2);
	
	/*nn::FullyConnected* fc1 = new nn::FullyConnected(3, 2);
	
	linalg::Matrix<double>* neuralInput = new linalg::Matrix<double>(linalg::MatrixType::Numeric, 3, 1);
	neuralInput->getElements()[0][0] = 1;
	neuralInput->getElements()[1][0] = 2;
	neuralInput->getElements()[2][0] = 2;

	neuralInput->print();
	fc1->getWeights()->getElements()[0][0] = 1;
	fc1->getWeights()->getElements()[0][1] = 1;
	fc1->getWeights()->getElements()[0][2] = 1;
	fc1->getWeights()->getElements()[1][0] = 1;
	fc1->getWeights()->getElements()[1][1] = 1;
	fc1->getWeights()->getElements()[1][2] = 1;

	//neuralInput->print();
	fc1->getWeights()->print();

	linalg::Matrix<double>* y1 = fc1->feedForward(neuralInput);
	y1->print();

	nn::FullyConnected* fc2 = new nn::FullyConnected(2, 2);
	fc2->getWeights()->print();
	linalg::Matrix<double>* y = fc2->feedForward(y1);

	y->print();
	*/
	// layers configuration

	int** conf = new int* [1];
	for (int i = 0; i < 3; i++) {
		conf[i] = new int[2];
	}
	conf[0][0] = 2;
	conf[0][1] = 1;

	nn::NeuralNet* neuralNet = new nn::NeuralNet(conf, 1);

	// training set
	linalg::Matrix<linalg::Matrix<linalg::Matrix<double>*>*>* trainingSet = new linalg::Matrix<linalg::Matrix<linalg::Matrix<double>*>*>(linalg::MatrixType::Matrix, 1, 10);
	
	
	for (int i = 0; i < 10; i++) {
		trainingSet->getElements()[0][i] = new linalg::Matrix<linalg::Matrix<double>*>(linalg::MatrixType::Matrix, 1, 2);
		linalg::Matrix<linalg::Matrix<double>*>* xy = static_cast<linalg::Matrix<linalg::Matrix<double>*>*>(trainingSet->getElements()[0][i]);
		xy->getElements()[0][0] = new linalg::Matrix<double>(linalg::MatrixType::Numeric, 2, 1);
		xy->getElements()[0][1] = new linalg::Matrix<double>(linalg::MatrixType::Numeric, 1, 1);
		
		linalg::Matrix<double>* x = static_cast<linalg::Matrix<double>*>(xy->getElements()[0][0]);
		x->getElements()[0][0] = i;
		x->getElements()[0][0] = i + 1;

		linalg::Matrix<double>* y = static_cast<linalg::Matrix<double>*>(xy->getElements()[0][1]);
		y->getElements()[0][0] = i + i + 1;
	}
	
	linalg::Matrix<double>* x = new linalg::Matrix<double>(linalg::MatrixType::Numeric, 2, 1);

	x->getElements()[0][0] = 1;
	x->getElements()[0][0] = 2;

	neuralNet->learn(trainingSet);

	linalg::Matrix<double>* y = neuralNet->fit(x);
	std::cout << y;


}

