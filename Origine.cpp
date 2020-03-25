#include "NeuralNetwork.h"

int main(int argc, char* argv[]) {

	linalg::Matrix* m1 = new linalg::Matrix(2, 2);
	m1->getElements()[0][0] = 1;
	m1->getElements()[0][1] = 2;
	m1->getElements()[1][0] = 3;
	m1->getElements()[1][1] = 1;
	m1->print();
	linalg::Matrix* m2 = new linalg::Matrix(2, 3);
	m2->getElements()[0][0] = 1;
	m2->getElements()[0][1] = 2;
	m2->getElements()[0][2] = 3;
	m2->getElements()[1][0] = 1;
	m2->getElements()[1][1] = 3;
	m2->getElements()[1][2] = 1;
	m2->print();
	linalg::Matrix* m3 = (*m1) * (*m2);
	m3->print();

	nn::FullyConnected* fc1 = new nn::FullyConnected(3, 2);
	
	linalg::Matrix* neuralInput = new linalg::Matrix(3, 1);
	neuralInput->getElements()[0][0] = 1;
	neuralInput->getElements()[1][0] = 2;
	neuralInput->getElements()[2][0] = 2;

	neuralInput->print();
	/*fc1->getWeights()->getElements()[0][0] = 1;
	fc1->getWeights()->getElements()[0][1] = 1;
	fc1->getWeights()->getElements()[0][2] = 1;
	fc1->getWeights()->getElements()[1][0] = 1;
	fc1->getWeights()->getElements()[1][1] = 1;
	fc1->getWeights()->getElements()[1][2] = 1;*/

	//neuralInput->print();
	fc1->getWeights()->print();

	linalg::Matrix* y1 = fc1->feedForward(neuralInput);
	y1->print();

	nn::FullyConnected* fc2 = new nn::FullyConnected(2, 2);
	fc2->getWeights()->print();
	linalg::Matrix* y = fc2->feedForward(y1);

	y->print();
}

