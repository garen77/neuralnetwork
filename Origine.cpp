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
}

