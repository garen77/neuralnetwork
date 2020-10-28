#include "NeuralNetwork.h"
//#include "NeuralNetwork.cpp"

int main(int argc, char* argv[]) {

	unordered_map<string, double*>* mappa = new unordered_map<string, double*>();
	double* arr = new double[3];
	arr[0] = 3.4; arr[1] = 3.4; arr[2] = 3.4;
	(*mappa)["w"] = arr;
	std::cout << "mappa : " << (*mappa)["w"][2];


}

