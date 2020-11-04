/*
 * Prove.cpp
 *
 *  Created on: 24/gen/2019
 *      Author: cgarofalo
 */
#include "Funzioni.h"
//#include "Classi.h"
//#include "LinearAlgebra.h"
#include "NeuralNetwork.h"


using namespace neuralnetworks;

double square( double x) {
    return x*x;
}

void print_square(double x) {
    std::cout<<" The square root of "<<x<<" is "<<square(x)<<"\n";
}

int main() {
    

    int* conf = new int[3] {2,2,2};
    
    vector<vector<double>*>* trainingSet = new vector<vector<double>*>();
    trainingSet->reserve(10);
    trainingSet->push_back(new vector<double>{2.7810836,2.550537003,0});
    trainingSet->push_back(new vector<double>{1.465489372,2.362125076,0});
    trainingSet->push_back(new vector<double>{3.396561688,4.400293529,0});
    trainingSet->push_back(new vector<double>{1.38807019,1.850220317,0});
    trainingSet->push_back(new vector<double>{3.06407232,3.005305973,0});
    trainingSet->push_back(new vector<double>{7.627531214,2.759262235,1});
    trainingSet->push_back(new vector<double>{5.332441248,2.088626775,1});
    trainingSet->push_back(new vector<double>{6.922596716,1.77106367,1});
    trainingSet->push_back(new vector<double>{8.675418651,-0.242068655,1});
    trainingSet->push_back(new vector<double>{7.673756466,3.508563011,1});
    
    NeuralNetwork* network = new NeuralNetwork(conf,2);
    
    network->trainNetwork(trainingSet, 0.5, 40, 2);
    
    vector<double>* inps = new vector<double>{6.922596716,1.77106367};
    int o = network->fit(inps);
    cout<<"\nFit\nexpected = 1 out = "<<o;
    free(inps);
    inps = new vector<double>{3.396561688,4.400293529};
    o = network->fit(inps);
    cout<<"\nFit\nexpected = 0 out = "<<o<<"\n";
    free(inps);
    
    cout<<"\nSecond train\n";
    trainingSet->clear();
    trainingSet->push_back(new vector<double>{0,0,0});
    trainingSet->push_back(new vector<double>{0,1,1});
    trainingSet->push_back(new vector<double>{1,0,1});
    trainingSet->push_back(new vector<double>{1,1,0});
    trainingSet->push_back(new vector<double>{0,0,0});
    trainingSet->push_back(new vector<double>{0,1,1});
    trainingSet->push_back(new vector<double>{1,0,1});
    trainingSet->push_back(new vector<double>{1,1,0});
    trainingSet->push_back(new vector<double>{0,0,0});
    trainingSet->push_back(new vector<double>{0,1,1});
    trainingSet->push_back(new vector<double>{1,0,1});
    trainingSet->push_back(new vector<double>{1,1,0});
    trainingSet->push_back(new vector<double>{0,0,0});
    trainingSet->push_back(new vector<double>{0,1,1});
    trainingSet->push_back(new vector<double>{1,0,1});
    trainingSet->push_back(new vector<double>{1,1,0});
    trainingSet->push_back(new vector<double>{0,0,0});
    trainingSet->push_back(new vector<double>{0,1,1});
    trainingSet->push_back(new vector<double>{1,0,1});
    trainingSet->push_back(new vector<double>{1,1,0});
    trainingSet->push_back(new vector<double>{0,0,0});
    trainingSet->push_back(new vector<double>{0,1,1});
    trainingSet->push_back(new vector<double>{1,0,1});
    trainingSet->push_back(new vector<double>{1,1,0});
    trainingSet->push_back(new vector<double>{0,0,0});
    trainingSet->push_back(new vector<double>{0,1,1});
    trainingSet->push_back(new vector<double>{1,0,1});
    trainingSet->push_back(new vector<double>{1,1,0});
    trainingSet->push_back(new vector<double>{0,0,0});
    trainingSet->push_back(new vector<double>{0,1,1});
    trainingSet->push_back(new vector<double>{1,0,1});
    trainingSet->push_back(new vector<double>{1,1,0});
    trainingSet->push_back(new vector<double>{0,0,0});
    trainingSet->push_back(new vector<double>{0,1,1});
    trainingSet->push_back(new vector<double>{1,0,1});
    trainingSet->push_back(new vector<double>{1,1,0});
    trainingSet->push_back(new vector<double>{0,0,0});
    trainingSet->push_back(new vector<double>{0,1,1});
    trainingSet->push_back(new vector<double>{1,0,1});
    trainingSet->push_back(new vector<double>{1,1,0});
    
    conf = new int[5] {2,3,4,4,2};
    network = new NeuralNetwork(conf,4);
    network->trainNetwork(trainingSet, 0.7, 2000, 2);
    
    inps = new vector<double>{0,1};
    o = network->fit(inps);
    cout<<"\nFit - xor\nexpected = 1 out = "<<o<<"\n";
    inps = new vector<double>{1,0};
    o = network->fit(inps);
    cout<<"\nexpected = 1 out = "<<o<<"\n";
    inps = new vector<double>{1,1};
    o = network->fit(inps);
    cout<<"\nexpected = 0 out = "<<o<<"\n";
    inps = new vector<double>{0,0};
    o = network->fit(inps);
    cout<<"\nexpected = 0 out = "<<o<<"\n";
    
    cout<<"\n-----end------\n";
}



