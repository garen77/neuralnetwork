#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <iostream>

using namespace std;


namespace neuralnetworks {

    class Neuron {

    private:

        int numInputs;
        vector<double>* w;
        double output;

    public:

        Neuron(int n);
        Neuron(int n, double(*activ)(double));

        int getNumInputs();
        vector<double>* getWeights();
        double(*activation)(double inp);
        double activate(vector<double>* x);

        void print();

    };

    class NeuralNetwork {

    private:
        vector<vector<unordered_map<string, void*>*>*>* layers;
        vector<vector<Neuron*>*>* _layers;

        int numOfLayers;
        int* configurazione;

    public:
        NeuralNetwork(int* conf, int nl);

        double activate(vector<double>* weights, vector<double>* inputs);
        vector<double>* forwardPropagate(vector<double>* inputs);
        void backPropagate(vector<double>* expected);
        void updateWeights(vector<double>* inputs, double lr);

        void trainNetwork(vector<vector<double>*>* trainingSet, double lr, int numEpochs, int numOutputs);

        int fit(vector<double>* inputs);
    };

}