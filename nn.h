#pragma once



namespace neuralnetworks {

    class Neuron {

    private:

        int numInputs;
        double* w;
        double b;

    public:

        Neuron(int n);
        Neuron(int n, double(*activ)(double));

        int getNumInputs();
        double* getWeights();
        double getBias();
        double(*activation)(double inp);
        double output(double* x);

        void print();

    };

}