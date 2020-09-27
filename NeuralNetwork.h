#pragma once


#include <vector>

#include "LinearAlgebra.h"
#include <cmath>
#include <cstdlib>

bool isLogActive = true;

double relu(double inp) {
    return inp > 0 ? inp : 0;
}

double linear(double inp) {
    return inp;
}

double sigmoid(double inp) {
    return 1/(1 + std::exp(-inp));
}

namespace nn {

    class FullyConnected {

    private:
        linalg::Matrix<double>* weights;
        int numInputs, numOutputs;
        linalg::Matrix<double>* input;
        linalg::Matrix<double>* output;
        linalg::Matrix<double>* biases;
        
        FullyConnected* previousLayer;
        double(*activation)(double inp);
        
    public:
    
        FullyConnected(int ni, int no);
        
        FullyConnected(int ni, int no, double(*activ)(double));

        
        
        void init(int ni, int no);

        linalg::Matrix<double>* getWeights();
        linalg::Matrix<double>* getInput();
        linalg::Matrix<double>* getOutput();

        linalg::Matrix<double>* feedForward(linalg::Matrix<double>* input);
        linalg::Matrix<double>* backPropagate(linalg::Matrix<double>* expected, double learningRate);

    };

    class NeuralNet {

    private:
        std::vector<FullyConnected*>* layers;
        int numOfLayers;
        int** configurazione;

    public:
        NeuralNet(int** conf, int nl);

        void learn(linalg::Matrix<linalg::Matrix<linalg::Matrix<double>*>*>* trainingSet);
        linalg::Matrix<double>* fit(linalg::Matrix<double>* x);

    };


    void FullyConnected::init(int ni, int no) {
        this->numInputs = ni;
        this->numOutputs = no;
        this->weights = new linalg::Matrix<double>(linalg::MatrixType::Numeric, no, ni);
        this->biases = new linalg::Matrix<double>(linalg::MatrixType::Numeric, no, 1);
        
        for (int i = 0; i < no; i++) {
            for (int j = 0; j < ni; j++) {
                this->weights->getElements()[i][j] = ((double)((rand() % 100 + 1)/100));
            }
            this->biases->getElements()[i][0] = ((double)((rand() % 100 + 1)/100));
        }
    }

    FullyConnected::FullyConnected(int ni, int no) :numInputs(ni), numOutputs(no) {
        this->activation = &sigmoid;
        this->weights = new linalg::Matrix<double>(linalg::MatrixType::Numeric, numOutputs, numInputs);
        this->biases = new linalg::Matrix<double>(linalg::MatrixType::Numeric, numOutputs, 1);
        
        //std::cout << "\nexample rand() : " << ((double)((double)(rand() % 100 + 1)/(double)100 )) << "\n";
        for (int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < numInputs; j++) {
                this->weights->getElements()[i][j] = ((double)((double)(rand() % 100 + 1) / (double)100));
            }
            this->biases->getElements()[i][0] = ((double)((double)(rand() % 100 + 1) / (double)100));
        }
    }

    linalg::Matrix<double>* FullyConnected::getWeights() {
        return this->weights;
    }

    linalg::Matrix<double>* FullyConnected::getInput() {
        return this->input;
    }

    linalg::Matrix<double>* FullyConnected::getOutput() {
        return this->output;
    }

    linalg::Matrix<double>* FullyConnected::feedForward(linalg::Matrix<double>* input) {
        this->input = input;
        //o = W*i
        this->output = (*this->weights) * (*input);
        //o = o + b
        for (int i=0; i<this->numOutputs; i++) {
            this->output->getElements()[i][0] = this->output->getElements()[i][0] + this->biases->getElements()[i][0];
        }
        
        for (int i=0; i<this->numOutputs; i++) {
            double o = this->output->getElements()[i][0];
            o = (this->activation)(o);//1/(1 + std::exp(-o));
            if (isLogActive) {
                std::cout << "\no=" << o << " i=" << i << "\n";
            }
            this->output->getElements()[i][0] = o;
        }
    
        std::cout << "\nout";
        std::cout << this->getOutput();
        return this->getOutput();
    }


    linalg::Matrix<double>* FullyConnected::backPropagate(linalg::Matrix<double>* expected, double learningRate) {
        int nr = this->numOutputs;
        int nc = this->numInputs;
        linalg::Matrix<double>* weightsMatrix = this->getWeights();
        
        double diffOut = 0.0;
        for (int i = 0; i < nr; i++) {
            diffOut += expected->getElements()[0][i] - this->output->getElements()[0][i];
        }
        for (int i = 0; i < nr; i++) {
            double o = this->output->getElements()[0][i];
            for (int j = 0; j < nc; j++) {
                double w = weightsMatrix->getElements()[i][j];
                double x = this->input->getElements()[j][0];
                
                w = w - learningRate * diffOut * x * o * (1 - o);
                weightsMatrix->getElements()[i][j] = w;
            }
        }
        if (isLogActive) {
            std::cout << "\nback propagation";
            std::cout << this->getWeights();
        }


        return this->getInput();
    }


    NeuralNet::NeuralNet(int** conf, int nl) :configurazione(conf), numOfLayers(nl) {
        /*
         [[ni1,no1],[ni2,no2],...,[nik,nok]]
        */
       
        this->layers = new std::vector<FullyConnected*>();
        this->layers->reserve(nl);
        for (int i = 0; i < nl; i++) {
            int ni = this->configurazione[i][0];
            int no = this->configurazione[i][1];
            FullyConnected* fc = new FullyConnected(ni, no);
            this->layers->push_back(fc);
        }

    }

    void NeuralNet::learn(linalg::Matrix<linalg::Matrix<linalg::Matrix<double>*>*>* trainingSet) {
        /*
         training set
         [[[x1],[y1]],[[x2],[y2]],...,[[xn],[yn]]]
         */
        int numOfSamples = trainingSet->getNumCols();
        for (int i = 0; i < numOfSamples; i++) {
            
            linalg::Matrix<linalg::Matrix<double>*>* sampleIO = trainingSet->getElements()[0][i];

            linalg::Matrix<double>* x = sampleIO->getElements()[0][0];
            linalg::Matrix<double>* y = sampleIO->getElements()[0][1];

            // forward phase
            linalg::Matrix<double>* yTemp = this->layers->front()->feedForward(x);
            
            for(int i = 1; i < this->layers->size(); ++i) {
                yTemp = this->layers->at(i)->feedForward(yTemp);    
            }
            
            // error back propagation
            linalg::Matrix<double>* xTemp = this->layers->back()->backPropagate(y, 0.05);
            for(int j = this->layers->size() - 2; j>=0; --j) {
                xTemp = this->layers->at(i)->backPropagate(xTemp, 0.05);
            }

            
        }
    }

    linalg::Matrix<double>* NeuralNet::fit(linalg::Matrix<double>* x) {
        linalg::Matrix<double>* y = new linalg::Matrix<double>(linalg::MatrixType::Numeric,1,1);
        for (std::vector<FullyConnected*>::iterator it = this->layers->begin(); it != this->layers->end(); ++it) {
            y = (*it)->feedForward(x);
        }
        return y;
    }

}








