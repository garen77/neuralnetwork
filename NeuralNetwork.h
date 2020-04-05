#pragma once


#include <vector>

#include "LinearAlgebra.h"

namespace nn {

    class FullyConnected {

    private:
        linalg::Matrix<double>* weights;
        int numInputs, numOutputs;
        linalg::Matrix<double>* input;
        linalg::Matrix<double>* output;

        FullyConnected* previousLayer;

    public:
    
        FullyConnected(int ni, int no);

        void init(int ni, int no);

        linalg::Matrix<double>* getWeights();
        linalg::Matrix<double>* getInput();
        linalg::Matrix<double>* getOutput();

        linalg::Matrix<double>* feedForward(linalg::Matrix<double>* input);
        void backPropagate(linalg::Matrix<double>* expected, double learningRate);

    };

    class NeuralNet {

    private:
        std::vector<FullyConnected*>* layers;
        int numOfLayers;
        int** configurazione;

    public:
        NeuralNet(int** conf, int nl);

        void learn(linalg::Matrix<linalg::Matrix<double>*>* trainingSet);
    };


    void FullyConnected::init(int ni, int no) {
        this->numInputs = ni;
        this->numOutputs = no;
        this->weights = new linalg::Matrix<double>(linalg::MatrixType::Numeric, no, ni);
        for (int i = 0; i < no; i++) {
            for (int j = 0; j < ni; j++) {
                this->weights->getElements()[i][j] = ((double)rand() / (RAND_MAX));
            }
        }
    }

    FullyConnected::FullyConnected(int ni, int no) :numInputs(ni), numOutputs(no) {
        this->weights = new linalg::Matrix<double>(linalg::MatrixType::Numeric, numOutputs, numInputs);
        for (int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < numInputs; j++) {
                this->weights->getElements()[i][j] = ((double)rand() / (RAND_MAX));
            }
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
        this->output = (*this->weights) * (*input);
        return this->getOutput();
    }


    void FullyConnected::backPropagate(linalg::Matrix<double>* expected, double learningRate) {
        int nr = this->numOutputs;
        int nc = this->numInputs;
        linalg::Matrix<double>* weightsMatrix = this->getWeights();
        double diffOut = 0.0;
        for (int i = 0; i < nr; i++) {
            diffOut += expected->getElements()[0][i] - this->output->getElements()[0][i];
        }
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < nc; j++) {
                double w = weightsMatrix->getElements()[i][j];
                double x = this->input->getElements()[0][j];
                w = w + learningRate * diffOut * x;
                weightsMatrix->getElements()[i][j] = w;
            }
        }
    }


    NeuralNet::NeuralNet(int** conf, int nl) :configurazione(conf), numOfLayers(nl) {
        /*
         [[ni1,no1],[ni2,no2],...,[nik,nok]]
        */
       
        this->layers = new std::vector<FullyConnected*>(nl);
        for (int i = 0; i < nl; i++) {
            int ni = this->configurazione[i][0];
            int no = this->configurazione[i][1];
            FullyConnected* fc = new FullyConnected(ni, no);
            this->layers->push_back(fc);
        }

    }

    void NeuralNet::learn(linalg::Matrix<linalg::Matrix<double>*>* trainingSet) {
        /*
         training set
         [[x1,y1],[x2,y2],...,[xn,yn]]
         */
        int numOfSamples = trainingSet->getNumCols();
        for (int i = 0; i < numOfSamples; i++) {
            
            linalg::Matrix<double>* sampleIO = trainingSet->getElements()[0][i];
            int numRows = sampleIO->getNumRows();

            double** x = new double*[numRows];
            
            for(int h=0; h<numRows; h++) {
                x[h] = new double[1];
                x[h][0] = sampleIO->getElements()[h][0];
            }
            linalg::Matrix<double>* sampleIn = new linalg::Matrix<double>(linalg::MatrixType::Numeric, x, numRows, 1);
            
            double** y = new double*[numRows];
            for(int h=0; h<numRows; h++) {
                y[h] = new double[1];
                y[h][0] = sampleIO->getElements()[h][1];
            }
            linalg::Matrix<double>* sampleOut = new linalg::Matrix<double>(linalg::MatrixType::Numeric, y, numRows, 1);

            // forward phase
            for(std::vector<FullyConnected*>::iterator it = this->layers->begin(); it != this->layers->end(); ++it) {
                (*it)->feedForward(sampleIn);
            }
            
            // error back propagation
            for(std::vector<FullyConnected*>::reverse_iterator it = this->layers->rbegin(); it != this->layers->rend(); ++it) {
                (*it)->backPropagate(sampleOut, 0.05);
            }
            
        }
    }

}








