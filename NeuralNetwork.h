#pragma once


#include <vector>

#include "LinearAlgebra.h"
#include <cmath>
#include <cstdlib>
#include <unordered_map>

using namespace linearalgebra;
using namespace std;

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

double heaviside(double inp) {
    if(inp < 0) {
        return 0;
    } else {
        return 1;
    }
}


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

    Neuron::Neuron(int n) :numInputs(n) {
        this->w = new double[n];
        this->activation = &heaviside;
        
        for(int i=0; i<n; i++) {
            this->w[i] = (((double) rand()) / (double) RAND_MAX) * (100 +100) - 100;
        }
        this->b = (((double) rand()) / (double) RAND_MAX) * (100 +100) - 100;
    }

    Neuron::Neuron(int n, double(*activ)(double)) :numInputs(n) {
        this->w = new double[n];
        this->activation = activ;
        for(int i=0; i<n; i++) {
            this->w[i] = (((double) rand()) / (double) RAND_MAX) * (100 +100) - 100;
        }
        this->b = (((double) rand()) / (double) RAND_MAX) * (100 +100) - 100;
    }
    
    int Neuron::getNumInputs() {
        return this->numInputs;
    }

    double* Neuron::getWeights() {
        return this->w;
    }
    
    double Neuron::getBias() {
        return this->b;
    }

    void Neuron::print() {
        std::cout << "w=[";
        int n = this->numInputs;
        for(int i=0; i<n; i++) {
            std::cout << this->w[i];
            if(i != n-1) {
                std::cout << ",";
            }
        }
        std::cout << "] b="<<this->b<<"\n";
        
    }

    double Neuron::output(double* x) {
        int n = this->numInputs;
        double res = 0.0;
        for(int i=0; i<n; i++) {
            res += x[i] * this->w[i];
        }
        res += this->b;
        return this->activation(res);
    }

    class FullyConnected {

    private:
        int numInputs, numOutputs;
        double** w;
        double* b;
        double** dCdW;
        double* dCdb;
        FullyConnected* previousLayer;
        double(*activation)(double inp);
        
    public:
        FullyConnected(int ni, int no);
        FullyConnected(int ni, int no, double(*activ)(double));
        
        int getNumInputs();
        int getNumOutputs();
        double** getWeights();
        double* getBiases();
        double** getdCdW();
        double* getdBdW();
        double* feedForward(double* input);
        double* backPropagate(double* deltaLnext, double** wLNext, int nrNext, double*outL, double* outPrevious);
        
    };

    int FullyConnected::getNumInputs() {
        return this->numInputs;
    }

    int FullyConnected::getNumOutputs() {
        return this->numOutputs;
    }

    FullyConnected::FullyConnected(int ni, int no) :numInputs(ni), numOutputs(no) {
        this->activation = &sigmoid;
        double** rows = new double*[no];
        for(int i=0; i<no; i++) {
            rows[i] = new double[ni];
        }
        this->w = rows;
        this->b = new double[no];
        
        for (int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < numInputs; j++) {
                this->w[i][j] = (((double) rand()) / (double) RAND_MAX) * (1 + 1) - 1;
                if (isLogActive) {
                    std::cout << "\nw["<<i<<"]["<<j<<"] = "<<w[i][j];
                }
            }
            if (isLogActive) {
                std::cout << "\n";
            }
            
            this->b[i] = (((double) rand()) / (double) RAND_MAX) * (1 + 1) - 1;
            if (isLogActive) {
                std::cout << "\nb["<<i<<"] = "<<this->b[i];
            }
        }
    }

    double** FullyConnected::getWeights() {
        return this->w;
    }

    double* FullyConnected::getBiases() {
        return this->b;
    }

    double** FullyConnected::getdCdW() {
        return this->dCdW;
    }

    double* FullyConnected::getdBdW() {
        return this->dCdb;
    }

    double* FullyConnected::feedForward(double* input) {

        if (isLogActive) {
            std::cout << "\nfeed forwad - start";
        }

        //o = W*i
        int nr = this->numOutputs;
        int nc = this->numInputs;
        double* o = new double[nr];
        
        for (int k = 0; k < nr; k++) {
            for (int i = 0; i < nr; i++) {
                double sp = 0.0;
                for (int j = 0; j < nc; j++) {
                    double w = this->w[i][j];
                    if (isLogActive) {
                         std::cout << "\nw["<<i<<"]["<<j<<"] = "<<w;
                         std::cout << "\nx["<<j<<"] = "<<input[j];
                     }
                    sp += w * input[j];
                }
                o[i] = sp;
            }
            if (isLogActive) {
                std::cout << "\n";
            }
        }
        
        //o = o + b
        for (int i=0; i<nr; i++) {
            o[i] = o[i] + this->b[i];
            if (isLogActive) {
                std::cout << "\nb["<<i<<"]=" << this->b[i]<<"\n";
            }
        }
        
        //o = activation(o)
        for (int i=0; i<nr; i++) {
            double z = o[i];
            z = (this->activation)(z);//1/(1 + std::exp(-o));
            if (isLogActive) {
                std::cout << "\no["<<i<<"]=" << z << " i=" << i << "\n";
            }
            o[i] = z;
        }

        if (isLogActive) {
            std::cout << "\nfeed forwad - end";
        }
        
        return o;
    }

    double* FullyConnected::backPropagate(double* deltaLnext, double** wLNext, int nrNext, double* outL, double* outPrevious) {
        
        if (isLogActive) {
            std::cout << "\nback propagation - start";
        }
        int nr = this->numOutputs;
        int nc = this->numInputs;
        
        // WLnexTrasp * deltaLnext
        double* wLtraspDeltaLnext = new double[nr];
        for (int j=0; j<nr; j++) {
            double wLnextDeltaLnext = 0;
            for (int i=0; i<nrNext; i++) {
                wLnextDeltaLnext += deltaLnext[i] * wLNext[i][j];
                if (isLogActive) {
                    std::cout << "\ndeltaLnext[i] = "<<deltaLnext[i]<<" wLNext[i][j] = "<<wLNext[i][j];
                }
            }
            wLtraspDeltaLnext[j] = wLnextDeltaLnext;

        }
        
        // derivative of l-th output layer (this->getInput())
        double* derSigmaOutL = new double[nr];
        for (int i=0; i<nr; i++) {
            double outLi = outL[i];
            derSigmaOutL[i] = outLi * (1 - outLi);
        }
        
        // Wt * deltaLnext product wise derivative of l-th output layer
        double* deltaL = new double[nr];
        for (int i=0; i<nr; i++) {
            double wLtraspDeltaLnexti = wLtraspDeltaLnext[i];
            double derSigmaOutLi = derSigmaOutL[i];
            deltaL[i] = wLtraspDeltaLnexti * derSigmaOutLi;
            if (isLogActive) {
                std::cout << "\nwLtraspDeltaLnexti = "<<wLtraspDeltaLnexti<<" derSigmaOutLi = "<<derSigmaOutLi;
            }
        }
        
        this->dCdW = new double*[nr];
        for(int i=0; i<nr; i++) {
            this->dCdW[i] = new double[nc];
        }
        this->dCdb = new double[nr];
        
        
        for (int i=0; i<nr; i++) {
            double deltaLi = deltaL[i];
            for (int j=0; j<nc; j++) {
                double outPrevj = outPrevious[j];
                this->dCdW[i][j] = deltaLi * outPrevj;
                if (isLogActive) {
                    std::cout << "\noutPrevj = "<<outPrevj<<" deltaLi = "<<deltaLi;
                    std::cout << "\ndCdW["<<i<<"]["<<j<<"]="<<this->dCdW[i][j]<<" ";
                }
            }
            std::cout << "\n";
        }
        
        for (int i=0; i<nr; i++) {
            this->dCdb[i] = deltaL[i];
        }
        
        
        if (isLogActive) {
            std::cout << "\nback propagation - end";
        }


        return deltaL;
    }

    class NeuralNet {

    private:
        std::vector<FullyConnected*>* layers;
        int numOfLayers;
        int** configurazione;

    public:
        NeuralNet(int** conf, int nl);

        void learn(double*** trainingSet, int numOfSamples, int numOfEpochs);
        double* fit(double* x);

    };

    NeuralNet::NeuralNet(int** conf, int nl) :configurazione(conf), numOfLayers(nl) {
        /*
         [[ni1,no1],[ni2,no2],...,[nik,nok]]
        */
       
        this->layers = new std::vector<FullyConnected*>();
        this->layers->reserve(nl);
        for (int i = 0; i < nl; i++) {
            int ni = this->configurazione[i][0];
            int no = this->configurazione[i][1];
            FullyConnected* fc = new FullyConnected(ni, no );
            this->layers->push_back(fc);
        }
    }

    void NeuralNet::learn(double*** trainingSet, int numOfSamples, int numOfEpochs) {
        /*
         training set
         [[[x1],[y1]],[[x2],[y2]],...,[[xn],[yn]]]
         */
        if (isLogActive) {
            std::cout << "\nlearn - start";
        }
        for(int e=0; e<numOfEpochs; e++) {
            for (int i = 0; i < numOfSamples; i++) {
                
                double** sampleIO = trainingSet[i];

                double* x = sampleIO[0];
                double* t = sampleIO[1];

                // forward phase
                int l = this->layers->size();
                double** outputs = new double*[l + 1];
                double* o = x;
                outputs[0] = x;
                for(int i = 0; i < l; ++i) {
                    o = this->layers->at(i)->feedForward(o);
                    outputs[i + 1] = o;
                }
                

                
                // compute local gradients vector of outputlayer
                int no = this->layers->at(l - 1)->getNumOutputs();
                if (isLogActive) {
                    std::cout << "\nno = "<<no<<"\n";
                }
                double* deltaLnext = new double[no];
                for(int j = 0; j < no; j++) {
                    double oj = o[j];
                    double tj = t[j];
                    deltaLnext[j] = (oj - tj)*oj*(1 - oj);
                    if (isLogActive) {
                        std::cout << "\noj = "<<oj<<" tj = "<<tj<<"\n";
                    }
                }
                //double* deltaLnext, double** wLNext, int nrNext, double* outL, double* outPrevious
                double** wLnext = this->layers->at(l - 1)->getWeights();
                int nrNext = this->layers->at(l - 1)->getNumOutputs();
                // error back propagation
                deltaLnext = this->layers->back()->backPropagate(deltaLnext, wLnext, nrNext, o, outputs[l-1]);
                for(int j = l - 2; j>=1; --j) {
                    FullyConnected* layer = this->layers->at(i);
                    wLnext = layer->getWeights();
                    nrNext = layer->getNumOutputs();
                    deltaLnext = layer->backPropagate(deltaLnext, wLnext, nrNext, outputs[j], outputs[j-1]);
                }
                
                // weights update
                double learningRate = 0.01;

                for(int l = this->layers->size() - 1; l>=0; --l) {
                    if (isLogActive) {
                        std::cout << "\nweights update\n";
                    }
                    FullyConnected* layer = this->layers->at(l);
                    int nr = layer->getNumOutputs();
                    int nc = layer->getNumInputs();
                    for (int i=0; i<nr; i++) {
                        double b = layer->getBiases()[i];
                        double dBdW = layer->getdBdW()[i];
                        layer->getBiases()[i] = b - learningRate * dBdW;

                        for (int j=0; j<nc; j++) {
                            double w = layer->getWeights()[i][j];
                            double dCdW = layer->getdCdW()[i][j];
                            layer->getWeights()[i][j] = w - learningRate * dCdW;
                            if (isLogActive) {
                                std::cout << "w = "<<w<<", dCdW =  "<<dCdW<<", "<<"dBdW = "<<dBdW<<"\n";
                            }
                        }
                    }
                }
                
            }
        }
        if (isLogActive) {
            std::cout << "\nlearn - end";
        }
    }

    double* NeuralNet::fit(double* x) {
        double* y = NULL;
        for (std::vector<FullyConnected*>::iterator it = this->layers->begin(); it != this->layers->end(); ++it) {
            y = (*it)->feedForward(x);
        }
        return y;
    }

    class NeuralNetwork {

    private:
        vector<vector<unordered_map<string,void*>*>*>* layers;
        
        int numOfLayers;
        int* configurazione;

    public:
        NeuralNetwork(int* conf, int nl);

        double activate(vector<double>* weights, vector<double>* inputs);
        vector<double>* forwardPropagate(vector<double>* inputs);
        void backPropagate(vector<double>* expected);
        void updateWeights(vector<double>* inputs, double lr);
        
        void trainNetwork(vector<vector<double>*>* trainingSet, double lr, int numEpochs, int numOutputs);

    };

    NeuralNetwork::NeuralNetwork(int* conf, int nl) :configurazione(conf), numOfLayers(nl) {
        
        /*
         nl : num layers
         [niumInput, numHiddenNeuron,..,numHiddenNeuron, numOutNeuron] -> [nl + 1]
        */
       
        this->layers = new vector<vector<unordered_map<string,void*>*>*>();
        this->layers->reserve(nl);
        for (int l = 1; l < nl + 1; l++) {
            int numInputs = this->configurazione[l-1];
            int numOutputs = this->configurazione[l];
            
            vector<unordered_map<string,void*>*>* layer = new vector<unordered_map<string,void*>*>();
            layer->reserve(numOutputs);
            
            for (int i=0; i<numOutputs; i++) {
                unordered_map<string,void*>* neuron = new unordered_map<string,void*>();
                vector<double>* weights = new vector<double>();
                weights->reserve(numInputs + 1);
                for (int j=0; j<numInputs; j++) {
                    weights->push_back((((double) rand()) / (double) RAND_MAX) * (1 + 1) - 1);
                }
                weights->push_back((((double) rand()) / (double) RAND_MAX) * (1 + 1) - 1);
                (*neuron)["weights"] = weights;
                layer->push_back(neuron);
            }
            
            this->layers->push_back(layer);
            
        }
    }

    double NeuralNetwork::activate(vector<double>* weights, vector<double>* inputs) {
        double sum = 0.0;
        int wSize = weights->size();
        for (int i=0; i<wSize; i++) {
            sum += weights->at(i)*inputs->at(i);
        }
        sum += weights->at(wSize); // bias sum
        return sigmoid(sum);
    }

    vector<double>* NeuralNetwork::forwardPropagate(vector<double>* inputs) {
        vector<double>* currInputs = new vector<double>(*inputs);
        for(int l=0; l<this->numOfLayers; l++) {
            vector<unordered_map<string,void*>*>* layer = this->layers->at(l);
            int layerSize = layer->size();
            vector<double>* newInputs = new vector<double>();
            newInputs->reserve(layerSize);
            for(int n=0; n<layerSize; n++) {
                unordered_map<string,void*>* neuron = layer->at(n);
                double neuronOut = this->activate(static_cast<vector<double>*>((*neuron)["weights"]),currInputs);
                *(double *)(*neuron)["output"] = neuronOut;
                newInputs->push_back(neuronOut);
            }
            delete currInputs;
            currInputs = newInputs;
        }
        return currInputs;
    }

    void NeuralNetwork::backPropagate(vector<double>* expected) {
        
        for(int i=this->numOfLayers-1; i>=0; i--) {
            vector<unordered_map<string,void*>*>* layer = this->layers->at(i);
            int layerSize = layer->size();
            vector<double>* errors = new vector<double>();
            errors->reserve(layerSize);
            if(i !=  this->numOfLayers - 1) {
                for (int j=0; j<layerSize; j++) {
                    double error = 0.0;
                    vector<unordered_map<string,void*>*>* nextLayer = this->layers->at(i+1);
                    int nextLayerSize = nextLayer->size();
                    for (int n=0; n<nextLayerSize; n++) {
                        unordered_map<string,void*>* neuron = layer->at(n);
                        vector<double>* w = static_cast<vector<double>*>((*neuron)["weights"]);
                        error += w->at(j) * (*(double *)(*neuron)["delta"]) ;
                    }
                    errors->push_back(error);
                }
            } else {
                for (int j=0; j<layerSize; j++) {
                    unordered_map<string,void*>* neuron = layer->at(j);
                    errors->push_back(expected->at(j) - (*(double *)(*neuron)["output"]) );
                    
                }
            }
            for (int j=0; j<layerSize; j++) {
                unordered_map<string,void*>* neuron = layer->at(j);
                double neuronOut = *(double *)(*neuron)["output"];
                *(double *)(*neuron)["delta"] = errors->at(j)*neuronOut*(1-neuronOut);
            }
        }
    }

    void NeuralNetwork::updateWeights(vector<double>* inputs, double lr) {
        vector<double>* currInputs = new vector<double>(*inputs);
        for (int i=0; i<this->numOfLayers; i++) {
            vector<unordered_map<string,void*>*>* layer = this->layers->at(i);
            int layerSize = layer->size();
            if(i!=0) {
                vector<unordered_map<string,void*>*>* previousLayer = this->layers->at(i-1);
                int previuosLayerSize = previousLayer->size();
                currInputs = new vector<double>();
                currInputs->reserve(previuosLayerSize);
                for (int j=0; j<previuosLayerSize; j++) {
                    unordered_map<string,void*>* neuron = previousLayer->at(j);
                    double neuronOut = *(double *)(*neuron)["output"];
                    currInputs->push_back(neuronOut);
                }
            } else {
                for (int n=0; n<layerSize; n++) {
                    unordered_map<string,void*>* neuron = layer->at(n);
                    int currInputsSize = currInputs->size();
                    vector<double>* w = static_cast<vector<double>*>((*neuron)["weights"]);
                    double neuronDelta = *(double *)(*neuron)["delta"];
                    for (int j=0; j<currInputsSize; j++) {
                        w->at(j) = lr*currInputs->at(j)*neuronDelta;
                    }
                    w->at(currInputsSize) = lr*neuronDelta;
                }
            }
        }
    }

    void NeuralNetwork::trainNetwork(vector<vector<double>*>* trainingSet, double lr, int numEpochs, int numOutputs) {
        for(int epoch=0; epoch<numEpochs; epoch++) {
            double sumError = 0.0;
            int trainintSetSize = trainingSet->size();
            for (int i=0; i<trainintSetSize; i++) {
                vector<double>* row = trainingSet->at(i);
                int inputsSize = row->size()-1;
                vector<double>* inputs = new vector<double>();
                inputs->reserve(inputsSize-1);
                for(int j=0; j<inputsSize; j++) {
                    inputs->push_back(row->at(j));
                }
                vector<double>* outputs = this->forwardPropagate(inputs);
            }
        }
    }

}

namespace nn {

    class FullyConnected {

    private:
        linalg::Matrix<double>* weights;
        int numInputs, numOutputs;
        linalg::Matrix<double>* input;
        linalg::Matrix<double>* output;
        linalg::Matrix<double>* biases;
        
        linalg::Matrix<double>* dCdW;
        linalg::Matrix<double>* dCdb;
        
        FullyConnected* previousLayer;
        double(*activation)(double inp);
        
    public:
    
        FullyConnected(int ni, int no);
        
        FullyConnected(int ni, int no, double(*activ)(double));

        void init(int ni, int no);

        int getNumInputs();
        int getNumOutputs();
        
        linalg::Matrix<double>* getWeights();
        linalg::Matrix<double>* getInput();
        linalg::Matrix<double>* getOutput();

        linalg::Matrix<double>* feedForward(linalg::Matrix<double>* input);
        linalg::Matrix<double>* backPropagate(linalg::Matrix<double>* deltaLnext);

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

    int FullyConnected::getNumInputs() {
        return this->numInputs;
    }

    int FullyConnected::getNumOutputs() {
        return this->numOutputs;
    }

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


    linalg::Matrix<double>* FullyConnected::backPropagate(linalg::Matrix<double>* deltaLnext) {
        
        int nr = this->numOutputs;
        int nc = this->numInputs;
        linalg::Matrix<double>* weightsMatrix = this->getWeights();
        
        // Wt * deltaLnext
        linalg::Matrix<double>* wLtraspDeltaLnext = new linalg::Matrix<double>(linalg::MatrixType::Numeric, nc, 1);
        for (int j=0; j<nc; j++) {
            double wLdLnext = 0;
            for (int i=0; i<nr; i++) {
                wLdLnext += deltaLnext->getElements()[i][0] * weightsMatrix->getElements()[i][j];
            }
            wLtraspDeltaLnext->getElements()[0][j] = wLdLnext;
        }
        
        // derivative of l-th output layer (this->getInput())
        linalg::Matrix<double>* zL = this->getInput();
        linalg::Matrix<double>* derSigmaZL = new linalg::Matrix<double>(linalg::MatrixType::Numeric, nc, 1);
        for (int j=0; j<nc; j++) {
            double zLj = zL->getElements()[j][0];
            derSigmaZL->getElements()[j][0] = zLj * (1 - zLj);
        }
        // Wt * deltaLnext product wise derivative of l-th output layer
        linalg::Matrix<double>* deltaL = new linalg::Matrix<double>(linalg::MatrixType::Numeric, nc, 1);
        for (int j=0; j<nc; j++) {
            double wLtraspDeltaLnextj = wLtraspDeltaLnext->getElements()[j][0];
            double derSigmaZLj = derSigmaZL->getElements()[j][0];
            deltaL->getElements()[j][0] = wLtraspDeltaLnextj * derSigmaZLj;
        }
        
        this->dCdW = new linalg::Matrix<double>(linalg::MatrixType::Numeric, nr, nc);
        this->dCdb= new linalg::Matrix<double>(linalg::MatrixType::Numeric, nr, 1);
        
        for (int i=0; i<nr; i++) {
            double outputi = this->getOutput()->getElements()[i][0];
            for (int j=0; j<nc; j++) {
                double deltaLj = deltaL->getElements()[j][0];
                this->dCdW->getElements()[i][j] = deltaLj * outputi;
            }
        }
        
        for (int i=0; i<nr; i++) {
            this->dCdb->getElements()[i][0] = deltaL->getElements()[i][0];
        }
        
        
        if (isLogActive) {
            std::cout << "\nback propagation";
            std::cout << this->getWeights();
        }


        return deltaL;
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
            linalg::Matrix<double>* t = sampleIO->getElements()[0][1];

            // forward phase
            linalg::Matrix<double>* yTemp = this->layers->front()->feedForward(x);
            
            int l = this->layers->size() - 1;
            for(int i = 1; i <= l; ++i) {
                yTemp = this->layers->at(i)->feedForward(yTemp);    
            }
            
            // compute local gradients vector of outputlayer
            int no = this->layers->at(l)->getNumOutputs();
            linalg::Matrix<double>* o = this->layers->at(l)->getOutput();
            
            linalg::Matrix<double>* deltaLnext = new linalg::Matrix<double>(linalg::MatrixType::Numeric, no, 1);
            for(int j = no; j < no; j++) {
                double oj = o->getElements()[0][j];
                double tj = t->getElements()[0][j];
                deltaLnext->getElements()[j][0] = (oj - tj)*oj*(1 - oj);
            }
            
            // error back propagation
            linalg::Matrix<double>* deltaL = this->layers->back()->backPropagate(deltaLnext);
            for(int j = this->layers->size() - 2; j>=0; --j) {
                deltaL = this->layers->at(i)->backPropagate(deltaL);
            }
            
            // weights update
            double learningRate = 0.05;
            for(int l = this->layers->size() - 1; l>=0; --l) {
                FullyConnected* layer = this->layers->at(l);
                int nr = layer->getNumOutputs();
                int nc = layer->getNumInputs();
                for (int i=0; i<nr; i++) {
                    for (int j=0; j<nc; j++) {
                        double w = layer->getWeights()->getElements()[i][j];
                        layer->getWeights()->getElements()[i][j] = w - learningRate * w;
                    }
                }
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








