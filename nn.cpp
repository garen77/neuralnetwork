#include "nn.h"

using namespace neuralnetworks;

bool isLogActive = true;

double relu(double inp) {
    return inp > 0 ? inp : 0;
}

double linear(double inp) {
    return inp;
}

double sigmoid(double inp) {
    return 1 / (1 + std::exp(-inp));
}

double heaviside(double inp) {
    if (inp < 0) {
        return 0;
    }
    else {
        return 1;
    }
}

Neuron::Neuron(int n) :numInputs(n) {
    this->w = new vector<double>();
    this->w->reserve(numInputs + 1);

    this->activation = &heaviside;

    for (int j = 0; j < numInputs; j++) {
        this->w->push_back((((double)rand()) / (double)RAND_MAX) * (1 + 1) - 1);
    }
    this->w->push_back((((double)rand()) / (double)RAND_MAX) * (1 + 1) - 1);
}

Neuron::Neuron(int n, double(*activ)(double)) :numInputs(n) {
    this->w = new vector<double>();
    this->w->reserve(numInputs + 1);

    this->activation = activ;

    for (int j = 0; j < numInputs; j++) {
        this->w->push_back((((double)rand()) / (double)RAND_MAX) * (1 + 1) - 1);
    }
    this->w->push_back((((double)rand()) / (double)RAND_MAX) * (1 + 1) - 1);
}

int Neuron::getNumInputs() {
    return this->numInputs;
}

vector<double>* Neuron::getWeights() {
    return this->w;
}

double Neuron::getDelta() {
    return this->delta;
}

void Neuron::setDelta(double delta) {
    this->delta = delta;
}

double Neuron::getOutput() {
    return this->output;
}

void Neuron::print() {
    std::cout << "w=[";
    int n = this->numInputs;
    for (int i = 0; i < n; i++) {
        std::cout << this->w->at(i);
        if (i != n - 1) {
            std::cout << ",";
        }
    }
    std::cout << "] b=" << this->w->at(n) << "\n";
}

double Neuron::activate(vector<double>* x) {

    vector<double>* weights = this->getWeights();
    double sum = 0.0;
    int inputSize = x->size();
    for (int i = 0; i < inputSize; i++) {
        sum += weights->at(i) * x->at(i);
    }
    sum += weights->at(inputSize); // bias sum
    this->output = sum;
    return this->activation(sum);

}

NeuralNetwork::NeuralNetwork(int* conf, int nl) :configurazione(conf), numOfLayers(nl) {

    // nl : num layers
    // [niumInput, numHiddenNeuron,..,numHiddenNeuron, numOutNeuron] -> [nl + 1]

    this->_layers = new vector<vector<Neuron*>*>();
    this->_layers->reserve(nl);
    for (int l = 1; l < nl + 1; l++) {
        int numInputs = this->configurazione[l - 1];
        int numOutputs = this->configurazione[l];

        vector<Neuron*>* layer = new vector<Neuron*>();
        layer->reserve(numOutputs);
        for (int i = 0; i < numOutputs; i++) {
            Neuron* neuron = new Neuron(numInputs + 1);
            layer->push_back(neuron);
        }

        this->_layers->push_back(layer);

    }
}

double NeuralNetwork::activate(vector<double>* weights, vector<double>* inputs) {
    double sum = 0.0;
    int inputSize = inputs->size();
    for (int i = 0; i < inputSize; i++) {
        sum += weights->at(i) * inputs->at(i);
    }
    sum += weights->at(inputSize); // bias sum
    return sigmoid(sum);
}

vector<double>* NeuralNetwork::forwardPropagate(vector<double>* inputs) {
    if (isLogActive) {
        cout << "\nForwardpropagate\n";
    }
    vector<double>* currInputs = new vector<double>(*inputs);
    for (int l = 0; l < this->numOfLayers; l++) {
        if (isLogActive) {
            cout << "\nlayer " << l;
        }

        vector<Neuron*>* layer = this->_layers->at(l);
        int layerSize = layer->size();
        vector<double>* newInputs = new vector<double>();
        newInputs->reserve(layerSize);
        for (int n = 0; n < layerSize; n++) {
            Neuron* neuron = layer->at(n);
            double neuronOut = neuron->activate(currInputs);
            double* pNeuronOut = new double[1];
            pNeuronOut[0] = neuronOut;
            newInputs->push_back(neuronOut);
            if (isLogActive) {
                cout << "\nneuronOut=" << neuronOut << " from map "<<endl;
            }
        }
        delete currInputs;
        currInputs = newInputs;
    }
    return currInputs;
}

void NeuralNetwork::backPropagate(vector<double>* expected) {
    if (isLogActive) {
        cout << "\nBackpropagate\n";
    }
    for (int i = this->numOfLayers - 1; i >= 0; i--) {
        if (isLogActive) {
            cout << "\nlayer " << i;
        }
        vector<Neuron*>* layer = this->_layers->at(i);
        int layerSize = layer->size();
        vector<double>* errors = new vector<double>();
        errors->reserve(layerSize);
        if (i != this->numOfLayers - 1) {
            for (int j = 0; j < layerSize; j++) {
                double error = 0.0;
                vector<Neuron*>* nextLayer = this->_layers->at(i + 1);
                int nextLayerSize = nextLayer->size();
                for (int n = 0; n < nextLayerSize; n++) {
                    Neuron* neuron = nextLayer->at(n);
                    vector<double>* w = neuron->getWeights();
                    double neuronDelta = neuron->getDelta();
                    error += w->at(j) * neuronDelta;
                }
                errors->push_back(error);
            }
        }
        else {
            for (int j = 0; j < layerSize; j++) {
                Neuron* neuron = layer->at(j);
                errors->push_back(expected->at(j) - neuron->getOutput());
                if (isLogActive) {
                    cout << "\nexpected[" << j << "]=" << expected->at(j) << " out from map=" << neuron->getOutput()<<endl;
                }
            }
        }
        for (int j = 0; j < layerSize; j++) {
            Neuron* neuron = layer->at(j);
            double neuronOut = neuron->getOutput();
            double neuronDelta = errors->at(j) * neuronOut * (1 - neuronOut);
            double* pNeuronDelta = new double[1];
            neuron->setDelta(neuronDelta);
            if (isLogActive) {
                cout << "\nneuronDelta =" << neuronDelta<<endl;
            }
        }
    }
}

void NeuralNetwork::updateWeights(vector<double>* inputs, double lr) {
    if (isLogActive) {
        cout << "\n-update weights\n";
    }
    vector<double>* currInputs = new vector<double>(*inputs);
    for (int i = 0; i < this->numOfLayers; i++) {
        vector<Neuron*>* layer = this->_layers->at(i);
        int layerSize = layer->size();
        if (i != 0) {
            vector<Neuron*>* previousLayer = this->_layers->at(i - 1);
            int previuosLayerSize = previousLayer->size();
            currInputs = new vector<double>();
            currInputs->reserve(previuosLayerSize);
            for (int j = 0; j < previuosLayerSize; j++) {
                Neuron* neuron = previousLayer->at(j);
                double neuronOut = neuron->getOutput();
                currInputs->push_back(neuronOut);
            }
        }
        for (int n = 0; n < layerSize; n++) {
            Neuron* neuron = layer->at(n);
            int currInputsSize = currInputs->size();
            vector<double>* w = neuron->getWeights();
            double neuronDelta = neuron->getDelta();
            if (isLogActive) {
                cout << "\nLayer " << n;
            }
            for (int j = 0; j < currInputsSize; j++) {
                if (isLogActive) {
                    cout << "\n w[" << j << "]=" << w->at(j) << " pre update";
                }
                double wUpdated = w->at(j) + lr * currInputs->at(j) * neuronDelta;
                if (isLogActive) {
                    cout << "\n currInp[" << j << "]=" << currInputs->at(j) << " delta=" << neuronDelta;
                }
                w->at(j) += lr * currInputs->at(j) * neuronDelta;
                if (isLogActive) {
                    cout << "\n w[" << j << "]=" << w->at(j) << " post update\n";
                }
            }
            w->at(currInputsSize) += lr * neuronDelta;
        }

    }
}

void NeuralNetwork::trainNetwork(vector<vector<double>*>* trainingSet, double lr, int numEpochs, int numOutputs) {
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        double sumError = 0.0;
        int trainintSetSize = trainingSet->size();
        for (int i = 0; i < trainintSetSize; i++) {
            vector<double>* row = trainingSet->at(i);
            int inputsSize = row->size() - 1;
            vector<double>* inputs = new vector<double>();
            inputs->reserve(inputsSize);

            for (int j = 0; j < inputsSize; j++) {
                inputs->push_back(row->at(j));
            }
            vector<double>* outputs = this->forwardPropagate(inputs);
            vector<double>* expected = new vector<double>();
            expected->reserve(numOutputs);
            for (int j = 0; j < numOutputs; j++) {
                expected->push_back(0.0);
            }
            expected->at(row->at(inputsSize)) = 1;

            for (int k = 0; k < numOutputs; k++) {
                sumError += pow((expected->at(k) - outputs->at(k)), 2);
            }
            this->backPropagate(expected);
            this->updateWeights(inputs, lr);
        }
        if (isLogActive) {
            cout << "\nepoch = " << epoch << ", learning rate = " << lr << ", error = " << sumError << "\n";
        }
    }
}

int NeuralNetwork::fit(vector<double>* inputs) {
    vector<double>* outputs = this->forwardPropagate(inputs);
    vector<Neuron*>* outLayer = this->_layers->at(this->_layers->size() - 1);
    int numOutputs = outLayer->size();
    Neuron* neuron = outLayer->at(0);
    double maxOut = neuron->getOutput();
    for (int n = 1; n < numOutputs; n++) {
        Neuron* neuron = outLayer->at(n);
        double neuronOut = neuron->getOutput();
        if (neuronOut > maxOut) {
            maxOut = neuronOut;
        }
    }
    vector<double>::iterator it = find(outputs->begin(), outputs->end(), maxOut);
    if (it != outputs->end()) {
        return it - outputs->begin();
    }
    else {
        return -1;
    }
}
