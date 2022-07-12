// Neural Net implementation in C++ using ANSI ISO C++ Standards that can run on any standard compiler

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;


// Class to read training data from a text file 
class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}


// Structure for Neuron with the weight and the change in weight of each neuron 
// to all of the other neurons that it feeds as members.
struct Connection
{
    double weight;
    double deltaWeight; 
};

class Neuron;

typedef vector<Neuron> Layer;

// Class Neuron
class Neuron
{
public:
    //Constructor 
    Neuron(unsigned numOutputs, unsigned myIndex);

    // Takes a double and sets it to a variable called m_outputVal. 
    void setOutputVal(float val) {m_outputVal = val;} 
    // Returns stored variable called m_outputVal.
    // The object is not modified and so the entire function is wrapped with a const.
    double getOutputVal(void) const {return m_outputVal;}
    
    // Function to calculate ouput values using the previous layer.
    // The previous layer is not modified and so const is used in teh argument.
    void feedForward(const Layer &prevLayer);
     
    void calcOutputGradients(double targetVal); 

    void calcHiddenGradients(const Layer &nextLayer);

    void updateInputWeights(Layer &prevLayer);


private:
    
    static double transferFunction(float x);
    static double transferFunctionDerivative(float x);

    // Return a random function between 0 and 1
    // Must include cstdlib standard header
    static double randomWeight(void) {return rand()/double(RAND_MAX); }

    double sumDOW(const Layer &nextLayer) const; 
    
    // The neuron's output values
    float m_outputVal;

    // The weight and change in weight of each neuron to all of the other neurons that it feeds.
    // Need to store 2 doubles and so a struct called connection was created instead of a vector.
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;

    // Overall net training rate
    // [0.0 .. 1.0]
    static double eta; 

    // Multiplier of last weight change, momentum
    // [0.0 .. n]
    static double alpha; 

    // Hyperparameter that controls the value to which an ISRLU saturates for negative inputs
    static double ISRLU_alpha;

};

double Neuron::eta = 0.15; 

double Neuron::alpha = 0.5; 

double Neuron::ISRLU_alpha = 3;

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}


double Neuron::sumDOW(const Layer &nextLayer) const{

    double sum = 0.0;

    // Sum all the error contributions that a neuron makes to the nodes that they feed

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}


void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);

}


void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);

}


double Neuron::transferFunction(float x)
{
    float y;
    if (x < 0) {
        long i;
        float x1, x2;
        const float threehalfs = 1.5F;
        x1 = 1 + ISRLU_alpha * x * x;
        x2 = x1 * 0.5F;
        y  = x1;
        i  = * ( long * ) &y;    // evil floating point bit level hacking
        i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
        y  = * ( float * ) &i;
        y  = x * y * ( threehalfs - ( x2 * y * y ) );   

    } else {
        y = x;
    }

    return y;

};

double Neuron::transferFunctionDerivative(float x)
{
    float y;
    if (x < 0) {
        long i;
        float x2;
        const float threehalfs = 1.5F;
        x = 1 + ISRLU_alpha * x * x;
        x2 = x * 0.5F;
        y  = x;
        i  = * ( long * ) &y;    // evil floating point bit level hacking
        i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
        y  = * ( float * ) &i;
        y  =  pow(y * ( threehalfs - ( x2 * y * y ) ),3);   

    } else {
        y = 1;
    }

    return y;
};

void Neuron::feedForward(const Layer &prevLayer)
{
    float sum = 0.0;
    //Sum the previous layer's outputs (which are our inputs)
    //Include the bias node from the previous layer
    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        // Sum of product of all previous outputs and each weight.
        // Need to specify the current neuron's index so the layer is knows how to connect the weights. 
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;

    }

    m_outputVal = Neuron::transferFunction(sum);

};

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    // Loop through all the connections connected to a single neuron in the network
    for (unsigned c = 0; c < numOutputs; ++c){
        // Append a connection structure to the outputWeights container
        m_outputWeights.push_back(Connection());

        // Set the weight for that connection to a random value
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
};

class Net
{
public:
    // Topology is a non-changing vector of doubles - add const to the argument. 
    Net(const vector<unsigned> &topology); 

    // Function used to carryout feedforward. 
    // Taking inputVals as reference of doubles instead of the entire array.
    // Const since the argument, inputVals, will not be altered. 
    void feedForward(const vector<double> &inputVals); 
   
    // Function used to carryout back propogation. 
    // Taking targetVals as reference of doubles instead of the entire array.
    // Const since the argument, targetVals, will not be altered. 
    void backProp(const vector<double> &targetVals);

    // Function used to get results. 
    // Taking resultVals as reference of doubles instead of the entire array.
    // Const of the argument does not apply since the  vector of doubles will be changed and returned as a callback.
    // Const of the entire function since the entire object remains the same size.  
    void getResults(vector<double> &resultVals) const;

    double getRecentAverageError(void) const { return m_recentAverageError; }


private:

    // All the neurons are defined as 2D objects, layers and neurons.
    // m_layers[layerNum][neuronNum].
    vector<Layer> m_layers; 

    // Error item f
    double m_error; 

    // Running average variables
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;

};

// Number of training samples to average over
double Net::m_recentAverageSmoothingFactor = 100.0; 

void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}


void Net::backProp(const vector<double> &targetVals) {
    // Calculate overall net error, RMS of output neuron errors

    // Reference to the last layer/output layer
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n=0; n<outputLayer.size() -1; ++n)
    {
        // TOTAL(Target - Actual)^2
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;     
    };

    m_error /= outputLayer.size() - 1 ; // get average error squared
    m_error = sqrt(m_error); // RMS

    // Implement a recent average measurement to get a running average that reflects the network's performance
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) 
    / (m_recentAverageSmoothingFactor + 1.0);


    // Calculate output layer gradients
    for (unsigned n=0; n < outputLayer.size() - 1; ++n){
        outputLayer[n].calcOutputGradients(targetVals[n]);

    }


    // Calculate gradients on hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum){
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }

    }

    // For all layers from outputs to first hidden layer, update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n){
            layer[n].updateInputWeights(prevLayer);             
        }
    }


};



void Net::feedForward(const vector<double> &inputVals)
{
    // Error handling to ensure that the number of elements 
    // in inputVals is the same as the number of input neurons we have. 
    // Must include cassert standard header 
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Assigned (latch) the input values into the input neurons
    for (unsigned i=0; i<inputVals.size(); ++i)
    {
        // Output of each neuron is a private member of Neuron
        // setOutputVal gives class Net a way to set the output value
        m_layers[0][i].setOutputVal(inputVals[i]);

        // Forward propagate, skip the input layer by starting with 1
        for (unsigned layerNum =1; layerNum < m_layers.size(); ++layerNum)
        {
            // Reference to the previous layer using a pointer
            Layer &prevLayer = m_layers[layerNum - 1];
            for (unsigned n=0; n< m_layers[layerNum].size() - 1; ++n){
                m_layers[layerNum][n].feedForward(prevLayer);
            }
        }
    };

};


Net::Net(const vector<unsigned> &topology) 
{
    // Determine the number of layers in the neural net.
    unsigned numLayers = topology.size();

    // Creating the neural net by looping through the number of layers and the number of neurons per layer + 1  (bias).
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        // push_back is the standard container member function used to append a new element onto a standard container
        // Create a new empty layer and append it to the m_layer container
        m_layers.push_back(Layer());

        // Determine the number of outputs for a given neuron by calculating the number of neurons in the next layer.  
        // A conditional operator is used to determine whether the current layer is the output layer 
        // syntax: condition ? result_if_true : result_if_false
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        
        //We have created new layer, now fill in all of the neurons and add a bias neuron to the layer
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            // back() is the standard container member function that returns the last member in the container
            // Append the nearly constructed neuron to the m_layer container
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
        }


        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        m_layers.back().back().setOutputVal(1.0);
    }

};

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}


int main()
{
    TrainingData trainData("XORtrainingData.txt");

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
                << myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;
};

