package neurons

class Perceptron(bias: Double = 3.0, weights: List<Double> = emptyList() ) :
        LogisticNeuron(bias, weights, stepFunction)
