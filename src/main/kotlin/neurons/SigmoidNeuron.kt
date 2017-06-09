package neurons

class SigmoidNeuron(bias: Double = 3.0, weights: List<Double> = emptyList()) :
        LogisticNeuron(bias, weights, sigmoidFunction)