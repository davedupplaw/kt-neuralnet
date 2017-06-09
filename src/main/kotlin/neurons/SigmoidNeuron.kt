package neurons

class SigmoidNeuron(bias: Double = 3.0, inputs: List<Double> = emptyList(), weights: List<Double> = emptyList()) :
        LogisticNeuron(bias, inputs, weights, sigmoidFunction)