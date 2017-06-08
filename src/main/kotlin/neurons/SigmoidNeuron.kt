package neurons

class SigmoidNeuron(bias: Double = 3.0, inputs: List<WeightedInput> = emptyList()) :
        LogisticNeuron(bias, inputs, sigmoidFunction)