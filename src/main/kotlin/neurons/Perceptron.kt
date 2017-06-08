package neurons

class Perceptron(bias : Double = 3.0, inputs: List<WeightedInput> = emptyList() ) :
        LogisticNeuron(bias, inputs, stepFunction)
