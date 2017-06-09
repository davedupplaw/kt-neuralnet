package neurons

class InputNeuron(value: Double)
    : LogisticNeuron(0.0, listOf(WeightedInput(value,1.0)), {v -> v} )