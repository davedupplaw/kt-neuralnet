package neurons

class InputNeuron(value: Double)
    : LogisticNeuron(0.0, listOf(value), listOf(1.0), {v -> v} )