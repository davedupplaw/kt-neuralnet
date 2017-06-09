
import neurons.LogisticNeuron

class NeuralNetwork(val layers: List<NetworkLayer>) {
    fun process(givenInput: List<Double>) : List<Double> {
        var inputValues = givenInput
        layers.forEach { inputValues = it.process(inputValues) }
        return inputValues
    }
}

class NetworkLayer(numberOfNeurons: Int, creator: () -> LogisticNeuron ) {
    val neuronList : MutableList<LogisticNeuron> = mutableListOf()

    init {
        for( i in 1..numberOfNeurons ) {
            neuronList.add(creator())
        }
    }

    fun process(input: List<Double>) = neuronList.map { it.fire(input) }.toList()
}

inline fun <reified T : LogisticNeuron> createNeuron() : LogisticNeuron {
    return T::class.java.newInstance();
}
