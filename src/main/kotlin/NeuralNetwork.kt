
import koma.matrix.Matrix
import koma.matrix.mtj.MTJMatrixFactory
import neurons.LogisticNeuron

class NeuralNetwork(val networkLayers: List<NetworkLayer>) {
    init {
        networkLayers.forEachIndexed { index, networkLayer ->
            if( index > 0 ) {
                networkLayer.previousLayer = networkLayers[index-1]
            }
        }
    }

    fun process(givenInput: List<Double>) : List<Double> {
        var inputValues = givenInput
        networkLayers.forEach { inputValues = it.process(inputValues) }
        return inputValues
    }
}

interface NetworkLayer {
    val numberOfNeurons: Int
    var previousLayer: NetworkLayer?

    fun process( input: List<Double> ) : List<Double>
}

class HiddenNetworkLayer(override val numberOfNeurons: Int, creator: () -> LogisticNeuron ) : NetworkLayer {
    override var previousLayer : NetworkLayer? = null
        set(value) {
            weightMatrix = MTJMatrixFactory().zeros(numberOfNeurons, value!!.numberOfNeurons)
            biasVector = MTJMatrixFactory().zeros(numberOfNeurons, 1)
            field = value
        }

    var weightMatrix: Matrix<Double>? = null
    var biasVector: Matrix<Double>? = null
    val neuronList : MutableList<LogisticNeuron> = mutableListOf()

    init {
        for( i in 1..numberOfNeurons ) {
            neuronList.add(creator())
        }
    }

    override fun process(input: List<Double>) = neuronList.map { it.fire(input) }.toList()
}

class InputNetworkLayer(override val numberOfNeurons: Int) : NetworkLayer {
    override var previousLayer: NetworkLayer? = null
    override fun process(input: List<Double>) = input
}

inline fun <reified T : LogisticNeuron> createNeuron() : LogisticNeuron {
    return T::class.java.newInstance();
}
