
import koma.matrix.Matrix
import koma.matrix.mtj.MTJMatrixFactory

class NeuralNetwork(val networkLayers: List<NetworkLayer>) {
    init {
        networkLayers.forEachIndexed { index, networkLayer ->
            if( index > 0 ) {
                networkLayer.previousLayer = networkLayers[index-1]
            }
        }
    }

    fun feedforward(givenInput: Matrix<Double>) : Matrix<Double> {
        var inputValues = givenInput
        networkLayers.forEach { inputValues = it.process(inputValues) }
        return inputValues
    }
}

interface NetworkLayer {
    val numberOfNeurons: Int
    var previousLayer: NetworkLayer?

    fun process( input: Matrix<Double> ) : Matrix<Double>
}

class HiddenNetworkLayer(override val numberOfNeurons: Int, val activationFunction: (Double) -> Double) : NetworkLayer {
    override var previousLayer : NetworkLayer? = null
        set(value) {
            weightMatrix = MTJMatrixFactory().zeros(numberOfNeurons, value!!.numberOfNeurons).fill( { _,_ -> Math.random() } )
            biasVector = MTJMatrixFactory().zeros(numberOfNeurons, 1).fill( { _,_ -> Math.random() } )
            field = value
        }

    var weightMatrix: Matrix<Double>? = null
    var biasVector: Matrix<Double>? = null
    var weightedInput: Matrix<Double>? = null

    override fun process(input: Matrix<Double>): Matrix<Double> {
        val z = weightMatrix!! * input + biasVector!!
        weightedInput = z
        return z.map( activationFunction )
    }
}

class InputNetworkLayer(override val numberOfNeurons: Int) : NetworkLayer {
    override var previousLayer: NetworkLayer? = null
    override fun process(input: Matrix<Double>) = input
}

val stepFunction =  { biasedSum: Double -> if (biasedSum > 0) 1.0 else 0.0 }
val sigmoidFunction =  { biasedSum: Double -> 1.0 / (1.0 + Math.exp(-biasedSum)) }