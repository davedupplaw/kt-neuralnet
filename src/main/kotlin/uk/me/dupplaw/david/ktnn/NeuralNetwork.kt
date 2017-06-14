package uk.me.dupplaw.david.ktnn

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
        networkLayers.forEach {
            it.process(inputValues)
            inputValues = it.layerOutput!!
        }
        return inputValues
    }

    fun backPropagate(expected: Matrix<Double>) {
        (networkLayers.last() as HiddenNetworkLayer).apply {
            outputError = costFunction.derivative(expected, layerOutput!!) ʘ weightedInput!!.map { activationFunction.derivative(it) }
        }

        for (i in networkLayers.size - 2 downTo 1) {
            val currentLayer = networkLayers.get(i) as HiddenNetworkLayer
            val nextLayer = networkLayers.get(i + 1) as HiddenNetworkLayer

            currentLayer.outputError = (nextLayer.weightMatrix!!.T * nextLayer.outputError!!) ʘ
                    currentLayer.weightedInput!!.map { currentLayer.activationFunction.derivative(it) }
        }
    }

    override fun toString(): String {
        return "NeuralNetwork(networkLayers=${networkLayers.mapIndexed { idx, it -> "\n\t $idx - "+it.toString()}})"
    }
}

interface NetworkLayer {
    val numberOfNeurons: Int
    var previousLayer: NetworkLayer?
    var layerOutput: Matrix<Double>?

    fun process( input: Matrix<Double>)
}

class HiddenNetworkLayer(override val numberOfNeurons: Int,
                         val activationFunction: ActivationFunction,
                         val costFunction: CostFunction) : NetworkLayer {
    var weightMatrix: Matrix<Double>? = null
    var biasVector: Matrix<Double>? = null
    var weightedInput: Matrix<Double>? = null
    var outputError: Matrix<Double>? = null

    override var layerOutput: Matrix<Double>? = null
    override var previousLayer : NetworkLayer? = null
        set(value) {
            weightMatrix = MTJMatrixFactory().zeros(numberOfNeurons, value!!.numberOfNeurons).fill( { _, _ -> Math.random() } )
            biasVector = MTJMatrixFactory().zeros(numberOfNeurons, 1).fill( { _, _ -> Math.random() } )
            field = value
        }

    override fun process(input: Matrix<Double>) {
        val z = weightMatrix!! * input + biasVector!!
        weightedInput = z
        layerOutput = z.map { activationFunction.apply(it) }
    }

    override fun toString(): String {
        return "HiddenNetworkLayer(numberOfNeurons=$numberOfNeurons, " +
                "activationFunction=${activationFunction::class.java.simpleName}, " +
                "costFunction=${costFunction::class.java.simpleName}, " +
                "weightMatrix=${weightMatrix?.dims()}, " +
                "biasVector=${biasVector?.dims()}, " +
                "weightedInput=${weightedInput?.dims()}, " +
                "outputError=${outputError?.dims()}, " +
                "layerOutput=${layerOutput?.dims()})"
    }


}

fun Matrix<Double>.dims() : String = "${this?.numRows()}x${this?.numCols()}"
infix fun Matrix<Double>.ʘ(that: Matrix<Double>): Matrix<Double> {
    if( this.numRows() != that.numRows() ) {
        throw IndexOutOfBoundsException("A.numRows != B.numRows (${this.numRows()} != ${that.numRows()}")
    }
    if( this.numCols() != that.numCols() ) {
        throw IndexOutOfBoundsException("A.numCols != B.numCols (${this.numCols()} != ${that.numCols()}")
    }

    return this.mapIndexed { row, col, ele -> that[row, col] * ele }
}

class InputNetworkLayer(override val numberOfNeurons: Int) : NetworkLayer {
    override var layerOutput: Matrix<Double>? = null
    override var previousLayer: NetworkLayer? = null

    override fun process(input: Matrix<Double>) {
        this.layerOutput = input
    }

    override fun toString(): String {
        return "InputNetworkLayer(numberOfNeurons=$numberOfNeurons, layerOutput=${layerOutput?.dims()})"
    }
}