package uk.me.dupplaw.david.ktnn

import koma.matrix.Matrix
import koma.matrix.mtj.MTJMatrixFactory


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
        set(previous) {
            weightMatrix = MTJMatrixFactory().zeros(numberOfNeurons, previous!!.numberOfNeurons).fill( { _, _ -> randomGauss() } )
            biasVector = MTJMatrixFactory().zeros(numberOfNeurons, 1).fill( { _, _ -> randomGauss() } )
            field = previous
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


fun randomGauss() : Double {
    var w  = 0.0
    var x1 = 0.0
    do {
        x1 = 2.0 * Math.random() - 1.0
        w = x1 * x1
    } while (w >= 1.0)

    w = Math.sqrt(-2.0 * Math.log(w) / w)
    return x1 * w
}