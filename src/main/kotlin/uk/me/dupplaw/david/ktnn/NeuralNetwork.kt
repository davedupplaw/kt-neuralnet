package uk.me.dupplaw.david.ktnn

import koma.mat
import koma.matrix.Matrix
import koma.matrix.mtj.MTJMatrix
import koma.matrix.mtj.MTJMatrixFactory
class NeuralNetwork(val networkLayers: List<NetworkLayer>) {
    var trainedErrors: List<Matrix<Double>> = mutableListOf()
    var weightedInputs: List<Matrix<Double>> = mutableListOf()
    var trainedOutputs: List<Matrix<Double>> = mutableListOf()
    var learningRate: Double = 3.0

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

    fun training(trainingInputs: Matrix<Double>, trainingLabels: Matrix<Double>) {
        val allLayersButTheInputLayer = getAllLayersButTheInputLayer()

        val nTrainingExamples = trainingInputs.numCols()

        val trainedErrorList  = createMatrixForEachNodeBy(nTrainingExamples)
        val trainedOutputList = createMatrixForEachNodeBy(nTrainingExamples)
        val weightedInputList = createMatrixForEachNodeBy(nTrainingExamples)

        (0 until nTrainingExamples).map { trainingExampleIndex ->
            val givenInput = trainingInputs.selectCols(trainingExampleIndex)
            val desiredOutput = trainingLabels.selectCols(trainingExampleIndex)

            feedforward(givenInput)

            if (trainingExampleIndex==0) {
                println("++++++++++++++++++++++++++++++++")
                println(networkLayers[networkLayers.lastIndex].layerOutput!!)
            }

            allLayersButTheInputLayer.forEachIndexed { index, networkLayer ->
                if( networkLayer is HiddenNetworkLayer ) {
                    trainedOutputList[index].setCol(trainingExampleIndex, networkLayer.layerOutput!!)
                    weightedInputList[index].setCol(trainingExampleIndex, networkLayer.weightedInput!!)
                }
            }
            backPropagate(desiredOutput)
            allLayersButTheInputLayer.forEachIndexed { index, networkLayer ->
                if( networkLayer is HiddenNetworkLayer ) {
                    trainedErrorList[index].setCol(trainingExampleIndex, networkLayer.outputError!!)
                }
            }
        }

        trainedOutputs = trainedOutputList
        weightedInputs = weightedInputList
        trainedErrors = trainedErrorList

        (networkLayers.size-1 downTo 1).map { index ->
            val learningRateOverNumberOfInputs = (learningRate / trainingInputs.numCols())
            val networkLayer = networkLayers[index] as HiddenNetworkLayer
            val previousLayerOutput = if (index == 1) trainingInputs.T else trainedOutputs[index-2].T
            networkLayer.weightMatrix = networkLayer.weightMatrix!!.minus(
                    trainedErrors[index-1] * previousLayerOutput * learningRateOverNumberOfInputs)
            networkLayer.biasVector = networkLayer.biasVector!!.minus(
                    trainedErrors[index-1].mapRows { mat[it.elementSum()] } * learningRateOverNumberOfInputs)
        }
    }

    private fun createMatrixForEachNodeBy(nTrainingExamples: Int): MutableList<MTJMatrix> {
        return getAllLayersButTheInputLayer().map { networkLayer ->
            MTJMatrixFactory().zeros(networkLayer.numberOfNeurons, nTrainingExamples)
        }.toMutableList()
    }

    private fun getAllLayersButTheInputLayer() = networkLayers.filterIndexed { i, _ -> i != 0 }

    fun backPropagate(desiredOutput: Matrix<Double>) {
        calculateTrainingError(desiredOutput)
        updateErrors()
        updateWeights()
    }

    private fun updateWeights() {
    }

    private fun calculateTrainingError(desiredOutput: Matrix<Double>) {
        (networkLayers.last() as HiddenNetworkLayer).apply {
            outputError = costFunction.derivative(desiredOutput, layerOutput!!) ʘ
                    weightedInput!!.map { activationFunction.derivative(it) }
        }
    }

    private fun updateErrors() {
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
