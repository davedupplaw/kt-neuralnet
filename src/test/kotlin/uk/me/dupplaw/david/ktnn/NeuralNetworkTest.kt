package uk.me.dupplaw.david.ktnn
import com.nhaarman.mockito_kotlin.doReturn
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import koma.end
import koma.mat
import koma.util.test.assertMatrixEquals
import org.assertj.core.api.KotlinAssertions.assertThat
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.junit.platform.runner.JUnitPlatform
import org.junit.runner.RunWith
import org.mockito.Mockito.verify

@RunWith(JUnitPlatform::class)
internal class NeuralNetworkTest: Spek({
    describe("A neural network having an input and output layer") {
        val finalOutput = mat[0.2, 0.3].T
        val givenInput = mat[2.0, 4.0].T

        on("being constructed") {
            val inputLayer = InputNetworkLayer(2)
            val outputLayer = HiddenNetworkLayer(3, StepFunction(), MeanSquaredError())
            NeuralNetwork(listOf(inputLayer, outputLayer))

            it("should set the output layer previous layer to the input layer" ) {
                assertThat(outputLayer.previousLayer).isEqualTo( inputLayer )
            }
        }

        on("feeding forward the input") {
            val inputLayerOutput = mat[7.0,8.0].T

            val inputLayer = mock<InputNetworkLayer> {}
            whenever(inputLayer.layerOutput).thenReturn( inputLayerOutput )

            val outputLayer = mock<HiddenNetworkLayer> {}
            whenever(outputLayer.layerOutput).thenReturn( finalOutput )

            val unit = NeuralNetwork(listOf(inputLayer, outputLayer))

            val itsTheFinalOutputDoDoDoDooo = unit.feedforward(givenInput)

            it("should call the output layer to process the result of the input layer") {
                verify(outputLayer).process(inputLayerOutput)
            }

            it("should return the output from the last layer") {
                assertThat(itsTheFinalOutputDoDoDoDooo).isEqualTo(finalOutput)
            }
        }

        on("propagating backwards") {
            val expected = mat[10, 20].T
            val mockActivationFunction = mock<ActivationFunction> {
                on{ derivative(12.0) } doReturn 8.0
                on{ derivative(78.0) } doReturn 6.0
                on{ derivative(20.0) } doReturn 0.25
                on{ derivative(10.0) } doReturn 0.25
            }

            val outputLayer = HiddenNetworkLayer(2, mockActivationFunction, MeanSquaredError())
            val hiddenLayer = HiddenNetworkLayer(2, mockActivationFunction, MeanSquaredError())
            val unit = NeuralNetwork(listOf(InputNetworkLayer(2), hiddenLayer, outputLayer))

            outputLayer.layerOutput = mat[12,22].T
            outputLayer.weightedInput = mat[12,78].T
            outputLayer.weightMatrix = mat[2,0 end 0,2]

            hiddenLayer.layerOutput = mat[60,40].T
            hiddenLayer.weightedInput = mat[20,10].T

            unit.backPropagate(expected)

            it("should calculate the output error of the final layer") {
                assertMatrixEquals(mat[16.0,12.0].T, outputLayer.outputError!!)
            }

            it("should calculate the error of the penultimate layer") {
                assertMatrixEquals(mat[8.0,6.0].T, hiddenLayer.outputError!!)
            }
        }

        on("training") {
            val trainingInputs = mat[1,2 end -3,-4 end 5,6].T
            val inputLayer = InputNetworkLayer(2)
            val hiddenLayer = HiddenNetworkLayer(1,StepFunction(),MeanSquaredError())
            var outputLayer = HiddenNetworkLayer(2,StepFunction(),MeanSquaredError())

            val unit = NeuralNetwork(listOf(inputLayer, hiddenLayer, outputLayer))

            hiddenLayer.biasVector = mat[3]
            hiddenLayer.weightMatrix = mat[1.0,1.0]
            outputLayer.biasVector = mat[0, 0].T
            outputLayer.weightMatrix = mat[3.0, 4.0].T

            unit.training(trainingInputs)

            it("should store all layer outputs per training example") {
                val expectedOutputFromHiddenLayer = mat[1,0,1]
                val expectedOutputFromOutputLayer = mat[1,1 end 0,0 end 1,1].T

                assertThat( unit.trainedOutputs ).isNotNull().hasSize(2)
                assertMatrixEquals( expectedOutputFromHiddenLayer, unit.trainedOutputs.get(0) )
                assertMatrixEquals( expectedOutputFromOutputLayer, unit.trainedOutputs.get(1) )
            }

            it("should store all the layer weighted inputs per training example") {
                val expectedWeightedInputToHiddenLayer = mat[6,-4,14]
                val expectedWeightedInputToOutputLayer = mat[3,4 end 0,0 end 3,4].T

                assertThat( unit.weightedInputs ).isNotNull().hasSize(2)
                assertMatrixEquals(expectedWeightedInputToHiddenLayer, unit.weightedInputs.get(0))
                assertMatrixEquals(expectedWeightedInputToOutputLayer, unit.weightedInputs.get(1))
            }
        }

        on("training with a Sigmoid") {
            val trainingInputs = mat[1,2 end -3,-4 end 5,6].T
            val inputLayer = InputNetworkLayer(2)
            val hiddenLayer = HiddenNetworkLayer(1,SigmoidFunction(),MeanSquaredError())
            var outputLayer = HiddenNetworkLayer(2,SigmoidFunction(),MeanSquaredError())

            val unit = NeuralNetwork(listOf(inputLayer, hiddenLayer, outputLayer))

            hiddenLayer.biasVector = mat[3]
            hiddenLayer.weightMatrix = mat[1.0,1.0]
            outputLayer.biasVector = mat[0, 0].T
            outputLayer.weightMatrix = mat[3.0, 4.0].T

            unit.training(trainingInputs)

            val desiredOutputLayerOutputs = mat[ 1, 1 end 0.5, 0.5 end 0.95257401412533, 0.98201373128967 ].T
            val expectedOutputLayerErrors  = (unit.trainedOutputs[unit.trainedOutputs.lastIndex] - desiredOutputLayerOutputs) ʘ
                    unit.weightedInputs[unit.weightedInputs.lastIndex].map { SigmoidFunction().derivative(it) }

            it("should store all the neuron errors per training example per layer") {
                assertThat( unit.trainedErrors ).isNotNull().hasSize(2)
//                assertMatrixEquals( expectedOutputFromHiddenLayer, unit.trainedErrors.get(0) )
                assertMatrixEquals( expectedOutputLayerErrors, unit.trainedErrors.get(1) )
            }
        }

        on("updating weights") {
            it("should work") {

            }
        }
    }
})