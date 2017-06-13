
import com.nhaarman.mockito_kotlin.doReturn
import com.nhaarman.mockito_kotlin.mock
import koma.mat
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
            val dummyActivationFunction = {_:Double -> 1.0 }
            val inputLayer = InputNetworkLayer(2)
            val outputLayer = HiddenNetworkLayer(3, dummyActivationFunction )
            NeuralNetwork(listOf(inputLayer, outputLayer))

            it("should set the output layer previous layer to the input layer" ) {
                assertThat( outputLayer.previousLayer ).isEqualTo( inputLayer )
            }
        }

        on("feeding forward the input") {
            val inputLayerOutput = mat[7.0,8.0].T
            val inputLayer = mock<InputNetworkLayer> {
                on { process(givenInput) } doReturn inputLayerOutput
            }
            val outputLayer = mock<HiddenNetworkLayer> {
                on { process(inputLayerOutput) } doReturn finalOutput
            }
            val unit = NeuralNetwork(listOf(inputLayer, outputLayer))

            val itsTheFinalOutputDoDoDoDooo = unit.feedforward(givenInput)

            it("should call the output layer to process the result of the input layer") {
                verify(outputLayer).process(inputLayerOutput)
            }

            it("should return the output from the last layer") {
                assertThat(itsTheFinalOutputDoDoDoDooo).isEqualTo(finalOutput)
            }
        }
    }
})