
import com.nhaarman.mockito_kotlin.doReturn
import com.nhaarman.mockito_kotlin.mock
import neurons.Perceptron
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
        val givenInput = listOf(2.0, 4.0)
        val finalOutput = mutableListOf(0.2, 0.3)

        val inputLayer = InputNetworkLayer(2)

        on("being constructed") {
            val outputLayer = HiddenNetworkLayer(3, { createNeuron<Perceptron>() })
            NeuralNetwork(listOf(inputLayer, outputLayer))

            it("should set the output layer previous layer to the input layer" ) {
                assertThat( outputLayer.previousLayer ).isEqualTo( inputLayer )
            }
        }

        on("being called to process an input") {
            val outputLayer = mock<HiddenNetworkLayer> {
                on { process(givenInput) } doReturn finalOutput
            }
            val unit = NeuralNetwork(listOf(inputLayer, outputLayer))

            val itsTheFinalOutputDoDoDoDooo = unit.process(givenInput)

            it("should call the first layer to process the input") {
                verify(outputLayer).process(givenInput)
            }

            it("should return the output from the last layer") {
                assertThat(itsTheFinalOutputDoDoDoDooo).isEqualTo(finalOutput)
            }
        }
    }

    describe("A neural network having two network layers") {
        val outputOfFirstLayer = mutableListOf( 0.2, 0.3 )
        val mockLayer1 = mock<HiddenNetworkLayer> {
            on { process( listOf(2.0, 4.0) ) } doReturn outputOfFirstLayer
        }
        val finalOutput = mutableListOf( 0.02, 0.03 )
        val mockLayer2 = mock<HiddenNetworkLayer>{
            on { process(outputOfFirstLayer) } doReturn finalOutput
        }

        val unit = NeuralNetwork( listOf( mockLayer1, mockLayer2 ) )

        on("being called to process an input") {
            val givenInput = listOf( 2.0, 4.0 )
            val itsTheFinalOutputDoDoDoDooo = unit.process( givenInput )

            it("should call the first layer to process the input") {
                verify( mockLayer1 ).process( givenInput )
            }

            it("should call the second layer with the output of the first") {
                verify( mockLayer2 ).process( outputOfFirstLayer )
            }

            it("should return the output from the last layer") {
                assertThat(itsTheFinalOutputDoDoDoDooo).isEqualTo(finalOutput)
            }
        }
    }
})