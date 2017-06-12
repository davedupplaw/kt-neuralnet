
import neurons.Perceptron
import neurons.SigmoidNeuron
import org.assertj.core.api.KotlinAssertions.assertThat
import org.assertj.core.data.Offset
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.junit.platform.runner.JUnitPlatform
import org.junit.runner.RunWith

@RunWith(JUnitPlatform::class)
internal class HiddenNetworkLayerTest : Spek({
    describe("HiddenNetworkLayer") {

        on("construction") {
            val numberOfNeurons = 10
            val unit = HiddenNetworkLayer(numberOfNeurons, { createNeuron<Perceptron>() })

            it("should make a layer with a given number of specified Neurons") {
                assertThat(unit.neuronList).hasSize(numberOfNeurons)
                        .allSatisfy({ assertThat(it).isInstanceOf(Perceptron::class.java) })
            }
        }

        on("setting the previous layer") {
            val numberOfNeuronsInPreviousLayer = 2
            val inputNetworkLayer = InputNetworkLayer(numberOfNeuronsInPreviousLayer)

            val numberOfNeurons = 10
            val unit = HiddenNetworkLayer(numberOfNeurons, { createNeuron<Perceptron>() })
            unit.previousLayer = inputNetworkLayer

            it("should make a weightMatrix of with dimensions - number of neurons 'in this layer' x 'the previous layer' (10x2)") {
                assertThat(unit.weightMatrix?.numCols()).isEqualTo(numberOfNeuronsInPreviousLayer)
                assertThat(unit.weightMatrix?.numRows()).isEqualTo(numberOfNeurons)
            }

            it("should make a bias vector the size of the number of neurons" ) {
                assertThat(unit.biasVector?.numCols()).isEqualTo(1);
                assertThat(unit.biasVector?.numRows()).isEqualTo(numberOfNeurons);
            }
        }

        on("processing perceptrons") {
            val unit = HiddenNetworkLayer(2, { createNeuron<Perceptron>() })

            it("should calculate the output state") {
                assertThat(unit.process(listOf(2.0, 4.0)))
                        .hasSize(2)
                        .containsExactly(1.0, 1.0)
            }
        }

        on("processing sigmoid neurons") {
            val unit = HiddenNetworkLayer(2, { createNeuron<SigmoidNeuron>() })

            it("should calculate the output state") {
                assertThat(unit.process(listOf(-0.4, 0.2)))
                        .hasSize(2)
                        .allSatisfy { assertThat(it).isCloseTo(0.94, Offset.offset(0.01)) }
            }
        }
    }
})

