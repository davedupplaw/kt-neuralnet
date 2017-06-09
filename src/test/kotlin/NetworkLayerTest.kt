
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
internal class NetworkLayerTest : Spek({
    describe("NetworkLayer") {

        on("construction") {
            var unit = NetworkLayer(2,  { createNeuron<Perceptron>() })

            it("should make a layer with a given number of specified Neurons") {
                assertThat(unit.neuronList).hasSize(2)
                                           .allSatisfy({ assertThat(it).isInstanceOf(Perceptron::class.java)})

            }
        }

        on("processing perceptrons") {
            var unit = NetworkLayer(2,  { createNeuron<Perceptron>() })

            it("should calculate the output state") {
                assertThat(unit.process(listOf(2.0, 4.0)))
                        .hasSize(2)
                        .containsExactly(1.0, 1.0)
            }
        }

        on("processing sigmoid neurons") {
            var unit = NetworkLayer(2,  { createNeuron<SigmoidNeuron>() })

            it("should calculate the output state") {
                assertThat(unit.process(listOf(-0.4, 0.2)))
                        .hasSize(2)
                        .allSatisfy { assertThat(it).isCloseTo(0.94, Offset.offset(0.01)) }
            }
        }
    }
})

