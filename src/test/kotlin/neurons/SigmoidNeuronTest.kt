package neurons

import org.assertj.core.api.KotlinAssertions.assertThat
import org.assertj.core.data.Offset
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.junit.platform.runner.JUnitPlatform
import org.junit.runner.RunWith

@RunWith(JUnitPlatform::class)
internal class SigmoidNeuronTest : Spek({
    describe("a sigmoid neuron"){
        var unit = SigmoidNeuron()

        it("should have default bias of 3") {
            assertThat(unit.bias).isEqualTo( 3.0 )
        }

        it("should be an extension of a LogisticNeuron") {
            assertThat(unit).isInstanceOf(LogisticNeuron::class.java)
        }

        on("being constructed with very large inputs and weights") {
            val inputs = listOf(100.0,100.0,100.0)
            val weights = listOf(100.0,100.0,100.0)
            unit = SigmoidNeuron(3.0, inputs, weights)

            it("should return a value close to 1") {
                assertThat(unit.fire()).isCloseTo(1.0, Offset.offset(0.01))
                                       .isLessThanOrEqualTo(1.0)
            }
        }

        on("being constructed with very large negative inputs and weights") {
            val inputs = listOf(-100.0,-100.0,-100.0)
            val weights = listOf(100.0,100.0,100.0)
            unit = SigmoidNeuron(-3.0, inputs, weights)

            it("should return a value close to 0") {
                assertThat(unit.fire()).isCloseTo(0.0, Offset.offset(0.01))
                        .isGreaterThanOrEqualTo(0.0)
            }
        }

        on("being constructed with moderate values and weights") {
            val inputs = listOf(0.25,0.25)
            val weights = listOf(-2.0,-2.0)
            unit = SigmoidNeuron(-1.0, inputs, weights)

            it("should return a value close to 0.12") {
                assertThat(unit.fire()).isCloseTo(0.12, Offset.offset(0.01))
            }
        }
    }
})
