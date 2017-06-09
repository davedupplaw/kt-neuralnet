package neurons

import org.assertj.core.api.KotlinAssertions.assertThat
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.junit.platform.runner.JUnitPlatform
import org.junit.runner.RunWith

@RunWith(JUnitPlatform::class)
internal class InputNeuronTest: Spek({
    describe("an input neuron") {
        var unit = InputNeuron(1.0)

        on("construction") {
            it("should have a bias of 0") {
                assertThat( unit.bias ).isEqualTo( 0.0 );
            }

            it("should have a unweighted input") {
                assertThat( unit.weights ).hasSize(1)
                                         .contains(1.0)
            }
        }

        on("firing") {
            it("should return the input value") {
                for( i in 1..5 ) {
                    unit = InputNeuron( i.toDouble() );
                    assertThat( unit.fire() ).isEqualTo( i.toDouble() );
                }
            }
        }
    }
})