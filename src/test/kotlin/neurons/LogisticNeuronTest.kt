package neurons

import org.assertj.core.api.KotlinAssertions.assertThat
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.junit.platform.runner.JUnitPlatform
import org.junit.runner.RunWith

@RunWith(JUnitPlatform::class)
internal class LogisticNeuronTest: Spek({
    describe("A logistic neuron") {
        var unit = LogisticNeuron(0.0, emptyList(), {v -> v})

        on("firing with an empty weight list") {
            var result = unit.fire( listOf( 1.0, 2.0, 3.0, 4.0 ) )

            it("should not weight the inputs") {
                assertThat(result).isEqualTo( 10.0 )
            }
        }
    }
})