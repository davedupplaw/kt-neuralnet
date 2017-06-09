package neurons

import org.assertj.core.api.KotlinAssertions.assertThat
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.junit.platform.runner.JUnitPlatform
import org.junit.runner.RunWith

@RunWith(JUnitPlatform::class)
class PerceptronTest : Spek({
	describe("Perceptron") {
		var unit = Perceptron()

		it("should have default bias of 3") {
			assertThat( unit.bias ).isEqualTo( 3.0 )
		}

		on("being constructed with inputs and weight") {
            val desiredWeights = listOf(3.0,4.0,5.0)
			unit = Perceptron(-4.0, desiredWeights)

			it("should set the fields as expected") {
				assertThat(unit.bias).isEqualTo(-4.0)
				assertThat(unit.weights).isEqualTo(desiredWeights)
			}
		}

		on("being constructed with values that should make it fire") {
            val desiredWeights = listOf(-2.0,-2.0)
			unit = Perceptron(3.0, desiredWeights)

			it("should return 1 on being fired" ) {
			    val givenInputs = listOf(0.0, 0.0)
				assertThat( unit.fire(givenInputs) ).isEqualTo( 1.0 )
			}
		}

		on("being constructed with values that should not make it fire") {
            val desiredWeights = listOf(-2.0,-2.0)
			unit = Perceptron(3.0, desiredWeights)

			it("should return 0 on being fired" ) {
			    val givenInputs = listOf(1.0, 1.0)
				assertThat( unit.fire(givenInputs) ).isEqualTo( 0.0 )
			}
		}
	}
} )