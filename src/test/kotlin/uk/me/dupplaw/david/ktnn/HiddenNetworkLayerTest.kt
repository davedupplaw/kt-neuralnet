package uk.me.dupplaw.david.ktnn
import koma.mat
import koma.util.test.assertMatrixEquals
import org.assertj.core.api.KotlinAssertions.assertThat
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.junit.platform.runner.JUnitPlatform
import org.junit.runner.RunWith

@RunWith(JUnitPlatform::class)
internal class HiddenNetworkLayerTest : Spek({
    describe("HiddenNetworkLayer") {
        val numberOfNeuronsInPreviousLayer = 2
        val inputNetworkLayer = InputNetworkLayer(numberOfNeuronsInPreviousLayer)

        on("construction") {
            val numberOfNeurons = 10
            val unit = HiddenNetworkLayer(numberOfNeurons, StepFunction(), MeanSquaredError())

            it("should make a layer with a given number of Neurons") {
                assertThat(unit.numberOfNeurons).isEqualTo(numberOfNeurons)
            }
        }

        on("setting the previous layer") {
            val numberOfNeurons = 10
            val unit = HiddenNetworkLayer(numberOfNeurons, StepFunction(), MeanSquaredError())
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
            val unit = HiddenNetworkLayer(2, StepFunction(), MeanSquaredError())
            unit.previousLayer = inputNetworkLayer
            unit.weightMatrix?.fill( { _,_ -> 1.0 } )
            unit.biasVector?.fill( { _,_ -> 3.0 } )

            unit.process( mat[2.0, 4.0].T )
            val result = unit.layerOutput!!

            it("should calculate the output state") {
                assertThat(result.numRows()).isEqualTo( 2 )
                assertThat(result.numCols()).isEqualTo( 1 )
                assertMatrixEquals(mat[1.0, 1.0].T, result)
            }
        }

        on("processing sigmoid neurons") {
            val unit = HiddenNetworkLayer(2, SigmoidFunction(), MeanSquaredError())
            unit.previousLayer = inputNetworkLayer
            unit.weightMatrix?.fill( { _,_ -> 1.0 } )
            unit.biasVector?.fill( { _,_ -> 3.0 } )

            unit.process( mat[-0.4, 0.2].T )
            val result = unit.layerOutput!!

            it("should calculate the output state") {
                assertThat(result.numRows()).isEqualTo(2)
                assertThat(result.numCols()).isEqualTo(1)
                assertThat(result.getDoubleData().asList())
                        .allSatisfy { assertThat(it).isCloseTo(0.94, org.assertj.core.data.Offset.offset(0.01)) }
            }
        }
    }
})

