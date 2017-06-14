package uk.me.dupplaw.david.ktnn
import koma.mat
import org.assertj.core.api.KotlinAssertions.assertThat
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.junit.platform.runner.JUnitPlatform
import org.junit.runner.RunWith

@RunWith(JUnitPlatform::class)
internal class InputNetworkLayerTest : Spek({
    describe("an inputNetworkLayer") {
        val unit = InputNetworkLayer(10)
        on("construction") {
            it("should make a layer with a given number of Neurons") {
                assertThat(unit.numberOfNeurons).isEqualTo(10)
            }
        }

        on("process") {
            val input = mat[2.0,4.0].T

            unit.process(input)
            val output = unit.layerOutput!!

            it("should return the input as the output") {
                assertThat(output).isEqualTo(input)
            }
        }
    }
})