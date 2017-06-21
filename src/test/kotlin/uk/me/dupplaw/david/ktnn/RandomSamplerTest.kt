package uk.me.dupplaw.david.ktnn

import koma.matrix.mtj.MTJMatrixFactory
import koma.util.test.assertMatrixEquals
import org.assertj.core.api.KotlinAssertions.assertThat
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.jetbrains.spek.api.dsl.xon
import org.junit.platform.runner.JUnitPlatform
import org.junit.runner.RunWith

@RunWith(JUnitPlatform::class)
internal class RandomSamplerTest: Spek({
    describe("A Random Sampler") {
        val input = MTJMatrixFactory().zeros(10,12).fill { _,_ -> Math.random() }

        xon("being constructed with a matrix") {
            val unit = RandomSampler(input)

            it("should create a randomised version of the matrix") {
                assertThat( unit.shuffledValues ).isNotNull();
                assertThat( unit.shuffledValues!!.numRows() ).isEqualTo( input.numRows() )
                assertThat( unit.shuffledValues!!.numCols() ).isEqualTo( input.numCols() )
                assertThat( unit.shuffledValues!!.getDoubleData() ).isNotEqualTo( input.getDoubleData() )

                (0..input.numCols()-1).forEach {
                    assertThat( unit.shuffledValues!!.mapColsToList{ it.getDoubleData() } ).contains( input.getCol(it).getDoubleData() );
                }
            }

            it("should reshuffle the matrix on reset") {
                var lastMatrix = input.getDoubleData()
                (0..10).forEach {
                    unit.reset()

                    val shuffledMatrix = unit.shuffledValues!!.getDoubleData();
                    assertThat( shuffledMatrix ).isNotEqualTo( lastMatrix )

                    lastMatrix = shuffledMatrix
                }
            }

            it("should be able to iterate through the shuffled matrix") {
                val shuffledData = unit.shuffledValues!!
                (0..shuffledData.numCols()-1).forEach {
                    assertMatrixEquals( shuffledData.getCol(it), unit.next() )
                }
            }

            it("should return true if there are more iterations and false otherwise") {
                unit.reset()
                val shuffledData = unit.shuffledValues!!
                (0..shuffledData.numCols()-1).forEach {
                    assertThat( unit.hasNext() ).isTrue()
                    unit.next()
                }
                assertThat( unit.hasNext() ).isFalse()
            }
        }

        on("construction with a sample size") {
            val unit = RandomSampler(input, 4)

            it("should return a sample of the expected size") {
                assertThat(unit.next().numRows()).isEqualTo(10)
                assertThat(unit.next().numCols()).isEqualTo(4)
            }
        }

        xon("construction with sample size not integer division of number of samples") {
            var caughtException : IllegalArgumentException? = null
            try {
                RandomSampler(input, 3)
            } catch( illegalArgumentException: IllegalArgumentException ) {
                caughtException = illegalArgumentException
            }

            it("should throw an IllegalArgumentException") {
                assertThat( caughtException ).isNotNull()
            }
        }
    }
})