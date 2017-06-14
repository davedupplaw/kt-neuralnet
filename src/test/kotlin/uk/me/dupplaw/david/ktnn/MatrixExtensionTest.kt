package uk.me.dupplaw.david.ktnn

import koma.end
import koma.mat
import koma.util.test.assertMatrixEquals
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import org.junit.platform.runner.JUnitPlatform
import org.junit.runner.RunWith


@RunWith(JUnitPlatform::class)
internal class MatrixExtensionTest: Spek({
    describe("the extensions to the matrix class") {
        val unit = mat[ 5, 10 end
                       15, 20 ]
        on("hadamard") {
            val expectedMatrix = mat[ 15.0, 2  end
                                       7.5, 40 ]

            val otherMatrix =    mat[ 3.0, 0.2 end
                                      0.5, 2.0 ]

            val resultMatrix = unit ʘ otherMatrix

            it("should return a new matrix where the elements ij are the product of elements ij of the original two matricies") {

                assertMatrixEquals(expectedMatrix, resultMatrix)
            }
        }
    }
})