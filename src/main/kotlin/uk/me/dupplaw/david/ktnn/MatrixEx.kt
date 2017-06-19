package uk.me.dupplaw.david.ktnn

import koma.matrix.Matrix

fun Matrix<Double>.dims(): String = "${this.numRows()}x${this.numCols()}"
infix fun Matrix<Double>.ʘ(that: Matrix<Double>): Matrix<Double> {
    if (this.numRows() != that.numRows()) {
        throw IndexOutOfBoundsException("A.numRows != B.numRows (${this.numRows()} != ${that.numRows()})")
    }
    if (this.numCols() != that.numCols()) {
        throw IndexOutOfBoundsException("A.numCols != B.numCols (${this.numCols()} != ${that.numCols()})")
    }

    return this.mapIndexed { row, col, ele -> that[row, col] * ele }
}