package uk.me.dupplaw.david.ktnn

import koma.matrix.Matrix

interface CostFunction {
    fun apply(expected: Matrix<Double>, output: Matrix<Double>) : Matrix<Double>
    fun derivative(expected: Matrix<Double>, output: Matrix<Double>) : Matrix<Double>
}

class MeanSquaredError : CostFunction {
    override fun apply(expected: Matrix<Double>, output: Matrix<Double>) : Matrix<Double> {
        val difference = (expected - output)
        return difference * difference / 2
    }

    override fun derivative(expected: Matrix<Double>, output: Matrix<Double>) : Matrix<Double> {
        return output - expected
    }
}
