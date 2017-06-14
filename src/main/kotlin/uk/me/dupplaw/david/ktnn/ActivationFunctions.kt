package uk.me.dupplaw.david.ktnn

interface ActivationFunction {
    fun apply( biasedSum: Double ) : Double
    fun derivative( biasedSum: Double ) : Double
}

class StepFunction : ActivationFunction {
    override fun apply( biasedSum: Double ) = if (biasedSum > 0) 1.0 else 0.0
    override fun derivative(biasedSum: Double): Double = 0.0
}

class SigmoidFunction : ActivationFunction {
    override fun apply( biasedSum: Double ) = 1.0 / (1.0 + Math.exp(-biasedSum))
    override fun derivative( biasedSum: Double ) : Double {
        val s = this.apply(biasedSum)
        return s * (1-s)
    }
}