package neurons

import java.lang.Math.exp

open class LogisticNeuron(var bias: Double = 3.0,
                          var weights: List<Double> = emptyList(),
                          val activationFunction: (Double) -> Double ) {

    fun fire(inputs: List<Double> = emptyList()) : Double {
        if (weights.isEmpty()) {
            weights = (1..inputs.size).map { 1.0 }
        }
        val weightedInput = inputs.zip(weights)
        val weightedSum = weightedInput.sumByDouble { it.first * it.second }
        return activationFunction( weightedSum + bias )
    }
}

val stepFunction =  { biasedSum: Double -> if (biasedSum > 0) 1.0 else 0.0 }
val sigmoidFunction =  { biasedSum: Double -> 1.0 / (1.0 + exp(-biasedSum)) }
