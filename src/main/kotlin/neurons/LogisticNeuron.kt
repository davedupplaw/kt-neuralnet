package neurons

import java.lang.Math.exp

open class LogisticNeuron(var bias: Double = 3.0,
                          var inputs: List<WeightedInput> = emptyList(),
                          val activationFunction: (Double) -> Double ) {
    fun fire() : Double {
        val weightedSum = inputs.sumByDouble { it.input * it.weight }
        return activationFunction( weightedSum + bias )
    }
}

data class WeightedInput(val input: Double, val weight: Double)

val stepFunction =  { biasedSum: Double -> if (biasedSum > 0) 1.0 else 0.0 }
val sigmoidFunction =  { biasedSum: Double -> 1.0 / (1.0 + exp(-biasedSum)) }
