package nodes

class Perceptron(var bias: Double = -2.0,
				 var inputs: List<Double> = emptyList(),
				 var weights: List<Double> = emptyList() ) {
	fun fire() : Int {
		val weightedSum = inputs.zip(weights).sumByDouble { (first, second) -> first * second }
		return if(weightedSum + bias > 0) 1 else 0
	}
}
