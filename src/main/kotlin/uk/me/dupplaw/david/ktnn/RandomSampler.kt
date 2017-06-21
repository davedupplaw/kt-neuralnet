package uk.me.dupplaw.david.ktnn

import koma.matrix.Matrix
import koma.matrix.mtj.MTJMatrixFactory
import java.util.*

class RandomSampler( val originalValues: Matrix<Double>, val sampleSize: Int = 1 ) {
    var shuffledValues: Matrix<Double>? = null
    var iterator: Iterator<Matrix<Double>>? = null

    init {
        if( (originalValues.numCols() / sampleSize).toDouble() !=
                originalValues.numCols() / sampleSize.toDouble() ) {
            throw IllegalArgumentException("sample size should be whole division of number of samples")
        }

        this.reset()
    }

    fun hasNext(): Boolean {
        return iterator!!.hasNext()
    }

    fun next(): Matrix<Double> {
        return iterator!!.next()
    }

    fun reset() {
        this.shuffledValues = originalValues.copy().shuffle()
        this.iterator = this.shuffledValues!!.mapColsToList { it }
                                             .withIndex()
                                             .groupBy { it.index / (shuffledValues!!.numCols()/sampleSize) }
                                             .map { (_,v) ->
                                                 val subMat = MTJMatrixFactory().zeros(shuffledValues!!.numRows(), sampleSize)
                                                 v.forEachIndexed { indx, it ->
                                                    subMat.setCol(indx, it.value)
                                                 }
                                                 subMat
                                             }.toList()
                                             .iterator()
    }
}

fun Matrix<Double>.shuffle() : Matrix<Double> {
    val rg : Random = Random()
    for (i in 0..this.numCols() - 1) {
        val randomPosition = rg.nextInt( this.numCols() )
        val tmp = this.getCol(i)
        this.setCol( i, this.getCol(randomPosition) )
        this.setCol( randomPosition, tmp )
    }
    return this
}