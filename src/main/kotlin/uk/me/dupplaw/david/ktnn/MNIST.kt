package uk.me.dupplaw.david.ktnn

import koma.matrix.Matrix
import koma.matrix.mtj.MTJMatrixFactory
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream

class MNIST {
    var trainingImages: Matrix<Double>? = null
    var trainingLabels: Matrix<Double>? = null
    var testImages: Matrix<Double>? = null
    var testLabels: Matrix<Double>? = null

    fun read(directory: File) : MNIST {
        val trainingImagesFile = directory.resolve("train-images-idx3-ubyte")
        val trainingLabelsFile = directory.resolve("train-labels-idx1-ubyte")
        val testImagesFile = directory.resolve("t10k-images-idx3-ubyte")
        val testLabelsFile = directory.resolve("t10k-labels-idx1-ubyte")

        println( "Reading training images...." )
        trainingImages = MNISTImageFileReader(trainingImagesFile).read()
        trainingLabels = MNISTLabelFileReader(trainingLabelsFile, trainingImages!!).read()

        println( "Reading test images...." )
        testImages = MNISTImageFileReader(testImagesFile).read()
        testLabels = MNISTLabelFileReader(testLabelsFile, testImages!!).read()

        println( "Data Reading Done!" )

        return this
    }
}

//data class MNISTImage(val width: Int, val height: Int, val data: Matrix<Double>, var label: Int? = null)

class MNISTImageFileReader(val imageFile: File) {
    fun read(): Matrix<Double> {
        DataInputStream(BufferedInputStream(FileInputStream(imageFile))).use { dis ->
            val magicNumber = dis.readInt()
            if (magicNumber != 2051)
                throw IllegalStateException("Magic number should be 2051")

            val numberOfImages = dis.readInt()
            val numberOfRowsPerImage = dis.readInt()
            val numberOfColumnsPerImage = dis.readInt()

            println( "\tReading $numberOfImages images" )
            println( "\tEach image is $numberOfColumnsPerImage x $numberOfRowsPerImage" )

            val matrix = MTJMatrixFactory().zeros(numberOfRowsPerImage*numberOfColumnsPerImage, numberOfImages)
            (1..numberOfImages).map {
                for (p in 0 until numberOfRowsPerImage * numberOfColumnsPerImage) {
                    matrix[p, it-1] = dis.readUnsignedByte() / 255.0
                }
            }
            return matrix
        }
    }
}

class MNISTLabelFileReader(val labelFile: File, var trainingImages: Matrix<Double>) {
    fun read() : Matrix<Double> {
        DataInputStream(BufferedInputStream(FileInputStream(labelFile))).use { dis ->
            val magicNumber = dis.readInt()
            if (magicNumber != 2049)
                throw IllegalStateException("Magic number should be 2049")

            val numLabel = dis.readInt()

            println( "\tReading $numLabel labels" )

            val labelMatrix = MTJMatrixFactory().zeros( 1, trainingImages.numCols() )
            (1..numLabel).map {
                labelMatrix[0,it-1] = dis.readUnsignedByte().toDouble()
            }
            return labelMatrix
        }
    }
}

fun main(args: Array<String>) {
    val mnist = MNIST().read(File("/home/extreme/mnist"))

    var network = NeuralNetwork(listOf(InputNetworkLayer(784),
                                       HiddenNetworkLayer(30, SigmoidFunction(), MeanSquaredError()),
                                       HiddenNetworkLayer(10, SigmoidFunction(), MeanSquaredError())
                                ))

    println( "Network: $network")

    println( "Training...")
    (0..10).map {
        print("    - Iteration $it...")

        network.training(mnist.trainingImages!!, mnist.trainingLabels!!)

        val avg = network.trainedErrors[ network.trainedErrors.lastIndex ].mean()
        println("... Average error: $avg")
    }

    println( "Testing...")
    var nCorrect = 0
    (0..mnist.testImages!!.numCols()-1).map{ indx ->
        val image = mnist.testImages!!.selectCols(indx)
        val number = mnist.testLabels!![0,indx].toInt()

        network.feedforward( image )

        val maxNeuron = network.networkLayers[ network.networkLayers.lastIndex ].layerOutput!!.argMax();

        println( "Test image $indx should be $number classified as $maxNeuron" )

        if( maxNeuron == number ) {
            nCorrect++
        }
    }

    println( "Number correct: $nCorrect / ${mnist.testImages!!.numCols()} - ${nCorrect / mnist.testImages!!.numCols().toDouble()*100.0}%")
}