package uk.me.dupplaw.david.ktnn

import koma.matrix.Matrix
import koma.matrix.mtj.MTJMatrixFactory
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream

class MNIST {
    var trainingImages: List<MNISTImage>? = null
    var testImages: List<MNISTImage>? = null

    fun read(directory: File) : MNIST {
        val trainingImagesFile = directory.resolve("train-images-idx3-ubyte")
        val trainingLabelsFile = directory.resolve("train-labels-idx1-ubyte")
        val testImagesFile = directory.resolve("t10k-images-idx3-ubyte")
        val testLabelsFile = directory.resolve("t10k-labels-idx1-ubyte")

        println( "Reading training images...." )
        trainingImages = MNISTImageFileReader(trainingImagesFile).read()
        MNISTLabelFileReader(trainingLabelsFile, trainingImages!!).read()

        println( "Reading test images...." )
        testImages = MNISTImageFileReader(testImagesFile).read()
        MNISTLabelFileReader(testLabelsFile, testImages!!).read()

        println( "Data Reading Done!" )

        return this
    }
}

data class MNISTImage(val width: Int, val height: Int, val data: Matrix<Double>, var label: Int? = null)

class MNISTImageFileReader(val imageFile: File) {
    fun read(): List<MNISTImage> {
        DataInputStream(BufferedInputStream(FileInputStream(imageFile))).use { dis ->
            val magicNumber = dis.readInt()
            if (magicNumber != 2051)
                throw IllegalStateException("Magic number should be 2051")

            val numberOfImages = dis.readInt()
            val numberOfRowsPerImage = dis.readInt()
            val numberOfColumnsPerImage = dis.readInt()

            println( "\tReading $numberOfImages images" )
            println( "\tEach image is $numberOfColumnsPerImage x $numberOfRowsPerImage" )

            return (1..numberOfImages).map {
                val matrix = MTJMatrixFactory().zeros(numberOfRowsPerImage*numberOfColumnsPerImage, 1)

                for (p in 0 until numberOfRowsPerImage * numberOfColumnsPerImage) {
                    matrix[p, 0] = dis.readUnsignedByte() / 255.0
                }

                MNISTImage(numberOfColumnsPerImage, numberOfRowsPerImage, matrix)
            }.toList()
        }
    }
}

class MNISTLabelFileReader(val labelFile: File, var trainingImages: List<MNISTImage>) {
    fun read() {
        DataInputStream(BufferedInputStream(FileInputStream(labelFile))).use { dis ->
            val magicNumber = dis.readInt()
            if (magicNumber != 2049)
                throw IllegalStateException("Magic number should be 2049")

            val numLabel = dis.readInt()

            println( "\tReading $numLabel labels" )

            (1..numLabel).map {
                trainingImages.get(it-1).label = dis.readUnsignedByte()
            }
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

    val givenInput = mnist.testImages!![0].data
    network.feedforward( givenInput )
}