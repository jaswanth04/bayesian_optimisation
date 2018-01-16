import breeze.linalg.{DenseMatrix, DenseVector, argmax, linspace, max}
import breeze.numerics.{pow, sin}
import breeze.stats.distributions.Rand

/**
  * Created by jj79896 on 1/12/2018.
  */
object bayesOpt_multiDim {

  def main(args: Array[String]): Unit = {

    val x1 = linspace(-5,5,5)
    val x2 = linspace(-4,4,5)
    val x3 = linspace(-3,3,5)

    val x = for (x <- x1.toArray; y <- x2.toArray; z <- x3.toArray) yield Array(x,y,z)

    val x_test: DenseMatrix[Double] = new DenseMatrix(3,125,x.flatten).t

    // Defining a sin function with some noise
    def f(x: DenseVector[Double]) = (pow(x(0),5)*19 + sin(x(1)) * 0.9 * (x(1)-5) + pow(x(2),2)*4)/100000
    val noiseVariance = 0.0005
    def randGen = Rand.uniform
    def y(x: DenseVector[Double]) = f(x) + randGen.draw()*800*noiseVariance

    val y_test = new DenseVector((0 until x_test.rows).map(x_test(_,::).t).map(y).toArray)

    println("The actual maximum is: " + max(y_test) + " and is at the position: " + argmax(y_test) + " for the value of x: " + x_test(argmax(y_test),::))

    val expkernel = ExpSquaredKernel(paramVector = DenseVector(0.8,0.8,0.8))
    val acqFunc = ExpectedImprovement(0.1)

    val optimizer = new BayesianOptimisation(x_space = x_test,
      f= y,
      initialPointIndicesToEvaluate = Array(10,120),
      noiseVariance = noiseVariance,
      kernel = expkernel,
      acquisitionFunction = acqFunc)

    optimizer.runOptimization(30)


  }

}
