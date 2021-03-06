import breeze.linalg.{DenseVector, _}
import breeze.numerics._
import breeze.stats.distributions.Rand

object bayesOpt_trial {

  def main(args: Array[String]): Unit = {
    
    // Generating points between -5,5
    val n = 50
    val x_test: DenseMatrix[Double] = linspace(-10, 10, n).toDenseMatrix.t

    // Defining a sin function with some noise
    def f(x: Double): Double = sin(x) * 0.9 * (x - 5)
    val noiseVariance = 0.0005
    def randGen: Rand[Double] = Rand.uniform
    def y(x: DenseVector[Double]): Double = f(x(0)) + randGen.draw() * 8000 * noiseVariance

    val y_test = new DenseVector((0 until x_test.rows).map(x_test(_,::).t).map(y).toArray)
    println(" Y values: " + y_test)
    println("The actual maximum is: " + max(y_test) + " and is at the position: " + argmax(y_test))

    val expkernel = ExpSquaredKernel(0.1)
    val acqFunc = ExpectedImprovement(0.05)

    val optimizer = new BayesianOptimisation(x_space = x_test,
      f= y,
      initialPointIndicesToEvaluate = Array(3,44),
      noiseVariance = noiseVariance,
      kernel = expkernel,
      acquisitionFunction = acqFunc)

    optimizer.runOptimization(10)
    
  }


}
