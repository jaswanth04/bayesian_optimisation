import breeze.linalg.{DenseVector, _}
import breeze.numerics._
import breeze.stats.distributions.Rand

object bayesOpt_trial {

  def main(args: Array[String]): Unit = {
    
    // Generating points between -5,5
    val n = 50
    val x_test: DenseVector[Double] = linspace(-5, 5, n)

    // Defining a sin function with some noise
    def f(x: Double): Double = sin(x) * 0.9 * (x - 5)
    val noiseVariance = 0.0005
    def randGen: Rand[Double] = Rand.uniform
    def y(x: Double): Double = f(x) + randGen.draw() * 8000 * noiseVariance

    val y_test = x_test.map(y)
    println(" Y values: " + y_test)
    println("The actual maximum is: " + max(y_test) + " and is at the position: " + argmax(y_test))

    val expkernel = ExpSquaredKernel(0.1)
    val acqFunc = ExpectedImprovement(0.01)

    val optimizer = new BayesianOptimisation(x_space = x_test,
      f= y,
      initialPointIndicesToEvaluate = Array(),
      noiseVariance = noiseVariance,
      kernel = expkernel,
      acquisitionFunction = acqFunc)

    optimizer.runOptimization(10)
    
  }


}
