import breeze.linalg.{DenseVector, argmax, max}
import breeze.plot.{Figure, plot}
import breeze.stats.distributions.Rand

class BayesianOptimisation(x_space: DenseVector[Double],
                           f: Double => Double,
                           initialPointIndicesToEvaluate: Array[Int] = Array(),
                           noiseVariance: Double,
                           kernel: Kernel,
                           acquisitionFunction: AcquisitionFunction) {

  val GP = new GaussianProcess(kernel,noiseVariance)

  val randInt: Rand[Int] = Rand.randInt(0,x_space.length)
  val initialPointIndices = if (initialPointIndicesToEvaluate.isEmpty) Array(randInt.draw(),randInt.draw()) else initialPointIndicesToEvaluate
  println("Evaluating Random positions: " + initialPointIndices.toList)
  var x_evaluated = new DenseVector(initialPointIndices.map(x_space(_)))
  var y_evaluated: DenseVector[Double] = x_evaluated.map(f)

  def plotIteration(mu: DenseVector[Double], s: DenseVector[Double], acquisitionFunctionValues: DenseVector[Double], iteration: Int): Unit = {
        val fig1= Figure()
        val plt11 = fig1.subplot(2,1,0)

        plt11 += plot(x_space, mu, name = "Mean")
        plt11 += plot(x_space, mu + 2.0 * s, name = "Upper Boundary")
        plt11 += plot(x_space, mu - 2.0 * s, name = "Lower Boundary")

        plt11.title = "Mean and Boundaries - After Iteration - " + iteration.toString
        plt11.legend = true

        val plt12 = fig1.subplot(2,1,1)

        plt12 += plot(x_space,acquisitionFunctionValues)
        plt12.title = acquisitionFunction.name

        fig1.refresh()

  }

  def improve(x_nextPoint_index: Int, i: Int): Int = {

    val y_nextPoint: Double = f(x_space(x_nextPoint_index))

    x_evaluated = DenseVector.vertcat(x_evaluated, DenseVector(x_space(x_nextPoint_index)))
    y_evaluated = DenseVector.vertcat(y_evaluated, DenseVector(y_nextPoint))

    GP.update(DenseVector(x_space(x_nextPoint_index)), DenseVector(y_nextPoint))

    val (mu_new,s_new,sigma_new) = GP.getMeanAndStandardDeviation(x_space)
    val e = acquisitionFunction.getImprovementValues(max(y_evaluated),mu_new,s_new)

    println("The values for " + acquisitionFunction.name + " is: " + e)
    println("Next position to consider after iteration " + i.toString + " is " + argmax(e).toString)
    plotIteration(mu_new,s_new,e,i)

    argmax(e)

  }

  def runOptimization(maxIterations: Int): Unit = {

    var i = 1

    GP.update(x_evaluated,y_evaluated)
    val (mu,s,sigma) = GP.getMeanAndStandardDeviation(x_space)
    var nextPoint_index: Int = argmax(acquisitionFunction.getImprovementValues(max(y_evaluated),mu,s))

    while (i <= maxIterations){
      println("Start of Iteration: " + i + "------------")
      nextPoint_index = improve(nextPoint_index,i)
      println("Evaluated x points: " + x_evaluated)
      println("Evaluated y points: " + y_evaluated)
      println("Present maximum is: " + max(y_evaluated))
      println("End of Iteration: "+ i + "---------------")
      i = i + 1
    }

  }

}
