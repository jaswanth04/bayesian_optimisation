import breeze.linalg.{DenseVector, argmax, max}
import breeze.plot.{Figure, plot}
import breeze.stats.distributions.Rand

class BayesianOptimisation(x_space: DenseVector[Double], f: Double => Double, noiseVariance: Double, kernel: Kernel, acquisitionFunction: AcquisitionFunction) {

  val GP = new GaussianProcess(kernel,noiseVariance)

  val randInt: Rand[Int] = Rand.randInt(0,x_space.length)
  val initialPointIndices = Array(randInt.draw(),randInt.draw())
  val x_initial = new DenseVector(initialPointIndices.map(x_space(_)))
  var y: DenseVector[Double] = x_initial.map(f)

  def plotIteration(mu: DenseVector[Double], s: DenseVector[Double], acquisitionFunctionValues: DenseVector[Double], iteration: Int): Unit = {
        val fig1= Figure()
        val plt11 = fig1.subplot(2,1,0)

        plt11 += plot(x_space, mu, name = "Mean")
        plt11 += plot(x_space, mu + 2.0 * s, name = "Upper Boundary")
        plt11 += plot(x_space, mu - 2.0 * s, name = "Lower Boundary")

        plt11.title = "Mean and Boundaries - After Training - " + iteration.toString
        plt11.legend = true

        val plt12 = fig1.subplot(2,1,1)

        plt12 += plot(x_space,acquisitionFunctionValues)
        plt12.title = "Acquisition Function Values"

        fig1.refresh()

  }

  var i = 1

  def improve(x_nextPoint: Int): Int = {

    val y_nextPoint: Double = y(x_space(x_nextPoint))

    y = DenseVector.vertcat(y,DenseVector(y_nextPoint))

    GP.update(DenseVector(x_space(x_nextPoint)), DenseVector(y_nextPoint))

    val (mu_new,s_new,sigma_new) = GP.getMeanAndStandardDeviation(x_space)
    val e = acquisitionFunction.getImprovementValues(max(y),mu_new,s_new)

    println("Next element to consider after iteration " + i.toString + " is " + argmax(e).toString)
    plotIteration(mu_new,s_new,e,i)

    i = i + 1

    argmax(e)

  }

}
