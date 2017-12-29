import breeze.linalg.{DenseVector, argmax, linspace, max}
import breeze.numerics.sin
import breeze.plot.{Figure, plot}
import breeze.stats.distributions.{Gaussian, Rand}

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

    val initialPoints_indices = Array(10, 25, 40)

    val x_initial = new DenseVector(initialPoints_indices.map(x_test(_)))
    val y_initial: DenseVector[Double] = x_initial.map(y)

    val GP = new GaussianProcess(ExpSquaredKernel(0.1), noiseVariance)

    GP.update(x_initial, y_initial)

    val (mean, s, sigma) = GP.getMeanAndStandardDeviation(x_test)

    val fig = Figure()
    val plt1 = fig.subplot(2,1,0)

    plt1 += plot(x_test, mean, name = "Mean")
    plt1 += plot(x_test, mean + 2.0 * s, name = "Upper Boundary")
    plt1 += plot(x_test, mean - 2.0 * s, name = "Lower Boundary")

    plt1.title = "Mean and Boundaries - After Training 1"

    var y_v = y_initial

    val y_best = max(y_initial)

//    println(y_initial)
//    println(y_best)
//    println(mean)
//    println(s)

    val gauss = Gaussian(0,1)

    def probabilityImprovement(y_best: Double, mu: DenseVector[Double],s: DenseVector[Double], eps: Double): DenseVector[Double] = ((mu - y_best - eps)/s).map(gauss.cdf)

    def expectedImprovement(y_best: Double, mu: DenseVector[Double],s: DenseVector[Double], eps: Double): DenseVector[Double] = {
      val Z = (mu - y_best - eps) / s

      s :* ((Z :* Z.map(gauss.cdf)) + Z.map(gauss.pdf))

    }

    val k = expectedImprovement(y_best,mean,s,0.1)
    println(k)

    println(probabilityImprovement(y_best,mean,s,0.1))

    val plt2 = fig.subplot(2,1,1)

    plt2 += plot(x_test,probabilityImprovement(y_best,mean,s,0.1))

    plt2.title = "PI"


    val plt3 = fig.subplot(3,1,2)

    plt3 += plot(x_test,k)
    plt3.title = "EI"

    fig.refresh()

    val x_nextPoint = argmax(k)

    println(x_nextPoint)

    var i = 2

    def improve(x_nextPoint: Int): Int = {

      val y_nextPoint: Double = y(x_test(x_nextPoint))

      y_v = DenseVector.vertcat(y_v,DenseVector(y_nextPoint))

      GP.update(DenseVector(x_test(x_nextPoint)), DenseVector(y_nextPoint))

      val (mu_new,s_new,sigma_new) = GP.getMeanAndStandardDeviation(x_test)

      val p = probabilityImprovement(max(y_v),mu_new,s_new,0.1)
      val e = expectedImprovement(max(y_v),mu_new,s_new,0.1)

      val fig1= Figure()
      val plt11 = fig1.subplot(3,1,0)

      plt11 += plot(x_test, mu_new, name = "Mean")
      plt11 += plot(x_test, mu_new + 2.0 * s_new, name = "Upper Boundary")
      plt11 += plot(x_test, mu_new - 2.0 * s_new, name = "Lower Boundary")

      plt11.title = "Mean and Boundaries - After Training - " + i.toString
      plt11.legend = true

      val plt12 = fig1.subplot(3,1,1)

      plt12 += plot(x_test,p)
      plt12.title = "PI"

      val plt13 = fig1.subplot(3,1,2)

      plt13 += plot(x_test,e)
      plt13.title = "EI"


      fig1.refresh()

      println("Next element to consider after iteration " + i.toString + " is " + argmax(e).toString)

      i = i + 1

      argmax(e)
    }


    val np2 = improve(x_nextPoint)
    val np3 = improve(np2)
    val np4 = improve(np3)







  }


}
