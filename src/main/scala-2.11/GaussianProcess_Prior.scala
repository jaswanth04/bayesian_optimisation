import breeze.linalg.linspace
import breeze.plot.{Figure,plot}


object GaussianProcess_Prior {

  def main(args: Array[String]): Unit = {

    val n = 50
    val x_test = linspace(5,-5,50)

    val GP = new GaussianProcess(ExpSquaredKernel(0.1))

    val f_prior = GP.getPriorSample(x_test,4)

    val fig = Figure()
    val plt = fig.subplot(0)

    for (prior <- f_prior)
      plt += plot(x_test,prior)

    plt.title = "Prior Samples for Gaussian Process"

    fig.refresh()
  }
}
