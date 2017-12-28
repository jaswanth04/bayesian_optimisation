import breeze.linalg.{DenseMatrix, DenseVector, cholesky, linspace}
import breeze.plot.{Figure,plot}
import breeze.stats.distributions.Gaussian

import scala.math.{exp, pow}

object GaussianProcess_trial01 {

  def main(args: Array[String]): Unit = {

    val n = 50
    val x_test = linspace(5,-5,50)

    def kernel(x: Double, y: Double, param: Double =0.1) = exp(-0.5*(1/param)*pow(x-y,2))

    def covMat(x: DenseVector[Double]) = {
      val xl = x.length

      DenseMatrix.tabulate(xl,xl){
        case(i,j) => kernel(x(i),x(j))
      }
    }

    val k_ss = covMat(x_test)

    val L = cholesky(k_ss)

    val randNormal = Gaussian(0,1)
    val n_gp_samples = 3
    val randMatrix = DenseMatrix.rand(n,n_gp_samples,randNormal)

    val f_prior = L * randMatrix

    val fig = Figure()
    val plt = fig.subplot(0)

    for (x <- Range(0,f_prior.cols)){
      plt += plot(x_test,f_prior(::,x))
    }

    fig.refresh()
  }
}
