import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, cholesky, diag, linspace, sum}
import breeze.numerics.{sin, sqrt}
import breeze.plot.{Figure, plot}
import breeze.stats.distributions.{Gaussian, Rand}

import scala.math.{exp, pow}

object GaussianProcess_Regression {

  def main(args: Array[String]): Unit = {
    val n = 50
    val x_test = linspace(-5,5,n)

    def kernel(x: Double, y: Double, param: Double =0.1) = exp(-0.5*(1/param)*pow(x-y,2))

    def covMat(x: DenseVector[Double], y: DenseVector[Double]) = {

      DenseMatrix.tabulate(x.length,y.length){
        case(i,j) => kernel(x(i),y(j))
      }
    }

    def f(x: Double) = sin(x) * 0.9 * (x-5)
    val noiseVariance = 0.0005
    def randGen = Rand.uniform
    def y(x: Double) = f(x) + randGen.draw()*8000*noiseVariance

    val x_train_indices = Array(4,18,29,35,47,25,22,17,19,10)
    val x_train = new DenseVector(x_train_indices.map(x_test(_)))

    val y_train: DenseVector[Double] = x_train.map(y)

    val K_train = covMat(x_train,x_train) + DenseMatrix.eye[Double](10) * noiseVariance
    val L_train = cholesky(K_train)

    val m_train = L_train \ y_train

    val K_s = covMat(x_train,x_test)
    val K_ss = covMat(x_test,x_test)

    val LK_train = L_train \ K_s

    val s2 = diag(K_ss) - sum(LK_train.mapValues(pow(_,2)),Axis._0).t
    val s = sqrt(s2)

    val mu = LK_train.t * m_train
    val sigma = K_ss - (LK_train.t * LK_train)

    val randNormal = Gaussian(0,1)
    val n_gp_samples = 3
    val randMatrix = DenseMatrix.rand(n,n_gp_samples,randNormal)

    def f_prior = cholesky(K_ss) * DenseVector.rand(n,randNormal)

    def f_posterior = mu + cholesky(sigma) * DenseVector.rand(n,randNormal)

//    println(K_train.rows,K_train.cols)
    println(mu)
    println(sigma)
    println(s2)
    println(s)

    val fig = Figure()

    val plt3 = fig.subplot(2,2,0)
    plt3 += plot(x_test,x_test.map(y), name = "true Function", style = '-')
    plt3.title = "True Function"

    val plt2 = fig.subplot(2,2,1)
    plt2 += plot(x_test,f_prior)
    plt2 += plot(x_test,f_prior)
    plt2 += plot(x_test,f_prior)

    plt2.title = "Prior"

    val plt = fig.subplot(2,2,2)

    plt += plot(x_test,mu,name = "mean")
    plt += plot(x_test,mu + s * 2.0, name = "upper boundary",style = '-')
    plt += plot(x_test,mu - s * 2.0, name = "upper boundary",style = '-')
//    plt += plot(x_test,x_test.map(y), name = "true Function", style = '-')

    plt.title = "Mean and boundaries"
    plt.legend = true

//    fig.refresh()

    val plt1 = fig.subplot(2,2,3)

    plt1 += plot(x_test,f_posterior)
    plt1 += plot(x_test,f_posterior)
    plt1 += plot(x_test,f_posterior)

    plt1.title = "Posterior"





    fig.refresh()
  }
}
