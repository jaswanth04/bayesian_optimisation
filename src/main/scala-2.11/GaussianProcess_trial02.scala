import breeze.linalg.{DenseMatrix, DenseVector, cholesky, linspace}
import breeze.numerics.sin
import breeze.plot.{Figure, plot}
import breeze.stats.distributions.{Gaussian, Rand}

import scala.math.{exp, pow}

object GaussianProcess_trial02 {

  def main(args: Array[String]): Unit = {
    val n = 50
    val x_test = linspace(-5,5,n)

    def kernel(x: Double, y: Double, param: Double =0.1) = exp(-0.5*(1/param)*pow(x-y,2))

    def covMat(x: DenseVector[Double], y: DenseVector[Double]) = {

      DenseMatrix.tabulate(x.length,y.length){
        case(i,j) => kernel(x(i),y(j))
      }
    }

    val randNormal = Gaussian(0,1)

    def f(x: Double) = sin(x) * 0.9 * (x-5)
    val noiseVariance = 0.0005
    def randGen = Rand.uniform
    def y(x: Double) = f(x) + randGen.draw()*8000*noiseVariance

    val x_train_indices = Array(4,18,29,35,47,25,22,17,19,10)
    val x_train = new DenseVector(x_train_indices.map(x_test(_)))

    val y_train = x_train.map(y)

    val K_train = covMat(x_train,x_train)
    val L_train = cholesky(K_train)

    val m_train = L_train \ y_train
    val alpha_train = L_train.t \ m_train

    val K_s = covMat(x_train,x_test)
    val K_ss = covMat(x_test,x_test)

    val mu = K_s.t * alpha_train

    println(mu)

  }
}
