import breeze.linalg.{Axis, DenseMatrix, DenseVector, cholesky, diag, sum}
import breeze.stats.distributions.Gaussian
import breeze.numerics.{pow, sqrt}


/*
This is an implementation for Gaussian Process. The input to this the kernel and noiseVariance
This has methods update with the training Data and you can get Prior Samples
and Posterior Samples

There is some Optimisation possible in the update code.
  Instead of computing K_y with new Points we can only compute
the K_ss and K_s and then update the K_y with the new rows
*/

class GaussianProcess(kernel: Kernel, noiseVariance: Double = 0.0005) {

  var x: DenseVector[Double] = DenseVector()
  var x_matrix: DenseMatrix[Double] = DenseMatrix(999.0)
  var y: DenseVector[Double] = DenseVector()
  var K_y: DenseMatrix[Double] = DenseMatrix(0.0)
  var L: DenseMatrix[Double] = DenseMatrix(0.0)
  var m: DenseVector[Double] = DenseVector()

  val randNormal = Gaussian(0,1)

  def setKernel(k : Kernel) : GaussianProcess = {
    new GaussianProcess(k, noiseVariance)
  }

  def getPriorSample(x: DenseVector[Double], numberOfSamples: Int) : List[DenseVector[Double]] = {

    val n = x.length

    val K_ss = kernel.getCovarianceMatrix(x,x)
    val L = cholesky(K_ss)

    (for (i <- Range(0,numberOfSamples)) yield L * DenseVector.rand(n,randNormal)).toList
  }

  def getPriorSample(x: DenseMatrix[Double], numberOfSamples: Int) : List[DenseVector[Double]] = {

    val n = x.rows

    val K_ss = kernel.getCovarianceMatrix(x,x)
    val L = cholesky(K_ss)

    (for (i <- Range(0,numberOfSamples)) yield L * DenseVector.rand(n,randNormal)).toList
  }

  def update(x_newPoint: DenseVector[Double], y_newPoint: DenseVector[Double]): Unit = {

    x = DenseVector.vertcat(x,x_newPoint)
    y = DenseVector.vertcat(y,y_newPoint)

    K_y = kernel.getCovarianceMatrix(x,x) + DenseMatrix.eye[Double](x.length) * noiseVariance
    L = cholesky(K_y)
    m = L \ y

  }

  def update(x_newPoint: DenseMatrix[Double], y_newPoint: DenseVector[Double]): Unit = {

    x_matrix = if (x_matrix == DenseMatrix(999.0)) x_newPoint else DenseMatrix.vertcat(x_matrix,x_newPoint)
    y = DenseVector.vertcat(y,y_newPoint)

    K_y = kernel.getCovarianceMatrix(x_matrix,x_matrix) + DenseMatrix.eye[Double](x_matrix.rows) * noiseVariance
    L = cholesky(K_y)
    m = L \ y

  }


  def getMeanAndStandardDeviation(x_test: DenseVector[Double]): (DenseVector[Double], DenseVector[Double], DenseMatrix[Double]) = {
    val K_s = kernel.getCovarianceMatrix(x,x_test)
    val K_ss = kernel.getCovarianceMatrix(x_test,x_test)

    val Lk = L \ K_s

    val mu = Lk.t * m

    val s2 = diag(K_ss) - sum(pow(Lk,2),Axis._0).t
    val sigma = K_ss - (Lk.t * Lk)

    (mu,sqrt(s2),sigma)
  }

  def getMeanAndStandardDeviation(x_test: DenseMatrix[Double]): (DenseVector[Double], DenseVector[Double], DenseMatrix[Double]) = {
    val K_s = kernel.getCovarianceMatrix(x_matrix,x_test)
    val K_ss = kernel.getCovarianceMatrix(x_test,x_test)

    val Lk = L \ K_s

    val mu = Lk.t * m

    val s2 = diag(K_ss) - sum(pow(Lk,2),Axis._0).t
    val sigma = K_ss - (Lk.t * Lk)

    (mu,sqrt(s2),sigma)
  }

  def getPosteriorSample(x_test:DenseVector[Double],numberOfSamples: Int): List[DenseVector[Double]] = {
    val (mu,s,sigma) = getMeanAndStandardDeviation(x_test)

    (for (i <- Range(0,numberOfSamples)) yield
      mu + cholesky(sigma) * DenseVector.rand(x_test.length,randNormal)).toList

  }

  def getPosteriorSample(x_test:DenseMatrix[Double],numberOfSamples: Int): List[DenseVector[Double]] = {
    val (mu,s,sigma) = getMeanAndStandardDeviation(x_test)

    (for (i <- Range(0,numberOfSamples)) yield
      mu + cholesky(sigma) * DenseVector.rand(x_test.rows,randNormal)).toList

  }

}
