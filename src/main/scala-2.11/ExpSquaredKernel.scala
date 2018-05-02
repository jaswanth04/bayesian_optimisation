import breeze.linalg.{DenseVector, sum}
import breeze.numerics.{exp, pow}

case class ExpSquaredKernel(param: Double = 0.1, paramVector: DenseVector[Double] = DenseVector(0.1, 0.1, 0.1)) extends Kernel {

  def this(param: Double) = this(param, DenseVector(0.1, 0.1, 0.1))
  def this(paramVector: DenseVector[Double]) = this(0.3, paramVector)

  def _kernel(x: Double, y: Double): Double = exp(-0.5*(1/param)*pow(x-y,2))
  def _kernel(x: DenseVector[Double], y:DenseVector[Double]): Double = {
    val r2 = sum(pow(x-y,2)/pow(paramVector,2))
    exp(-0.5*r2)
  }

}
