import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian

abstract class AcquisitionFunction {

  val gauss = Gaussian(0,1)

  def name: String
  def getImprovementValues(y_best: Double, mu: DenseVector[Double], s: DenseVector[Double]): DenseVector[Double]

}

case class ProbabilityOfImprovement(eps: Double) extends AcquisitionFunction {

  val name = "Probability of Improvement"
  def getImprovementValues(y_best: Double, mu: DenseVector[Double], s: DenseVector[Double]): DenseVector[Double] = ((mu - y_best - eps)/s).map(gauss.cdf)
}

case class ExpectedImprovement(eps: Double) extends AcquisitionFunction {

  val name = "Expected Improvement"
  def getImprovementValues(y_best: Double, mu: DenseVector[Double], s: DenseVector[Double]): DenseVector[Double] = {
    val Z = (mu - y_best - eps) / s

    s :* ((Z :* Z.map(gauss.cdf)) + Z.map(gauss.pdf))

  }
}
