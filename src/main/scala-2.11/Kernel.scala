import breeze.linalg.{DenseMatrix, DenseVector}


abstract class Kernel {

  def _kernel(x: Double, y: Double): Double

  def getCovarianceMatrix(x: DenseVector[Double], y: DenseVector[Double]): DenseMatrix[Double] = {

    DenseMatrix.tabulate(x.length,y.length){
      case(i,j) => _kernel(x(i),y(j))
    }
  }
}
