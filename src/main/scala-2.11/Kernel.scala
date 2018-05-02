import breeze.linalg.{DenseMatrix, DenseVector}


abstract class Kernel {

  def _kernel(x: Double, y: Double): Double
  def _kernel(x: DenseVector[Double], y: DenseVector[Double]): Double

  def getCovarianceMatrix(x: DenseVector[Double], y: DenseVector[Double]): DenseMatrix[Double] = {

    DenseMatrix.tabulate(x.length,y.length){
      case(i,j) => _kernel(x(i),y(j))
    }
  }

  def getCovarianceMatrix(x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {

    DenseMatrix.tabulate(x.rows, y.rows){
      case(i,j) => _kernel(x(i,::).t, y(j,::).t)
    }
  }
}
