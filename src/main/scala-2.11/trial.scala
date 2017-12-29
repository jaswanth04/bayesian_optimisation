import breeze.linalg.{Axis, DenseMatrix, DenseVector, diag, mpow, sum}
import breeze.numerics.{exp, pow}
import breeze.stats.distributions.Rand

object trial {

  def main(args: Array[String]): Unit = {

//    val a = DenseMatrix((1,2),(3,4))
//    val b = DenseMatrix((4,5),(6,7))
//
//    println(a.t)
//    println("---------")
//    println(b.t)
//    println("----------")
//    println((a*b).t)
//    println("-------------")
//    println(b.t * a.t)
//
//    val c: DenseMatrix[Double] = a.mapValues(math.pow(_,2))

//    println(exp(a))

//    val u = DenseVector(1.0,3.0,5.0,6.0)
//    val v = DenseVector(2.0,9.0,3.0,8.0)
//    val x = DenseVector(1.0,2.0,3.0,4.0)
//    val y = DenseVector(2.0,4.0,6.0,8.0)
//
//
//    println(u :* ((v :* x) + y))
//
//    println(x :* y)
//    println(x)
//    println(y)

    val randInt = Rand.randInt

    println(randInt.draw(),randInt.draw())

  }

}
