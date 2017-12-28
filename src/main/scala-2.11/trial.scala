import breeze.linalg.DenseMatrix

object trial {

  def main(args: Array[String]): Unit = {

    val a = DenseMatrix((1,2),(3,4))
    val b = DenseMatrix((4,5),(6,7))

    println(a.t)
    println("---------")
    println(b.t)
    println("----------")
    println(a*b)
    println("-------------")
    println(a.t * b.t)

  }

}
