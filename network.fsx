#load "packages/MathNet.Numerics.FSharp/MathNet.Numerics.fsx"

open MathNet.Numerics.LinearAlgebra
//open MathNet.Numerics.LinearAlgebra.Double

type Net = {
  numLayers : int
  sizes : int list
  biases : (Matrix<float>) list
  weights : (Matrix<float>) list
}

type TrainingData = (Matrix<float> * Matrix<float>) list

let network sizes =
  {
    numLayers = List.length sizes
    sizes = sizes
    biases = [for y in sizes.[1..] -> DenseMatrix.randomStandard<float>y 1]
    weights = [for x, y in List.zip (List.take (sizes.Length - 1) sizes) sizes.[1..]  -> 
                DenseMatrix.randomStandard<float> y x ]
  }


let sigmoidE (z:float) : float = 1.0 / (1.0 + exp (-z))
let sigmoid : (Matrix<float> -> Matrix<float>) = Matrix.map sigmoidE

let feedforward net a =
  List.zip net.biases net.weights 
  |> List.fold (fun s (b,w) -> sigmoid ((w * s) + b) ) a


let backprop net x y =
  (x, y)


let updateMiniBatch net miniBatch eta =
  let zeroB = [for b in net.biases -> DenseMatrix.zero<float> (b.RowCount) (b.ColumnCount)]
  let zeroW = [for w in net.weights -> DenseMatrix.zero<float> (w.RowCount) (w.ColumnCount)]

  let nablaB, nablaW = 
    miniBatch
    |> Seq.fold (fun (nablab, nablaw) (x, y) -> 
        let deltaNablaB, deltaNablaW = backprop net x y
        ( [for (nb, dnb) in Seq.zip nablab deltaNablaB -> nb + dnb], 
          [for (nw, dnw) in Seq.zip nablaw deltaNablaW -> nw + dnw]
        )
        ) (zeroB, zeroW)

  { net with 
      biases = [for b, nb in Seq.zip net.biases nablaB -> b - (eta / float (Seq.length miniBatch)) * nb]
      weights = [for w, nw in Seq.zip net.biases nablaW -> w - (eta / float (Seq.length miniBatch)) * nw]
  }

let SGD net (trainingData: TrainingData) epochs miniBatchSize eta testData = 
  let nTest = Seq.length testData
  let n = Seq.length trainingData
  seq {1 .. epochs}
  |> Seq.fold (fun net j -> 
    let miniBatches = [for k in [0 .. miniBatchSize .. n] ->
                        trainingData.[k..(min (n-1) (k+miniBatchSize-1))]
                      ]
    miniBatches
    |> Seq.fold (fun n mb -> updateMiniBatch n mb eta) net
  ) net

let net = network [2;3;2]

updateMiniBatch net [] 0.001







feedforward net (DenseMatrix.randomStandard<float> 2 1)
