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

let backIt idx = Seq.rev >> Seq.skip (-idx - 1) >> Seq.head
let sset idx v sq = 
  let i = if idx < 0 then (List.length sq) + idx else idx
  let (h, _::t) = List.splitAt i sq
  h @ [v] @ t

let sigmoid (z:Matrix<float>) : Matrix<float> = 1.0 / (1.0 + (Matrix.map System.Math.Exp (-z)))


let sigmoidPrime z = (sigmoid z).PointwiseMultiply(1.0 - (sigmoid z))

let costDerivative (outputActivations:Matrix<float>) (y:Matrix<float>) : Matrix<float>= outputActivations - y 

let toMatrix (b:byte) : Matrix<float> =
  let m = DenseMatrix.zero<float> 10 1
  m.[int b, 0] <- 1.0
  m

let feedforward net a =
  List.zip net.biases net.weights 
  |> List.fold (fun s (b,w) -> sigmoid (w.PointwiseMultiply(s) + b) ) a


let backprop net x y =
  // line 96
  let zeroB = [for b in net.biases -> DenseMatrix.zero<float> (b.RowCount) (b.ColumnCount)]
  let zeroW = [for w in net.weights -> DenseMatrix.zero<float> (w.RowCount) (w.ColumnCount)]

  let activations, zs, _ = 
    Seq.zip net.biases net.weights
    |> Seq.fold (fun (as', zs, a) (b, w) -> 
          let z = w * a + b
          let act = sigmoid z
          (as' @ [act], zs @ [z], act)
        ) ([x], [], x)

  let delta = (costDerivative (List.last activations) y).PointwiseMultiply(sigmoidPrime (List.last zs)) 
  let nablaB = sset -1 delta zeroB
  let nablaW = sset -1 (delta * (backIt -2 activations).Transpose()) zeroW


  let (nB, nW, _) = 
    [2 .. net.numLayers]
    |> Seq.fold (
      fun (nb, nw, d) l ->
        let z = backIt -l zs
        let sp = sigmoidPrime z
        let delta = ((backIt (-l + 1) net.weights).Transpose() ) * d
        let nablab = sset -l delta nb
        let nablaw = sset -l (delta * ( (backIt (-l - 1) activations).Transpose())) nw
        (nablab, nablaw, delta)
      ) (nablaB, nablaW, delta)
  (nB, nW)


let updateMiniBatch net (miniBatch: TrainingData) eta =
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

let net = network [28*28;30;10]

#load "mnist.fsx"


let data =  Mnist.getData()

let trd, vld, tsd = data

let trainingData:TrainingData = 
  trd
  |> Array.map (fun (d, l) -> (DenseMatrix.ofColumnArrays [| Array.map (fun px ->  (float px) / 255.0 ) d |], toMatrix l) )
  |> Array.toList

SGD net trainingData 30 10 3.0 []


feedforward net (DenseMatrix.randomStandard<float> 2 1)
