#I "packages/FsPickler/lib/net45"

#load "packages/MathNet.Numerics.FSharp/MathNet.Numerics.fsx"

#r "FsPickler.dll"

open MathNet.Numerics.LinearAlgebra
//open MathNet.Numerics.LinearAlgebra.Double

let rand = new System.Random()

let swap (a: _[]) x y =
    let tmp = a.[x]
    a.[x] <- a.[y]
    a.[y] <- tmp

// shuffle an array (in-place)
let shuffle a =
    Array.iteri (fun i _ -> swap a i (rand.Next(i, Array.length a))) a


type Net = {
  numLayers : int
  sizes : int list
  biases : (Matrix<float>) list
  weights : (Matrix<float>) list
}

type TrainingData = (Matrix<float> * Matrix<float>) []
type MiniBatch = (Matrix<float> * Matrix<float>) list

let network (sizes: int list) : Net =
  {
    numLayers = List.length sizes
    sizes = sizes
    biases = [for y in sizes.[1..] -> DenseMatrix.randomStandard<float>y 1]
    weights = [for x, y in List.zip (List.take (sizes.Length - 1) sizes) sizes.[1..]  -> 
                DenseMatrix.randomStandard<float> y x ]
  }

let backIt idx = List.rev >> List.skip (-idx - 1) >> List.head
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

let feedforward (net:Net) (a:Matrix<float>) : Matrix<float> =
  List.zip net.biases net.weights 
  |> List.fold (fun s (b,w) -> sigmoid (w.PointwiseMultiply(s) + b) ) a


let backprop (net:Net) x y =
  let zeroB = [for b in net.biases -> DenseMatrix.zero<float> (b.RowCount) (b.ColumnCount)]
  let zeroW = [for w in net.weights -> DenseMatrix.zero<float> (w.RowCount) (w.ColumnCount)]
  let activations, zs, _ = 
    List.zip net.biases net.weights
    |> List.fold (fun (as', zs, a) (b, w) -> 
          let z = w * a + b
          let act = sigmoid z
          (as' @ [act], zs @ [z], act)
        ) ([x], [], x)
  let delta = (costDerivative (List.last activations) y).PointwiseMultiply(sigmoidPrime (List.last zs)) 
  let nablaB = sset -1 delta zeroB
  let nablaW = sset -1 (delta * (backIt -2 activations).Transpose()) zeroW
  let (nB, nW, _) = 
    [2 .. (net.numLayers - 1)]
    |> List.fold (
      fun (nb, nw, d) l ->
        let z = backIt -l zs
        let sp = sigmoidPrime z
        let delta = ((backIt (-l + 1) net.weights).Transpose() ) * d
        let nablab = sset -l delta nb
        let nablaw = sset -l (delta * ( (backIt (-l - 1) activations).Transpose())) nw
        (nablab, nablaw, delta)
      ) (nablaB, nablaW, delta)
  (nB, nW)


let updateMiniBatch (net:Net) (miniBatch: MiniBatch) (eta:float) : Net =
  let zeroB = [for b in net.biases -> DenseMatrix.zero<float> (b.RowCount) (b.ColumnCount)]
  let zeroW = [for w in net.weights -> DenseMatrix.zero<float> (w.RowCount) (w.ColumnCount)]

  let nablaB, nablaW = 
    miniBatch
    |> List.fold (fun (nablab, nablaw) (x, y) -> 

        let deltaNablaB, deltaNablaW = backprop net x y

        ( [for (nb, dnb) in List.zip nablab deltaNablaB -> nb + dnb], 
          [for (nw, dnw) in List.zip nablaw deltaNablaW -> nw + dnw]
        )
        ) (zeroB, zeroW)
  { net with 
      biases = [for b, nb in List.zip net.biases nablaB -> b - (eta / float (List.length miniBatch)) * nb]
      weights = [for w, nw in List.zip net.weights nablaW -> w - (eta / float (List.length miniBatch)) * nw]
  }


let SGD (net:Net) (trainingData: TrainingData) (epochs:int) (miniBatchSize:int) (eta:float) : Net = 
//  let nTest = List.length testData
  let n = Array.length trainingData
  [1 .. epochs]
  |> List.fold (fun nnn j -> 
    
    shuffle trainingData

    let miniBatches = [for k in [0 .. miniBatchSize .. (n-1)] ->
                        trainingData.[k..(min (n-1) (k+miniBatchSize-1))]
                        |> Array.toList
                      ]             
    printfn "Running epoch\t%d /\t%d" j epochs
//    printfn "Epoch start Net: %A" nnn
    let newNet = 
      miniBatches
      |> List.fold (fun n mb -> 
      
//      printfn "minibatch start Net: %A" n
      
      let mbn = updateMiniBatch n mb eta
//      printfn "minibatch end Net: %A" mbn
      mbn
      ) nnn
    printfn "Epoch end Net: %A" newNet
    newNet

  ) net


#load "mnist.fsx"




let data =  Mnist.getData()

let trd, vld, tsd = data

let trainingData:TrainingData = 
  trd
  |> Array.map (fun (d, l) -> (DenseMatrix.ofColumnArrays [| Array.map (fun px ->  (float px) / 255.0 ) d |], toMatrix l) )

let net = network [28*28;30;10]
let trainedNet = SGD net trainingData 30 10 3.0 




