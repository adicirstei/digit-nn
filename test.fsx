#load "network.fsx"
#load "persistence.fsx"
#load "mnist.fsx"

open MathNet.Numerics.LinearAlgebra

let renderResult (r:Matrix<float>) : unit =
  [0..9]
  |> List.iter(fun i -> printfn "%d : %f %s" i r.[i,0] (String.replicate (int (r.[i,0] * 100.0)) "*"))


let trainedNet = Persistence.loadNet "nets/trained-30N-30E.net"

let data =  Mnist.getData()

let trd, vld, tsd = data

#load "render.fsx"

Network.shuffle tsd
tsd
|> Array.take 100
|> Array.iter (fun el -> 
  let img = DenseMatrix.ofColumnArrays [| fst el |> Array.map (fun px ->  (float px) )  |]
  Render.renderImage (fst el)
  |> printfn "%d\n\n%s" (snd el) 
  renderResult (Network.feedforward trainedNet (img))
)


// let trainingData:Network.TrainingData = 
//   trd
//   |> Array.map (fun (d, l) -> (DenseMatrix.ofColumnArrays [| Array.map (fun px ->  (float px) ) d |], Network.toMatrix l) )




