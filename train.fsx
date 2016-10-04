#load "network.fsx"

#load "persistence.fsx"
#load "mnist.fsx"

type TrainArgs = {
  MiniBatch : int
  Eta : float
  Brain : int list
  Epochs : int
  SavePath : string option
}

open MathNet.Numerics.LinearAlgebra
printfn "loading MNIST data..."
let data =  Mnist.getData()

let trd, vld, tsd = data

// #load "render.fsx"

// Render.renderImage (fst trd.[30])
// |> printfn "%d\n\n%s" (snd trd.[30]) 

let trainingData:Network.TrainingData = 
  trd
  |> Array.map (fun (d, l) -> (DenseMatrix.ofColumnArrays [| Array.map (fun px ->  (float px) ) d |], Network.toMatrix l) )


let testData:Network.TestData = 
  tsd
  |> Array.map (fun (d, l) -> (DenseMatrix.ofColumnArrays [| Array.map (fun px ->  (float px) ) d |], l) )

#load "args.fsx"

let defaultBrain = [784; 30; 10]

let defaultArgs = { 
  Epochs = 50
  Eta = 1.0
  Brain = defaultBrain
  MiniBatch = 10
  SavePath = None
}

let filePathFromArgs a =
  match a.SavePath with
  | None -> sprintf "nets/trained-%dN-%dE.net" a.Brain.[1] a.Epochs
  | Some p -> p 



let args =
  Args.parse fsi.CommandLineArgs
  |> Array.fold (fun s a -> 
    printfn "%A" a
    match a with
    | Args.Brain b -> { s with Brain = b }
    | Args.Epochs e -> { s with Epochs = e}
    | Args.Eta b -> { s with Eta = b }
    | Args.MiniBatch b -> { s with MiniBatch = b }
    | Args.SaveNetwork path -> { s with SavePath = Some path }      
    | _ -> s
  ) defaultArgs



let net = Network.network args.Brain
printfn "Started training network with args %A" args
let trainedNet = Network.SGD net trainingData args.Epochs args.MiniBatch args.Eta testData
printfn "Saving trained network"
Persistence.saveNet trainedNet (filePathFromArgs args)