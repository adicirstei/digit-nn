#load "network.fsx"
#load "persistence.fsx"
#load "mnist.fsx"

open MathNet.Numerics.LinearAlgebra

type TestArgs = {
  LoadPath : string option
}



let renderResult (r:Matrix<float>) : unit =
  [0..9]
  |> List.iter(fun i -> printfn "%d : %f %s" i r.[i,0] (String.replicate (int (r.[i,0] * 100.0)) "*"))

#load "args.fsx"

let defaultArgs = { LoadPath = None }

let args =
  Args.parse fsi.CommandLineArgs
  |> Array.fold (fun s a -> 
    match a with
    | Args.LoadNetwork path -> { s with LoadPath = Some path }      
    | _ -> s
  ) defaultArgs

let trainedNet = 
  match args.LoadPath with
  | Some x -> 
    printfn "Loading %s network" x
    Persistence.loadNet x
  | None -> failwith "no neural network path speciffied\nPlease add a -load:path/to/file option"

printfn "Loading MNIST data"
let data =  Mnist.getData()

let trd, vld, tsd = data
printfn "Running tests"
#load "render.fsx"

Network.shuffle tsd
tsd
|> Array.take 10
|> Array.iter (fun el -> 
  let img = DenseMatrix.ofColumnArrays [| fst el |> Array.map (fun px ->  (float px) )  |]
  Render.renderImage (fst el)
  |> printfn "%d\n\n%s" (snd el) 
  renderResult (Network.feedforward trainedNet (img))
)

