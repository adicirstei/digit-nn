type Arguments =
  | Program of string
  | Brain of int list
  | LoadNetwork of string
  | SaveNetwork of string
  | Eta of float
  | Epochs of int
  | MiniBatch of int

type CLIArgument =
  | Index of string
  | Key of key:string * value:string


let parseOne (arg:CLIArgument) : Arguments =
  Program "s"

let (|Brain|_|) (arg:string) =
  if arg.StartsWith("-brain:") then 
    let b = arg.Substring(7).Split([|' '|])
    printfn "%A" b
    b
    |> Array.map int
    |> Array.toList
    |> Some
  else None

let (|Eta|_|) (arg:string) =
  if arg.StartsWith("-eta:") then 
    let e = float (arg.Substring(5))
    Some e
  else None
 
let (|MiniBatch|_|) (arg:string) =
  if arg.StartsWith("-minibatch:") then 
    let i = int (arg.Substring(11))
    Some i
  else None
 
let (|Load|_|) (arg:string) =
  if arg.StartsWith("-load:") then 
    Some (arg.Substring(6))
  else None

let (|Save|_|) (arg:string) =
  if arg.StartsWith("-save:") then 
    Some (arg.Substring(6))
  else None

let (|Epochs|_|) (arg:string) =
  if arg.StartsWith("-epochs:") then 
    let i = int (arg.Substring(8))
    Some i
  else None


let parse (args:string []) : Arguments [] =
  args
  |> Array.mapi (fun i a -> 
      match a with 
        | Brain b -> Brain b
        | MiniBatch mb -> MiniBatch mb
        | Load p -> LoadNetwork p
        | Save p -> SaveNetwork p
        | Eta e -> Eta e
        | Epochs e -> Epochs e
        | _ -> Program a
      )

