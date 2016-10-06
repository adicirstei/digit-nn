let renderImage (img:byte []) : string =
  //let grad = " -·♦■█"
  let grad = " .,:;xX&@"
  let l = float (String.length grad - 1)
  img
  |> Array.map (fun b -> string grad.[int ( (float b / 255.0) * l) ])
  |> Array.chunkBySize 28
  |> Array.map (fun s -> String.concat "" s)
  |> String.concat "\n"