let private rand = new System.Random()

let private swap (a: _[]) x y =
    let tmp = a.[x]
    a.[x] <- a.[y]
    a.[y] <- tmp

// shuffle an array (in-place)
let inPlaceShuffle a =
    Array.iteri (fun i _ -> swap a i (rand.Next(i, Array.length a))) a
