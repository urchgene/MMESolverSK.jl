# MME.jl
MME solver for one random component only.

This repo aims to provide a Julia implementation for solving mixed model equations (MME). Julia is a fast programming language for numerical computing thus an MME solver will benefit from Julia. We provide MME solvers based on LU/Cholesky solvers, MUMPs solver and the iterative PCG solver. Target is for large equations. This project is in progress and collaborators are highly welcome. Target is a multi-trait model solver for plant and animal genetic applications and algorithm is based on canonical transformation. MME solvers are NOT variance component estimation engines though they can be coupled. Our solver assumes known variance components!
