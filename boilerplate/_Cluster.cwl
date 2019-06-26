#!/usr/bin/env cwltool

cwlVersion: v1.0

class: CommandLineTool

baseCommand: [kmeansCluster.py]

inputs:
  inFile:
    type: File
    inputBinding:
      position: 1
      prefix: --infile

  outFile:
    type: string
    inputBinding:
      position: 2
      prefix: --outfile

  k:
    type: int
    inputBinding:
      position: 3
      prefix: --k

  seed:
    type: int
    inputBinding:
      position: 4
      prefix: --seed

outputs:
  output:
    type: File
    outputBinding:
      glob: "$(inputs.outFile)"
