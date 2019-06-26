#!/usr/bin/env cwltool

cwlVersion: v1.0

class: CommandLineTool

baseCommand: [kmeansCluster.py]

inputs:
  InputTSV:
    type: File
    inputBinding:
      position: 1
      prefix: --infile
  K:
    type: int
    inputBinding:
      position: 3
      prefix: --k
  Seed:
    type: int
    inputBinding:
      position: 4
      prefix: --seed

arguments: [
  "--outfile",
  $(inputs.InputTSV.nameroot).cluster
  ]

outputs:
  OutputFile:
    type: File
    outputBinding:
      glob: $(inputs.InputTSV.nameroot).cluster