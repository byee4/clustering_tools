#!/usr/bin/env cwltool

cwlVersion: v1.0

class: CommandLineTool

baseCommand: [louvainCluster.py]

inputs:
  InputTSV:
    type: File
    inputBinding:
      position: 1
      prefix: --infile

outputs:
  output_file:
    type: File[]
    outputBinding:
      glob: "*"
