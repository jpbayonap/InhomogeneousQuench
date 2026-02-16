#!/usr/bin/env wolfrun
(* Batch driver to run the Mathematica GHD analytics notebook and export a CSV *)

(* Usage: wolfrun -script ghd_analytics.wls M r sign gamma [outDir] [zMin] [zMax] [nZ] *)

Module[
  {
    args = Rest[$ScriptCommandLine],
    M = 200,
    r = 1,
    sign = "+",
    gamma = 0.1,
    outDir = ".",
    zMin = -3.,
    zMax = 3.,
    nZ = 300,
    root, nbPath
  },

  If[Length[args] >= 1, M = ToExpression[args[[1]]]];
  If[Length[args] >= 2, r = ToExpression[args[[2]]]];
  If[Length[args] >= 3, sign = args[[3]]];
  If[Length[args] >= 4, gamma = ToExpression[args[[4]]]];
  If[Length[args] >= 5, outDir = args[[5]]];
  If[Length[args] >= 6, zMin = ToExpression[args[[6]]]];
  If[Length[args] >= 7, zMax = ToExpression[args[[7]]]];
  If[Length[args] >= 8, nZ = ToExpression[args[[8]]]];

  root = DirectoryName[$InputFileName];
  nbPath = FileNameJoin[{root, "GHD_analytics.nb"}];

  Print["Loading notebook definitions from ", nbPath, " ..."];
  Get[nbPath];

  Print["Running ComputeGHDProfiles with M=", M, " r=", r, " sign=", sign, " gamma=", gamma];
  res = ComputeGHDProfiles[M, 1.0, gamma, r, sign, zMin, zMax, nZ, outDir, "test"];

  Print["Finished. Saved data to: ", FileNameJoin[{outDir, StringTemplate["GHD_r``_sign``_M``_gamma````"][r, sign, M, NumberForm[gamma, {4, 2}], "test"]}] <> ".csv"];
]

