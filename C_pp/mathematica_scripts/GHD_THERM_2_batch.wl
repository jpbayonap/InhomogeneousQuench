(* ::Package:: *)

ClearAll["Global`*"];
$HistoryLength = 0;

parseBoolEnv[name_String, default_] := Module[{s = ToLowerCase @ StringTrim @ Environment[name]},
  Switch[s,
    "", default,
    "1" | "true" | "yes" | "on", True,
    "0" | "false" | "no" | "off", False,
    _, Print["ERROR: invalid ", name, " = ", s]; Quit[1]
  ]
];

parsePositiveIntEnv[name_String, default_Integer?Positive] := Module[{s = StringTrim @ Environment[name], parsed},
  If[StringLength[s] == 0, Return[default]];
  parsed = Quiet @ Check[ToExpression[s], $Failed];
  If[IntegerQ[parsed] && parsed > 0,
    parsed,
    Print["ERROR: invalid ", name, " = ", s, " (expected positive integer)"];
    Quit[1]
  ]
];

parseNonNegativeIntEnv[name_String, default_Integer?NonNegative] := Module[{s = StringTrim @ Environment[name], parsed},
  If[StringLength[s] == 0, Return[default]];
  parsed = Quiet @ Check[ToExpression[s], $Failed];
  If[IntegerQ[parsed] && parsed >= 0,
    parsed,
    Print["ERROR: invalid ", name, " = ", s, " (expected non-negative integer)"];
    Quit[1]
  ]
];

parseNumericEnv[name_String, default_?NumericQ] := Module[{s = StringTrim @ Environment[name], parsed},
  If[StringLength[s] == 0, Return[N[default]]];
  parsed = Quiet @ Check[ToExpression[s], $Failed];
  If[NumericQ[parsed],
    N[parsed],
    Print["ERROR: invalid ", name, " = ", s, " (expected numeric value)"];
    Quit[1]
  ]
];

M = 350;
Jhop = 1.0;
gamma = parseNumericEnv["MMA_GAMMA", 1.0];
beta = parseNumericEnv["MMA_BETA", 1.0];
wp = 80;
kVals = Subdivide[-Pi, Pi, 800];

Print["gamma = ", gamma];
Print["beta = ", beta];

deltaRFull = {
   1.589174270629882812*^-1, 0.0,
  -1.330617368221282959*^-1, 0.0,
  -1.641092076897621155*^-2, 0.0,
  -5.921255331486463547*^-3, 0.0,
  -1.632967847399413586*^-3, 0.0,
  -7.708735647611320019*^-4, 0.0,
  -4.044898378197103739*^-4, 0.0,
  -2.277035964652895927*^-4, 0.0,
  -1.356566790491342545*^-4, 0.0,
  -8.435355994151905179*^-5, 0.0,
  -5.324042285792529583*^-5, 0.0,
  -3.394593659322708845*^-5, 0.0,
  -2.000243694055825472*^-5, 0.0,
  -9.657909686211496592*^-6, 0.0,
  -1.980433808057568967*^-6, 0.0,
   4.740451913676224649*^-6
};

deltaRTable = Table[Take[deltaRFull, n], {n, 1, Length[deltaRFull], 2}];
deltaTagFrom[delta_List] := "delta" <> ToString[Length[delta]];

chiOnly = parseBoolEnv["MMA_CHI_ONLY", False];
skipChiPlus = parseBoolEnv["MMA_SKIP_CHI", False];
If[chiOnly && skipChiPlus,
  Print["MMA_CHI_ONLY=true overrides MMA_SKIP_CHI; ChiPlus exports will run."];
  skipChiPlus = False;
];

deltaBatchSize = parsePositiveIntEnv["MMA_DELTA_BATCH_SIZE", 4];
deltaBatchIndex = parseNonNegativeIntEnv["MMA_DELTA_BATCH_INDEX", 0];
numDeltaBatches = Ceiling[Length[deltaRTable]/deltaBatchSize];

selectedDeltaTable = If[
  chiOnly,
  deltaRTable,
  Module[{start, stop},
    If[deltaBatchIndex >= numDeltaBatches,
      Print[
        "ERROR: MMA_DELTA_BATCH_INDEX = ", deltaBatchIndex,
        " is out of range for ", numDeltaBatches,
        " batches of size ", deltaBatchSize
      ];
      Quit[1]
    ];
    start = 1 + deltaBatchSize*deltaBatchIndex;
    stop = Min[Length[deltaRTable], start + deltaBatchSize - 1];
    deltaRTable[[start ;; stop]]
  ]
];

Print["total delta truncations = ", Length[deltaRTable]];
Print["deltaBatchSize = ", deltaBatchSize];
Print["deltaBatchIndex = ", deltaBatchIndex];
Print["numDeltaBatches = ", numDeltaBatches];
Print["chiOnly = ", chiOnly];
Print["skipChiPlus = ", skipChiPlus];
Print["selected delta tags = ", deltaTagFrom /@ selectedDeltaTable];

profileListOdd = Table[{r, "+"}, {r, {1, 3, 5}}];
profileListEven = Table[{r, "-"}, {r, {2, 4, 6}}];

profileMode = Module[{s = ToLowerCase @ StringTrim @ Environment["MMA_PROFILE_MODE"]},
  Switch[s,
    "", "all",
    "odd" | "even" | "all", s,
    _, Print["ERROR: invalid MMA_PROFILE_MODE = ", s]; Quit[1]
  ]
];

profileList = Switch[
  profileMode,
  "odd", profileListOdd,
  "even", profileListEven,
  "all", Join[profileListOdd, profileListEven]
];

Print["profileMode = ", profileMode];
Print["profileList = ", profileList];

zetasVal = Join[
   Subdivide[-2.5, -1., 40],
   Rest @ Subdivide[-1., 1., 160],
   Rest @ Subdivide[1., 2.5, 40]
];

runDiagnostics = parseBoolEnv["MMA_RUN_DIAGNOSTICS", False];
Print["runDiagnostics = ", runDiagnostics];
rDiag = 1;
signDiag = "+";
zDiag = -1.0;

scriptDir = DirectoryName[$InputFileName];
repoBase = DirectoryName[scriptDir];
hostRepoBase = Module[{s = StringTrim @ Environment["MMA_HOST_REPO_BASE"]},
  If[StringLength[s] == 0, Missing["NotAvailable"], s]
];
outBase = FileNameJoin[{repoBase, "GHD_BETA_HYBRID"}];
hostOutBase = If[MissingQ[hostRepoBase], Missing["NotAvailable"], FileNameJoin[{hostRepoBase, "GHD_BETA_HYBRID"}]];

chiCsvDir = FileNameJoin[{outBase, "chi_plus", "csv"}];
chiPngDir = FileNameJoin[{outBase, "chi_plus", "png"}];
csvDir = FileNameJoin[{outBase, "csv"}];
pngDir = FileNameJoin[{outBase, "png"}];

Scan[
  If[! DirectoryQ[#], CreateDirectory[#, CreateIntermediateDirectories -> True]] &,
  {chiCsvDir, chiPngDir, csvDir, pngDir}
];

Print["scriptDir = ", scriptDir];
Print["repoBase = ", repoBase];
Print["outBase = ", outBase];
If[! MissingQ[hostRepoBase], Print["hostRepoBase = ", hostRepoBase]];
If[! MissingQ[hostOutBase], Print["hostOutBase = ", hostOutBase]];

numKernels = parsePositiveIntEnv["MMA_NUM_KERNELS", Max[1, $ProcessorCount]];
LaunchKernels[numKernels];
Print["Requested kernels: ", numKernels];
Print["Launched kernels: ", Length[Kernels[]]];

gammaTagFrom[g_?NumericQ] := StringReplace[
   ToString @ NumberForm[N[g], {Infinity, 2}, NumberPadding -> {"", "0"}],
   " " -> ""
];

betaTagFrom[b_?NumericQ] := StringReplace[
   ToString @ NumberForm[N[b], {Infinity, 1}, NumberPadding -> {"", "0"}],
   " " -> ""
];

finiteRealQ[x_] := NumericQ[x] && FreeQ[N[x], _DirectedInfinity | Indeterminate | ComplexInfinity] && TrueQ[Im[N[x]] == 0];

eps = SetPrecision[10^-4, wp];
ThetaN[x_?NumericQ] := 1/2 (1 + Tanh[x/eps]);

gMatElem[r_Integer?Positive, n_Integer?Positive, J_?NumericQ] := Module[{den, num, pf},
  If[r === n, Return[N[8 J^2 (2 n^2 - 1)/(4 n^2 - 1), wp]]];
  pf = 1 + (-1)^(r + n);
  If[pf === 0, Return[0]];
  num = (n^2 + r^2 - 1)*pf;
  den = n^4 - 2 n^2 (r^2 + 1) + (r^2 - 1)^2;
  N[-4 J^2 num/den, wp]
];

ClearAll[SolveThermalSystemDirectTherm];
SolveThermalSystemDirectTherm[mVal_, betaVal_, gammaVal_, jVal_, delta_, nMax_: 4000, wpVal_: 60] :=
Module[{rVals, gMat, A, rhs, a, DeltaPad, res},
  If[!IntegerQ[mVal] || mVal <= 0,
    Print["ERROR: mVal must be a positive integer, got ", InputForm[mVal]];
    Return[$Failed]
  ];
  If[!NumericQ[betaVal] || !NumericQ[gammaVal] || !NumericQ[jVal] || !NumericQ[wpVal],
    Print["ERROR: betaVal, gammaVal, jVal, wpVal must be numeric."];
    Return[$Failed]
  ];
  If[!IntegerQ[nMax] || nMax <= 0,
    Print["ERROR: nMax must be a positive integer, got ", InputForm[nMax]];
    Return[$Failed]
  ];
  If[Head[delta] =!= List,
    Print["ERROR: delta must be a list, got ", Head[delta]];
    Return[$Failed]
  ];

  rVals = Range[mVal];
  gMat = Table[gMatElem[r, n, jVal], {r, 1, mVal}, {n, 1, mVal}];
  DeltaPad = PadRight[N@delta, mVal, 0.0];
  A = gMat;
  rhs = -0.5*gammaVal*DeltaPad;

  Print["Dimensions[gMat] = ", Dimensions[gMat]];
  Print["Length[DeltaPad] = ", Length[DeltaPad]];
  Print["Dimensions[A] = ", Dimensions[A]];
  Print["Length[rhs] = ", Length[rhs]];

  a = LinearSolve[N[A, wpVal], N[rhs, wpVal], Method -> "Direct"];
  res = Norm[A . a - rhs]/Max[Norm[rhs], 10^-50];
  Print["relative residual = ", N[res, 20]];
  {rVals, N[a, 30]}
];

rhoThermal[k_?NumericQ, b_?NumericQ, J_?NumericQ] := 1/(2 Pi (Exp[-2 b J Cos[k]] + 1));
betaL = beta;
betaR = beta;
nL[k_?NumericQ] := rhoThermal[k, betaL, Jhop];
nR[k_?NumericQ] := rhoThermal[k, betaR, Jhop];

ChiPlus[k_?NumericQ, rVals_List, aVals_List] := rhoThermal[k, beta, Jhop] + (aVals . Cos[rVals k]);

nZetaPos[k_?NumericQ, z_?NumericQ, rVals_List, aVals_List] := Module[{epsp, chi},
  epsp = 2 Jhop Sin[k];
  chi = rhoThermal[k, beta, Jhop] + (aVals . Cos[rVals k]);
  ThetaN[-z]*nL[k] + ThetaN[z]*ThetaN[epsp - z]*chi + ThetaN[z - epsp]*nR[k]
];

nZetaNeg[k_?NumericQ, z_?NumericQ, rVals_List, aVals_List] := Module[{epsp, chi},
  epsp = 2 Jhop Sin[k];
  chi = rhoThermal[k, beta, Jhop] + (aVals . Cos[rVals k]);
  ThetaN[z]*nR[k] + ThetaN[-z]*ThetaN[-epsp + z]*chi + ThetaN[epsp - z]*nL[k]
];

qPlus[k_, r_?NumericQ] := 2 Cos[r k];
qMinus[k_, r_?NumericQ] := -2 Sin[r k];

HydCharge[r_?NumericQ, z_?NumericQ, sign_ : "+", rVals_List, aVals_List] := Module[{fpos, fneg},
  fpos[k_?NumericQ] := If[sign === "-", qMinus[k, r], qPlus[k, r]]*nZetaPos[k, z, rVals, aVals];
  fneg[k_?NumericQ] := If[sign === "-", qMinus[k, r], qPlus[k, r]]*nZetaNeg[k, z, rVals, aVals];
  NIntegrate[fpos[k], {k, 0, Pi}, Method -> "GlobalAdaptive", WorkingPrecision -> wp, AccuracyGoal -> 12, PrecisionGoal -> 12, MinRecursion -> 4, MaxRecursion -> 60] +
  NIntegrate[fneg[k], {k, -Pi, 0}, Method -> "GlobalAdaptive", WorkingPrecision -> wp, AccuracyGoal -> 12, PrecisionGoal -> 12, MinRecursion -> 4, MaxRecursion -> 60]
];

HydCurrent[r_?NumericQ, z_?NumericQ, sign_ : "+", rVals_List, aVals_List] := Module[{fpos, fneg},
  fpos[k_?NumericQ] := (2 Jhop Sin[k])*If[sign === "-", qMinus[k, r], qPlus[k, r]]*nZetaPos[k, z, rVals, aVals];
  fneg[k_?NumericQ] := (2 Jhop Sin[k])*If[sign === "-", qMinus[k, r], qPlus[k, r]]*nZetaNeg[k, z, rVals, aVals];
  NIntegrate[fpos[k], {k, 0, Pi}, Method -> "GlobalAdaptive", WorkingPrecision -> wp, AccuracyGoal -> 12, PrecisionGoal -> 12, MinRecursion -> 4, MaxRecursion -> 60] +
  NIntegrate[fneg[k], {k, -Pi, 0}, Method -> "GlobalAdaptive", WorkingPrecision -> wp, AccuracyGoal -> 12, PrecisionGoal -> 12, MinRecursion -> 4, MaxRecursion -> 60]
];

makeSolutionAssoc[delta_List] := Module[{sol, deltaTagLocal},
  deltaTagLocal = deltaTagFrom[delta];
  Print["Solving system for ", deltaTagLocal];
  sol = SolveThermalSystemDirectTherm[M, beta, gamma, Jhop, delta, 4000, wp];
  If[sol === $Failed, Return[$Failed]];
  <|"delta" -> delta, "deltaTag" -> deltaTagLocal, "rVals" -> sol[[1]], "aVals" -> sol[[2]]|>
];

exportChiPlus[solAssoc_Association] := Module[
  {deltaTagLocal, rValsLocal, aValsLocal, chiVals, badChi, dataVals, plot, betaTagLocal, gammaTagLocal, fileST, csvOut, pngOut},
  deltaTagLocal = solAssoc["deltaTag"];
  rValsLocal = solAssoc["rVals"];
  aValsLocal = solAssoc["aVals"];
  betaTagLocal = betaTagFrom[beta];
  gammaTagLocal = gammaTagFrom[gamma];

  Print["Running ChiPlus for beta=", beta, " ", deltaTagLocal];
  chiVals = N[ChiPlus[#, rValsLocal, aValsLocal] & /@ kVals, 16];
  badChi = Flatten @ Position[chiVals, _?(Not @* finiteRealQ), {1}, Heads -> False];
  If[badChi =!= {},
    Print["ERROR: non-numeric ChiPlus values found. badChi=", badChi];
    Return[$Failed]
  ];

  dataVals = N @ Transpose[{kVals, chiVals}];
  plot = ListLinePlot[
    {Transpose[{kVals, chiVals}]},
    PlotLegends -> {"ChiPlus"},
    Frame -> True,
    FrameLabel -> {"k", "\[Chi]Plus"},
    PlotLabel -> Row[{"THERM beta, ", deltaTagLocal, ", beta=", betaTagLocal, ", M=", M, ", gamma=", gammaTagLocal, ", wp=", wp}],
    ImageSize -> 700
  ];

  fileST = StringTemplate["CHI_PLUS_BETA_``_beta``_M``_gamma``_wp``"][deltaTagLocal, betaTagLocal, M, gammaTagLocal, wp];
  csvOut = FileNameJoin[{chiCsvDir, fileST <> ".csv"}];
  pngOut = FileNameJoin[{chiPngDir, fileST <> ".png"}];
  Export[csvOut, Prepend[dataVals, {"k", "ChiPlus"}], "CSV"];
  Export[pngOut, plot, "PNG"];
  Print["Saved ChiPlus CSV: ", csvOut];
  Print["Saved ChiPlus PNG: ", pngOut];
  <|"status" -> "ok", "csv" -> csvOut, "png" -> pngOut|>
];

exportOneProfile[solAssoc_Association, rs_List] := Module[
  {deltaTagLocal, rValsLocal, aValsLocal, gammaTagLocal, betaTagLocal, rval, signval, qVals, jVals, badQ, badJ, dataVals, plot, fileST, csvOut, pngOut},
  deltaTagLocal = solAssoc["deltaTag"];
  rValsLocal = solAssoc["rVals"];
  aValsLocal = solAssoc["aVals"];
  gammaTagLocal = gammaTagFrom[gamma];
  betaTagLocal = betaTagFrom[beta];
  {rval, signval} = rs;

  Print["Running ", deltaTagLocal, " r=", rval, " sign=", signval];
  qVals = HydCharge[rval, #, signval, rValsLocal, aValsLocal] & /@ zetasVal;
  jVals = HydCurrent[rval, #, signval, rValsLocal, aValsLocal] & /@ zetasVal;

  badQ = Flatten @ Position[qVals, _?(Not @* finiteRealQ), {1}, Heads -> False];
  badJ = Flatten @ Position[jVals, _?(Not @* finiteRealQ), {1}, Heads -> False];
  If[badQ =!= {} || badJ =!= {},
    Print["ERROR: non-numeric values found. badQ=", badQ, " badJ=", badJ];
    Return[$Failed]
  ];

  dataVals = N @ Transpose[{zetasVal, qVals, jVals}];
  plot = ListLinePlot[
    {Transpose[{zetasVal, qVals}], Transpose[{zetasVal, jVals}]},
    PlotLegends -> {"q", "J"},
    Frame -> True,
    FrameLabel -> {"zeta", "value"},
    PlotLabel -> Row[{"THERM beta, ", deltaTagLocal, ", r=", rval, ", sign=", signval, ", beta=", betaTagLocal, ", gamma=", gammaTagLocal, ", M=", M, ", wp=", wp}],
    ImageSize -> 700
  ];

  fileST = StringTemplate["GHD_THERM2_Beta_``_r``_sign``_beta``_gamma``_M``_wp``_seggrid"][deltaTagLocal, rval, signval, betaTagLocal, gammaTagLocal, M, wp];
  csvOut = FileNameJoin[{csvDir, fileST <> ".csv"}];
  pngOut = FileNameJoin[{pngDir, fileST <> ".png"}];
  Export[csvOut, Prepend[dataVals, {"zeta", "q", "J"}], "CSV"];
  Export[pngOut, plot, "PNG"];
  Print["Saved CSV: ", csvOut];
  Print["Saved PNG: ", pngOut];
  <|"status" -> "ok", "csv" -> csvOut, "png" -> pngOut|>
];

main[] := Module[{solutions, chiResults, profileTasks, profileResults, q0Diag, j0Diag, lastSol},
  DistributeDefinitions[
    finiteRealQ, ThetaN, rhoThermal, nL, nR, ChiPlus, nZetaPos, nZetaNeg,
    qPlus, qMinus, HydCharge, HydCurrent,
    SolveThermalSystemDirectTherm, gMatElem,
    M, Jhop, gamma, beta, wp, kVals, zetasVal,
    chiCsvDir, chiPngDir, csvDir, pngDir,
    deltaTagFrom, gammaTagFrom, betaTagFrom,
    makeSolutionAssoc, exportChiPlus, exportOneProfile
  ];

  solutions = DeleteCases[
    ParallelMap[makeSolutionAssoc, selectedDeltaTable, Method -> "CoarsestGrained"],
    $Failed
  ];
  Print["Prepared ", Length[solutions], " delta solutions."];

  If[skipChiPlus,
    chiResults = {};
    Print["Skipping ChiPlus exports due to MMA_SKIP_CHI=true."],
    chiResults = DeleteCases[
      ParallelMap[exportChiPlus, solutions, Method -> "CoarsestGrained"],
      $Failed
    ];
    Print["ChiPlus exports completed for ", Length[chiResults], " deltas."]
  ];

  If[runDiagnostics && Length[solutions] > 0,
    lastSol = Last[solutions];
    q0Diag = HydCharge[rDiag, zDiag, signDiag, lastSol["rVals"], lastSol["aVals"]];
    j0Diag = HydCurrent[rDiag, zDiag, signDiag, lastSol["rVals"], lastSol["aVals"]];
    Print["Diagnostic q = ", q0Diag];
    Print["Diagnostic J = ", j0Diag];
  ];

  If[chiOnly,
    Print["Skipping profile exports due to MMA_CHI_ONLY=true."]; Return[]
  ];

  profileTasks = Flatten[Table[{solAssoc, rs}, {solAssoc, solutions}, {rs, profileList}], 1];
  Print["Prepared ", Length[profileTasks], " profile tasks."];
  profileResults = DeleteCases[
    ParallelMap[exportOneProfile @@ # &, profileTasks, Method -> "CoarsestGrained"],
    $Failed
  ];
  Print["Profile exports completed for ", Length[profileResults], " tasks."];
];

main[];
CloseKernels[];
Quit[];
