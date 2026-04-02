(* ::Package:: *)

ClearAll["Global`*"];
$HistoryLength = 0;

(* =====================USER SETTINGS===================== *)
state = Module[{s = ToLowerCase @ StringTrim @ Environment["MMA_STATE"]},
  Switch[s,
    "", "neel",
    "neel" | "polarized", s,
    _, Print["ERROR: invalid MMA_STATE = ", s]; Quit[1]
  ]
];

M = 350;
Jhop = 1.0;
gamma = Module[{s = Environment["MMA_GAMMA"], parsed},
  If[StringLength[s] == 0,
    1.0,
    parsed = Quiet @ Check[ToExpression[s], $Failed];
    If[NumericQ[parsed],
      N[parsed],
      Print["ERROR: invalid MMA_GAMMA = ", s];
      Quit[1]
    ]
  ]
];

Print["gamma = ", gamma];

wp = 80;
kVals = Subdivide[-Pi, Pi, 800];

deltaRNeel = {
   1.102685928344700000*^-6, 0.0, 8.457381278276443480*^-3, 0.0,
   1.617664564400911331*^-2, 0.0, 1.487777056172490120*^-2, 0.0,
   1.120693911798298359*^-2, 0.0, 8.220242802053689957*^-3
};

deltaRPol = {
   2.264976501464800000*^-6, 0.0, 1.691447943449020380*^-2, 0.0,
   3.235286474227905273*^-2, 0.0, 2.975438069552183152*^-2, 0.0,
   2.241201791912317276*^-2
};

deltaR = Switch[
  state,
  "neel", deltaRNeel,
  "polarized", deltaRPol,
  _, Print["ERROR: unknown state = ", state]; Quit[1]
];

deltaRTable = Table[Take[deltaR, n], {n, 1, Length[deltaR], 2}];
deltaRTableBatch = Take[deltaRTable, UpTo[3]];
profileDeltaTable = deltaRTableBatch;

profileListOdd = Table[{r, "-"}, {r, {1, 3, 5}}];
profileListEven = Table[{r, "+"}, {r, {0, 2, 4}}];

profileMode = Module[{s = ToLowerCase @ StringTrim @ Environment["MMA_PROFILE_MODE"]},
  Switch[s,
    "", "odd",
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
Print["delta tags = ", ("delta" <> ToString[Length[#]]) & /@ profileDeltaTable];

zetasVal = Join[
   Subdivide[-2.5, -1., 40],
   Rest @ Subdivide[-1., 1., 160],
   Rest @ Subdivide[1., 2.5, 40]
];


runDiagnostics = Module[{s = ToLowerCase @ StringTrim @ Environment["MMA_RUN_DIAGNOSTICS"]},
  Switch[s,
    "", False,
    "1" | "true" | "yes" | "on", True,
    "0" | "false" | "no" | "off", False,
    _, Print["ERROR: invalid MMA_RUN_DIAGNOSTICS = ", s]; Quit[1]
  ]
];

Print["runDiagnostics = ", runDiagnostics];

skipChiPlus = Module[{s = ToLowerCase @ StringTrim @ Environment["MMA_SKIP_CHI"]},
  Switch[s,
    "" | "auto", state === "polarized",
    "1" | "true" | "yes" | "on", True,
    "0" | "false" | "no" | "off", False,
    _, Print["ERROR: invalid MMA_SKIP_CHI = ", s]; Quit[1]
  ]
];

Print["skipChiPlus = ", skipChiPlus];

rDiag = 3;
signDiag = "-";
zDiag = -0.1;
(* ======================================================= *)

scriptDir = DirectoryName[$InputFileName];
repoBase = DirectoryName[scriptDir];
hostRepoBase = Module[{s = StringTrim @ Environment["MMA_HOST_REPO_BASE"]},
  If[StringLength[s] == 0, Missing["NotAvailable"], s]
];
stateDir = Switch[
  state,
  "neel", "GHD_NEEL_HYBRID",
  "polarized", "GHD_POLARIZED_HYBRID",
  _, "GHD_UNKNOWN_HYBRID"
];
outBase = FileNameJoin[{repoBase, stateDir}];
hostOutBase = If[MissingQ[hostRepoBase], Missing["NotAvailable"], FileNameJoin[{hostRepoBase, stateDir}]];

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

(* Set number of Kernels *)
numKernels = Module[{s = Environment["MMA_NUM_KERNELS"]},
  If[StringLength[s] > 0, ToExpression[s], $ProcessorCount]
];

LaunchKernels[numKernels];
Print["Requested kernels: ", numKernels];
Print["Launched kernels: ", Length[Kernels[]]];

deltaTagFrom[delta_List] := "delta" <> ToString[Length[delta]];
stateTagFrom[s_String] := Switch[s, "neel", "NEEL", "polarized", "POLARIZED", _, "UNKNOWN"];
gammaTagFrom[g_?NumericQ] := StringReplace[
   ToString @ NumberForm[N[g], {Infinity, 2}, NumberPadding -> {"", "0"}],
   " " -> ""
];

(* TODO: Copy the uncommented core physics definitions from GHD_hybrid_test.wl:
   - finiteRealQ                          (source around line 86)
   - SolveTruncatedSystemDirectEff        (source around lines 88-117)
   - stepTheta, nBase, nL, nR, ChiPlus,
     nZeta, chargeIntegrand, currentIntegrand,
     qAt, jAt                             (source around lines 119-155)
*)


finiteRealQ[x_]:=NumericQ[x]&&FreeQ[N[x],_DirectedInfinity|Indeterminate|ComplexInfinity]&&TrueQ[Im[N[x]]==0];

ClearAll[SolveTruncatedSystemDirectEff];
SolveTruncatedSystemDirectEff[M_Integer?Positive,J_?NumericQ,gamma_?NumericQ, delta_List,state_String,wp_:60]:=
Module[
{rVals=Range[M],fMat,A,rhs,b, q0, DeltaPad, Awp,rhswp,res,srcIdx},

fMat=Table[If[n==r,(32.0 J^2 r^2)/(1-4 r^2),If[EvenQ[n+r],(32.0 J^2 n r)/(n^4-2 n^2 (r^2+1)+(r^2-1)^2),0.0]],{n,1,M},{r,1,M}];

(*BC convention:DeltaJ=-2 gamma q0*)
A=Table[If[r==n,gamma*r*Pi*J,0.0]-fMat[[r,n]],{r,1,M},{n,1,M}];

(* q_0 *)
(*q0 = Table[ (1-(-1)^r)/(2.0 Pi ),{r,1,M}];*)
q0 =Switch[
state,
"neel", Table[ (1-(-1)^r)/(2.0 Pi ),{r,1,M}],
"polarized", Table[ (1-(-1)^r)/( Pi ),{r,1,M}],
_, (Print["ERROR: unknown state = ", state]; Return[$Failed])
];
(* delta_r *)
DeltaPad= PadRight[N@delta,M,0.0];

(* numerics *)
rhs= gamma J q0 - DeltaPad;
Print["Dimensions[A] = ",Dimensions[A],", Length[rhs] = ",Length[rhs],", wp = ",wp];
Awp=SetPrecision[A,wp];
rhswp=SetPrecision[rhs,wp];
b=LinearSolve[Awp,rhswp];
res=Norm[Awp . b-rhswp]/Max[Norm[rhswp],10^-50];
Print["relative residual = ",N[res,20]];
{rVals,N[b,30]}];

ClearAll[stepTheta,nBase,nL,nR,ChiPlus,nZeta,qPlus,qMinus];
stepTheta[x_?NumericQ]:=Which[x>0,1.,x<0,0.,True,0.5];

nBase[k_?NumericQ,r_Integer?NonNegative,J_?NumericQ,mode_String]:=
Switch[mode,"-", -2.0J  Sin[ r k],"+",2.0J Cos[r k] ];

nL[k_?NumericQ]:=0 ;
(*nR[k_?NumericQ]:= 1/(4  Pi);*)
nR[k_?NumericQ]:=Switch[
state,
"neel",1/(4 Pi),
"polarized",1/(2 Pi),
_,Indeterminate
];


ChiPlus[k_?NumericQ,rVals_List,bVals_List]:=stepTheta[k] (nL[k]+bVals . Sin[rVals k])+stepTheta[-k] (nR[k]+bVals . Sin[rVals k]);

nZeta[k_?NumericQ,zeta_?NumericQ,rVals_List,bVals_List]:=
Module[{epsp=2. Sin[k],chi},chi=ChiPlus[k,rVals,bVals];
If[k>0,stepTheta[-zeta] nL[k]+stepTheta[zeta] stepTheta[epsp-zeta] chi+stepTheta[zeta-epsp] nR[k],stepTheta[zeta] nR[k]+stepTheta[-zeta] stepTheta[-epsp+zeta] chi+stepTheta[epsp-zeta] nL[k]]];


ClearAll[chargeIntegrand,currentIntegrand];
chargeIntegrand[k_?NumericQ,r_Integer?NonNegative, J_?NumericQ,zeta_?NumericQ,sign_String,rVals_List,bVals_List]:=
If[sign==="-",nBase[k,r, J, "-"],nBase[k,r, J, "+"]]*nZeta[k,zeta,rVals,bVals];

currentIntegrand[k_?NumericQ,r_Integer?NonNegative,J_?NumericQ,zeta_?NumericQ,sign_String,rVals_List,bVals_List]:=
2.0 Sin[k]*If[sign==="-",nBase[k,r, J, "-"],nBase[k,r, J, "+"]]*nZeta[k,zeta,rVals,bVals];

ClearAll[qAt,jAt];

qAt[z_?NumericQ,rval_Integer?NonNegative, J_?NumericQ,signval_String,rVals_List,bVals_List]:=Module[{res},res=Check[Quiet[NIntegrate[Evaluate@chargeIntegrand[k,rval,J, z,signval,rVals,bVals],{k,-Pi,Pi},Method->{"GlobalAdaptive","SymbolicProcessing"->0},MinRecursion->3,MaxRecursion->20,WorkingPrecision->25,AccuracyGoal->8,PrecisionGoal->10],{NIntegrate::slwcon,NIntegrate::ncvb,NIntegrate::eincr}],$Failed,{NIntegrate::inumr}];
If[res===$Failed||!finiteRealQ[res],Indeterminate,N[res,16]]];

jAt[z_?NumericQ,rval_Integer?NonNegative, J_?NumericQ,signval_String,rVals_List,bVals_List]:=Module[{res},res=Check[Quiet[NIntegrate[Evaluate@currentIntegrand[k,rval,J, z,signval,rVals,bVals],{k,-Pi,Pi},Method->{"GlobalAdaptive","SymbolicProcessing"->0},MinRecursion->3,MaxRecursion->20,WorkingPrecision->25,AccuracyGoal->8,PrecisionGoal->10],{NIntegrate::slwcon,NIntegrate::ncvb,NIntegrate::eincr}],$Failed,{NIntegrate::inumr}];
If[res===$Failed||!finiteRealQ[res],Indeterminate,N[res,16]]];

(*---- Make SOLUTION ----*)
makeSolutionAssoc[delta_List] := Module[
  {sol, deltaTagLocal},
  deltaTagLocal = deltaTagFrom[delta];
  Print["Solving system for ", deltaTagLocal];
  sol = SolveTruncatedSystemDirectEff[M, Jhop, gamma, delta, state, wp];
  If[sol === $Failed, Return[$Failed]];
  <|
    "delta" -> delta,
    "deltaTag" -> deltaTagLocal,
    "rVals" -> sol[[1]],
    "bVals" -> sol[[2]]
  |>
];

exportChiPlus[solAssoc_Association] := Module[
  {
    deltaTagLocal, rValsLocal, bValsLocal, chiVals, badChi, dataVals, plot,
    stateTagLocal, gammaTagLocal, fileST, csvOut, pngOut
  },
  deltaTagLocal = solAssoc["deltaTag"];
  rValsLocal = solAssoc["rVals"];
  bValsLocal = solAssoc["bVals"];
  stateTagLocal = stateTagFrom[state];
  gammaTagLocal = gammaTagFrom[gamma];

  (* TODO: Copy the body of the ChiPlus export block from GHD_hybrid_test.wl
     lines 182-245, but DO NOT solve again.
     Use rValsLocal/bValsLocal from solAssoc instead of solLocal.
     Replace any Continue[] with Return[$Failed] or a small failure association.
  *)

	Print["Running ChiPlus for state=",state," ",deltaTagLocal];
	
	
	chiVals=N[ChiPlus[#,rValsLocal,bValsLocal]&/@kVals,16];
	
	badChi=Flatten@Position[chiVals,_?(Not@*finiteRealQ),{1},Heads->False];
	
	If[badChi=!={},Print["ERROR: non-numeric ChiPlus values found. badChi=",badChi];
	Return[$Failed];
	];
	
	dataVals=N@Transpose[{kVals,chiVals}];
	
	plot=ListLinePlot[
	{Transpose[{kVals,chiVals}]},
	PlotLegends->{"ChiPlus"},
	Frame->True,
	FrameLabel->{"k","\[Chi]Plus"},
	PlotLabel->Row[{
	ToUpperCase[state]," ChiPlus, ",deltaTagLocal,
	", M=",M,
	", gamma=",gamma,
	", wp=",wp
	}],
	ImageSize->700
	];
	
	gammaTagLocal=StringReplace[
	ToString@NumberForm[N[gamma],{Infinity,2},NumberPadding->{"","0"}],
	" "->""];
	
	stateTagLocal=Switch[
	state,
	"neel","NEEL",
	"polarized","POLARIZED",
	_,"UNKNOWN"
	];


  fileST = StringTemplate["CHI_PLUS_``_``_M``_gamma``_wp``"][
    stateTagLocal, deltaTagLocal, M, gammaTagLocal, wp
  ];
  csvOut = FileNameJoin[{chiCsvDir, fileST <> ".csv"}];
  pngOut = FileNameJoin[{chiPngDir, fileST <> ".png"}];

  Export[csvOut, Prepend[dataVals, {"k", "ChiPlus"}], "CSV"];
  Export[pngOut, plot, "PNG"];
  Print["Saved ChiPlus CSV: ", csvOut];
  Print["Saved ChiPlus PNG: ", pngOut];

  <|"status" -> "ok", "csv" -> csvOut, "png" -> pngOut|>
];

exportOneProfile[solAssoc_Association, rs_List] := Module[
  {
    deltaTagLocal, rValsLocal, bValsLocal, gammaTagLocal, stateTagLocal,
    rval, signval, qVals, JVals, badQ, badJ, dataVals, plot, fileST, csvOut, pngOut
  },
  deltaTagLocal = solAssoc["deltaTag"];
  rValsLocal = solAssoc["rVals"];
  bValsLocal = solAssoc["bVals"];
  stateTagLocal = stateTagFrom[state];
  gammaTagLocal = gammaTagFrom[gamma];
  {rval, signval} = rs;

  (* TODO: Copy the body of the INNER profile block from GHD_hybrid_test.wl
     lines 309-359.
     Important:
     - use rValsLocal/bValsLocal
     - do not solve again here
     - keep filenames tagged by deltaTagLocal
     - replace any Continue[] with Return[$Failed] or a small failure association
  *)
  
  

	gammaTagLocal=StringReplace[
	ToString@NumberForm[N[gamma],{Infinity,2},NumberPadding->{"","0"}],
	" "->""
	];
	

	Print["Running ",deltaTagLocal," r=",rval," sign=",signval];

	qVals=qAt[#,rval,Jhop,signval,rValsLocal,bValsLocal]&/@zetasVal;
	JVals=jAt[#,rval,Jhop,signval,rValsLocal,bValsLocal]&/@zetasVal;
	
	badQ=Flatten@Position[qVals,_?(Not@*finiteRealQ),{1},Heads->False];
	badJ=Flatten@Position[JVals,_?(Not@*finiteRealQ),{1},Heads->False];

	If[badQ=!={}||badJ=!={},
	Print["ERROR: non-numeric values found. badQ=",badQ," badJ=",badJ];
	Return[$Failed];
	];

	dataVals=N@Transpose[{zetasVal,qVals,JVals}];

	plot=ListLinePlot[
	{
	Transpose[{zetasVal,qVals}],
	Transpose[{zetasVal,JVals}]
	},
	PlotLegends->{"q","J"},
	Frame->True,
	FrameLabel->{"zeta","value"},
	PlotLabel->Row[{
	ToUpperCase[state]," hybrid, ",deltaTagLocal,
	", r=",rval,
	", sign=",signval,
	", M=",M,
	", gamma=",gamma,
	", wp=",wp
	}],
	ImageSize->700
	];


  fileST = StringTemplate[
     "GHD_``_HYBRID_``_r``_sign``_M``_gamma``_wp``_seggrid"
  ][stateTagLocal, deltaTagLocal, rval, signval, M, gammaTagLocal, wp];
  csvOut = FileNameJoin[{csvDir, fileST <> ".csv"}];
  pngOut = FileNameJoin[{pngDir, fileST <> ".png"}];

  Export[csvOut, Prepend[dataVals, {"zeta", "q", "J"}], "CSV"];
  Export[pngOut, plot, "PNG"];
  Print["Saved CSV: ", csvOut];
  Print["Saved PNG: ", pngOut];

  <|"status" -> "ok", "csv" -> csvOut, "png" -> pngOut|>
];

main[] := Module[
  {solutions, chiResults, profileTasks, profileResults, q0Diag, j0Diag, lastSol},

  DistributeDefinitions[
    finiteRealQ, SolveTruncatedSystemDirectEff,
    stepTheta, nBase, nL, nR, ChiPlus, nZeta,
    chargeIntegrand, currentIntegrand, qAt, jAt,
    M, Jhop, gamma, wp, kVals, zetasVal,
    state, profileList, profileDeltaTable,
    chiCsvDir, chiPngDir, csvDir, pngDir,
    deltaTagFrom, stateTagFrom, gammaTagFrom,
    makeSolutionAssoc, exportChiPlus, exportOneProfile
  ];

  solutions = DeleteCases[
    ParallelMap[makeSolutionAssoc, profileDeltaTable, Method -> "CoarsestGrained"],
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
    q0Diag = qAt[zDiag, rDiag, Jhop, signDiag, lastSol["rVals"], lastSol["bVals"]];
    j0Diag = jAt[zDiag, rDiag, Jhop, signDiag, lastSol["rVals"], lastSol["bVals"]];
    Print["Diagnostic q = ", q0Diag];
    Print["Diagnostic J = ", j0Diag];
  ];

  profileTasks = Flatten[
    Table[{solAssoc, rs}, {solAssoc, solutions}, {rs, profileList}],
    1
  ];
  Print["Prepared ", Length[profileTasks], " profile tasks."];

  profileResults = DeleteCases[
    ParallelMap[
      exportOneProfile @@ # &,
      profileTasks,
      Method -> "CoarsestGrained"
    ],
    $Failed
  ];
  Print["Profile exports completed for ", Length[profileResults], " tasks."];
];

main[];
CloseKernels[];
Quit[];
