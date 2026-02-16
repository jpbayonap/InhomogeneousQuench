(* ::Package:: *)

(*
  GHD_THERMAL_MAT_full.wl
  ----------------------
  Direct solver for the truncated linear system defining a_r(γ), plus the
  same hydrodynamic post-processing workflow used in the attached notebook
  (GHD_analytics.nb): ChiPlus, nZeta, HydCharge, HydCurrent, and profile export.

  Your truncated equation:
    a_r(γ)= -4/π n_r(β) - 1/(π r γ J) Sum_{n=1..M} a_n(γ) g_r^(n)

  Multiply by π r γ J (recommended for arbitrarily small γ):
    Sum_{n=1..M} g_r^(n) a_n  + (π r γ J) a_r = -4 r γ J n_r(β)

  with
    g_r^(n) =
      16 J^2 n^2/(4 n^2 - 1)                                    if r=n
      4 J^2 (n^2+r^2-1)(1+(-1)^(r+n)) / (n^4 -2n^2(r^2+1) + (r^2-1)^2)  if r≠n

  and
    n_r(β)= ∫_0^π dk  cos(rk)/(e^{-2β cos k}+1)

  Outputs are exported with prefix:  GHD_THERMAL_MAT_...
*)

ClearAll["Global`*"];

(* =========================
   User parameters (edit)
   ========================= *)
M      = 80;        (* truncation size *)
beta   = 2.0;       (* inverse temperature β *)
gamma  = 1.*^-8;    (* monitoring rate γ (can be very small) *)
Jhop   = 1.0;       (* set J=1 *)
Nw     = 4000;      (* number of positive Matsubara frequencies in n_r series *)
wp     = 80;        (* WorkingPrecision *)
zetaMin = -2.0;     (* ray window for profile plots *)
zetaMax =  2.0;
numZeta = 401;      (* number of ζ points *)
outDir  = NotebookDirectory[] /. $Failed -> Directory[];
fileTag = "thermal"; (* arbitrary tag appended to filenames *)

(* ============================================================
   0) Step function convention used throughout (matches notebook)
   ============================================================ *)
Theta[x_] := UnitStep[x];

(* ============================================================
   1) Thermal Fourier cosine coefficients n_r(β) via Matsubara sum
   ============================================================

   We use the paired (manifestly real) series:

     n_{2m}(β)=0 for m>=1

     n_{2m+1}(β)= (2π/β)(-1)^m ∑_{n=0}^∞  λ(ω_n)^{2m+1} / √(ω_n^2+4),
       ω_n=(2n+1)π/β,  λ(Ω)=(√(Ω^2+4)-Ω)/2 ∈ (0,1)

   Sanity checks:
     - Real by construction
     - Even r>0 vanish exactly
     - Converges absolutely (tail ~ ω_n^{-(r+1)})
*)

omega[n_, β_] := (2 n + 1) Pi/β;                (* n = 0,1,2,... *)
lambdaFromOmega[Ω_] := (Sqrt[Ω^2 + 4] - Ω)/2;   (* Ω>0 -> (0,1) *)

nCoeff[r_Integer?Positive, β_?NumericQ, nMax_Integer?NonNegative] := Module[
  {m, sum},
  If[EvenQ[r], Return[0.0`20]];
  m = (r - 1)/2;
  sum = Sum[
    With[{Ω = omega[n, β], λ = lambdaFromOmega[omega[n, β]]},
      λ^r / Sqrt[Ω^2 + 4]
    ],
    {n, 0, nMax}
  ];
  N[(2 Pi/β) * (-1)^m * sum, wp]
];

(* Optional numerical check (slow): nCoeffNIntegrate[r,beta] *)
nCoeffNIntegrate[r_Integer?Positive, β_?NumericQ] := NIntegrate[
  Cos[r k]/(Exp[-2 β Cos[k]] + 1),
  {k, 0, Pi},
  Method -> {"GlobalAdaptive", "MaxErrorIncreases" -> 200, "SymbolicProcessing" -> 0},
  WorkingPrecision -> wp,
  AccuracyGoal -> 25, PrecisionGoal -> 25
];

(* Low-T β→∞ limit *)
nCoeffLowT[r_Integer?NonNegative] := Module[{m},
  Which[
    r == 0, Pi/2,
    EvenQ[r], 0,
    True, m = (r - 1)/2; (-1)^m / r
  ]
];

(* Large-r (odd) estimate: keep only n=0 Matsubara term *)
nCoeffLargeROdd[r_Integer?Positive, β_?NumericQ] := Module[{m, Ω0, λ0, pref},
  If[EvenQ[r], Return[0]];
  m = (r - 1)/2;
  Ω0 = omega[0, β];
  λ0 = lambdaFromOmega[Ω0];
  pref = (2 Pi/β) * (-1)^m / Sqrt[Ω0^2 + 4];
  N[pref * λ0^r, wp]
];

(* ============================================================
   2) Matrix elements g_r^(n) (your closed form)
   ============================================================ *)
gMatElem[r_Integer?Positive, n_Integer?Positive, J_?NumericQ] := Module[
  {den, num, parityFactor},
  If[r === n, Return[N[16 J^2 * n^2/(4 n^2 - 1), wp]]];
  parityFactor = 1 + (-1)^(r + n);   (* 0 if r+n odd; 2 if r+n even *)
  If[parityFactor === 0, Return[0]];
  num = (n^2 + r^2 - 1) * parityFactor;
  den = n^4 - 2 n^2 (r^2 + 1) + (r^2 - 1)^2;
  N[4 J^2 * num/den, wp]
];

(* ============================================================
   3) Solve the truncated linear system for a_n(γ)
   ============================================================ *)
SolveThermalSystemDirect[M_Integer?Positive, β_?NumericQ, γ_?NumericQ, J_?NumericQ : 1.0, nMax_Integer?NonNegative : 4000] := Module[
  {rVals, nList, G, A, b, a, cond},
  rVals = Range[M];

  nList = Table[nCoeff[r, β, nMax], {r, 1, M}];

  G = Table[gMatElem[r, n, J], {r, 1, M}, {n, 1, M}];

  (* A a = b with A_{rn} = g_r^(n) + π r γ J δ_{rn},  b_r = -4 r γ J n_r *)
  A = G + DiagonalMatrix[N[Pi * γ * J * rVals, wp]];
  b = Table[N[-4 r γ J * nList[[r]], wp], {r, 1, M}];

  cond = ConditionNumber[N[A, wp]];
  Print["ConditionNumber(A) ≈ ", cond];

  a = LinearSolve[N[A, wp], N[b, wp], Method -> "Krylov"];

  <|
    "M" -> M, "beta" -> β, "gamma" -> γ, "J" -> J, "Nw" -> nMax,
    "rVals" -> rVals,
    "nList" -> nList,
    "G" -> G,
    "A" -> A,
    "b" -> b,
    "aVec" -> a,
    "psiR" -> Total[a],
    "ConditionNumber" -> cond
  |>
];

(* ============================================================
   4) Hydrodynamics workflow (mirrors attached notebook)
   ============================================================ *)

(* Default initial state: thermal-thermal at same β (edit if needed).
   IMPORTANT: In the attached notebook, nL/nR are root densities (include 1/(2π)).
   Here we keep that convention.
*)
rhoThermal[k_?NumericQ, β_?NumericQ] := 1.0/(2.0 Pi (Exp[-2.0 β Cos[k]] + 1.0));

betaL = beta; betaR = beta;
nL[k_?NumericQ] := rhoThermal[k, betaL];
nR[k_?NumericQ] := rhoThermal[k, betaR];

(* Constant used in ChiPlus on the k<0 side in the notebook.
   For particle-hole symmetric thermal states (μ=0), the average filling is 1/2,
   hence the constant Fourier mode is 1/(4π). We keep that default.
*)
chiConst = 1.0/(4.0 Pi);

(* ChiPlus: identical structure as the notebook, but using aVals instead of bVals *)
ChiPlus[k_, rVals_List, aVals_List] :=
  Theta[k] * (aVals . Sin[rVals * k]) +
  Theta[-k] * (chiConst + aVals . Sin[rVals * k]);

(* nZeta(k,ζ): copied from notebook logic *)
nZeta[k_, zeta_, rVals_List, aVals_List] := Module[{epsp, chi},
  epsp = 2.0 * Jhop * Sin[k];
  chi  = ChiPlus[k, rVals, aVals];

  If[k > 0,
    (* k>0 branch *)
    Theta[-zeta] * nL[k] +
    Theta[zeta]  * Theta[epsp - zeta] * chi +
    Theta[zeta - epsp] * nR[k]
    ,
    (* k<0 branch *)
    Theta[zeta] * nR[k] +
    Theta[-zeta] * Theta[-epsp + zeta] * chi +
    Theta[epsp - zeta] * nL[k]
  ]
];

(* Single-mode charge symbols (as in notebook) *)
qPlus[k_, r_Integer]  :=  2.0 * Cos[r * k];
qMinus[k_, r_Integer] := -2.0 * Sin[r * k];

(* Local hydrodynamic charge and current (as in notebook) *)
HydCharge[r_Integer, zeta_?NumericQ, sign_ : "-", rVals_List, aVals_List] := Module[{f},
  f[k_?NumericQ] := If[sign === "-", qMinus[k, r], qPlus[k, r]] * nZeta[k, zeta, rVals, aVals];
  NIntegrate[
    f[k],
    {k, -Pi, Pi},
    Method -> {"GlobalAdaptive", "SymbolicProcessing" -> 0},
    MinRecursion -> 4, MaxRecursion -> 40, AccuracyGoal -> 8
  ]
];

HydCurrent[r_Integer, zeta_?NumericQ, sign_ : "-", rVals_List, aVals_List] := Module[{f},
  f[k_?NumericQ] := (2.0 * Jhop * Sin[k]) * If[sign === "-", qMinus[k, r], qPlus[k, r]] * nZeta[k, zeta, rVals, aVals];
  NIntegrate[
    f[k],
    {k, -Pi, Pi},
    Method -> {"GlobalAdaptive", "SymbolicProcessing" -> 0},
    MinRecursion -> 4, MaxRecursion -> 40, AccuracyGoal -> 8
  ]
];

(* ============================================================
   5) Profile computation + export (same style as notebook)
   ============================================================ *)
ComputeGHDProfilesThermal[M_Integer?Positive, β_?NumericQ, γ_?NumericQ, rCharge_Integer?Positive, sign_ : "+",
  zetaMin_?NumericQ, zetaMax_?NumericQ, numZeta_Integer?Positive, outDir_String, tag_String : "run"] := Module[
  {sol, rVals, aVals, zetas, qVals, JVals, data, fileBase, plt, ymin, ymax, pad, safeGamma, safeBeta},

  Print["Solving truncated system for a_r(γ)..."];
  sol = SolveThermalSystemDirect[M, β, γ, Jhop, Nw];
  rVals = sol["rVals"]; aVals = sol["aVec"];

  Print["Building zeta grid..."];
  zetas = Subdivide[zetaMin, zetaMax, numZeta - 1];

  Print["Computing hydrodynamic charge/current profiles..."];
  qVals = Table[HydCharge[rCharge, z, sign, rVals, aVals], {z, zetas}];
  JVals = Table[HydCurrent[rCharge, z, sign, rVals, aVals], {z, zetas}];

  data = Transpose[{zetas, qVals, JVals}];

  safeGamma = StringReplace[ToString[ScientificForm[γ, 3]], {" " -> "", "*" -> "x", "^" -> ""}];
  safeBeta  = StringReplace[ToString[NumberForm[β, {Infinity, 6}]], {" " -> ""}];

  fileBase = FileNameJoin[{
    outDir,
    "GHD_THERMAL_MAT_" <> tag <>
    "_M" <> ToString[M] <>
    "_beta" <> safeBeta <>
    "_gamma" <> safeGamma <>
    "_r" <> ToString[rCharge] <>
    "_sign" <> sign
  }];

  Export[fileBase <> ".csv", Prepend[data, {"zeta", "q", "J"}]];

  ymin = Min[Join[qVals, JVals]]; ymax = Max[Join[qVals, JVals]];
  pad = 0.05 (ymax - ymin);

  plt = ListLinePlot[
    {Transpose[{zetas, qVals}], Transpose[{zetas, JVals}]},
    PlotLegends -> {"q", "J"},
    PlotRange -> {{zetaMin, zetaMax}, {ymin - pad, ymax + pad}},
    Frame -> True, FrameLabel -> {"zeta", "value"},
    ImageSize -> Large
  ];

  Export[fileBase <> ".png", plt];

  Print["Saved data to: ", fileBase <> ".csv"];
  Print["Saved plot to: ", fileBase <> ".png"];

  <|"Zetas" -> zetas, "qVals" -> qVals, "JVals" -> JVals, "rVals" -> rVals, "aVals" -> aVals, "Solution" -> sol, "FileBase" -> fileBase|>
];

(* ============================================================
   6) One-shot run (comment out if you want to call manually)
   ============================================================ *)
resultsPlus  = ComputeGHDProfilesThermal[M, beta, gamma, 1, "+", zetaMin, zetaMax, numZeta, outDir, fileTag];
resultsMinus = ComputeGHDProfilesThermal[M, beta, gamma, 1, "-", zetaMin, zetaMax, numZeta, outDir, fileTag];

(* Also export coefficients alone (for quick reuse) *)
tag2 = StringRiffle[{
  "M" <> ToString[M],
  "beta" <> ToString[NumberForm[beta, {Infinity, 6}]],
  "gamma" <> ToString[ScientificForm[gamma, 3]],
  DateString[{"YYYYMMDD", "_", "HHmmss"}]
}, "_"];

coeffBase = FileNameJoin[{outDir, "GHD_THERMAL_MAT_aCoeffs_" <> tag2}];
Export[coeffBase <> ".mx", resultsPlus["Solution"]];
Export[coeffBase <> ".csv", Join[{{"n", "a_n"}}, Table[{n, resultsPlus["aVals"][[n]]}, {n, 1, M}]]];
Print["Exported coefficients: ", coeffBase <> ".mx /.csv"];

