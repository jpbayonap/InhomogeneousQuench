(* ::Package:: *)

(*-- settings*)
M=350;
Jhop =1.0;
gamma =1.0;
beta= 1.0;
wp =80;
kVals = Subdivide[-Pi, Pi, 800];

deltaRtest = {
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






deltaTag = "delta"<>ToString[Length[deltaRtest]];
Print["deltaTag = ", deltaTag];

(* delta Table*)

deltaRTable = Table[Take[deltaRtest, n], {n, 1, Length[deltaRtest], 2}];
profileDeltaTable = deltaRTable;

Grid[Prepend[Table[{"delta"<>ToString[Length[d]], Length[d], d},
{d, deltaRTable}], {"tag", "length", "deltaR"}], Frame -> All ]

(*profiles list *)
profilesOdd = Table[{r,"+"},{r,{1, 3, 5}}] ;
profilesEven= Table[{r,"-"},{r,{2, 4, 6}}] ; 

profileList = Join[
	profilesOdd,
	profilesEven];
	
Print[profileList]
(* zetas*)
zetasVal=Join[Subdivide[-2.5,-1.,40],Rest@Subdivide[-1.,1.,160],
Rest@Subdivide[1.,2.5,40]];

repoBase="/Users/juan/Desktop/Git/InhomogeneousQuench/C_pp";
outBase=FileNameJoin[{repoBase,"GHD_BETA_HYBRID"}];

chiCsvDir=FileNameJoin[{outBase,"chi_plus","csv"}];
chiPngDir=FileNameJoin[{outBase,"chi_plus","png"}];

Scan[If[!DirectoryQ[#],CreateDirectory[#,CreateIntermediateDirectories->True]]&,{chiCsvDir,chiPngDir}];

Print["chiCsvDir = ",chiCsvDir];
Print["chiPngDir = ",chiPngDir];


csvDir=FileNameJoin[{outBase,"csv"}];
pngDir=FileNameJoin[{outBase,"png"}];

Scan[If[!DirectoryQ[#],CreateDirectory[#,CreateIntermediateDirectories->True]]&,{csvDir,pngDir}];

Print["state = beta"];
Print["csvDir = ",csvDir];
Print["pngDir = ",pngDir];






(* test parameters*)
rTest = 1;
zetasTest=Join[Subdivide[-2.5,-1.,4],Rest@Subdivide[-1.,1.,4],
Rest@Subdivide[1.,2.5,4]];
zetaTest = -1.0;



finiteRealQ[x_]:=NumericQ[x]&&FreeQ[N[x],_DirectedInfinity|Indeterminate|ComplexInfinity]&&TrueQ[Im[N[x]]==0];

(* =====Smooth step=====*)
eps=SetPrecision[10^-4,wp];
ThetaN[x_?NumericQ]:=1/2 (1+Tanh[x/eps]);

(* =====Matsubara series for n_r(beta)=====*)
omega[n_,b_]:=(2 n+1) Pi/b;

nCoeffMatsubara[r_Integer?Positive,b_?NumericQ,nMax_Integer?NonNegative,J_?NumericQ]:=
Module[{m,sum,om},If[EvenQ[r],Return[0.0`20]];
m=(r-1)/2;
sum=Sum[om=omega[n,b];
((Sqrt[om^2+4 J^2]-om)/(2 J))^r/Sqrt[om^2+4 J^2],{n,0,nMax}];
N[(2 Pi/b)*(-1)^m*sum,wp]];

(* Fourier Coefficients calculation *)
(* =====g_r^(n)=====*)
gMatElem[r_Integer?Positive,n_Integer?Positive,J_?NumericQ]:=
Module[{den,num,pf},
If[r===n,Return[N[8 J^2 (2 n^2-1)/(4 n^2-1),wp]]];
pf=1+(-1)^(r+n);
If[pf===0,Return[0]];
num=(n^2+r^2-1)*pf;
den=n^4-2 n^2 (r^2+1)+(r^2-1)^2;
N[-4 J^2 num/den, wp]
];
(* =====Linear system for a_r=====*)
ClearAll[SolveThermalSystemDirectTherm];
SolveThermalSystemDirectTherm[mVal_, betaVal_, gammaVal_, jVal_, delta_, nMax_: 4000, wpVal_: 60] :=
Module[
{rVals, gMat, A, rhs, a, DeltaPad, res},

If[!IntegerQ[mVal] || mVal <= 0,
  Print["ERROR: mVal must be a positive integer, got ", InputForm[mVal]];
  Return[$Failed];
];
If[!NumericQ[betaVal] || !NumericQ[gammaVal] || !NumericQ[jVal] || !NumericQ[wpVal],
  Print["ERROR: betaVal, gammaVal, jVal, wpVal must be numeric."];
  Return[$Failed];
];
If[!IntegerQ[nMax] || nMax <= 0,
  Print["ERROR: nMax must be a positive integer, got ", InputForm[nMax]];
  Return[$Failed];
];
If[Head[delta] =!= List,
  Print["ERROR: delta must be a list, got ", Head[delta]];
  Return[$Failed];
];

rVals = Range[mVal];

gMat=Table[gMatElem[r,n,jVal],{r,1,mVal},{n,1,mVal}];
(* delta_r *)
DeltaPad= PadRight[N@delta,mVal,0.0];
A =  gMat ;
rhs = -0.5*gammaVal * DeltaPad;

(* CHECK*)
Print["Dimensions[gMat] = ", Dimensions[gMat]];
Print["Length[DeltaPad] = ", Length[DeltaPad]];
Print["Dimensions[A] = ", Dimensions[A]];
Print["Length[rhs] = ", Length[rhs]];

a=LinearSolve[N[A,wpVal],N[rhs,wpVal],Method->"Direct"];
res=Norm[A . a-rhs]/Max[Norm[rhs],10^-50];
Print["relative residual = ",N[res,20]];
{rVals,N[a,30]}];

(* =====Hydro workflow=====*)
rhoThermal[k_?NumericQ,b_?NumericQ, J_?NumericQ]:=1/(2 Pi (Exp[-2 b J Cos[k]]+1));

betaL=beta;betaR=beta;
nL[k_?NumericQ]:=rhoThermal[k,betaL, Jhop];
nR[k_?NumericQ]:=rhoThermal[k,betaR, Jhop];

ChiPlus[k_, rVals_List, aVals_List] := rhoThermal[k, beta, Jhop] + (aVals . Cos[rVals k]);

nZetaPos[k_?NumericQ,z_?NumericQ,rVals_List,aVals_List]:=Module[{epsp,chi},epsp=2 Jhop Sin[k];
chi=rhoThermal[k,beta, Jhop]+(aVals . Cos[rVals k]);
ThetaN[-z]*nL[k]+ThetaN[z]*ThetaN[epsp-z]*chi+ThetaN[z-epsp]*nR[k]];

nZetaNeg[k_?NumericQ,z_?NumericQ,rVals_List,aVals_List]:=Module[{epsp,chi},epsp=2 Jhop Sin[k];
chi=rhoThermal[k,beta, Jhop]+(aVals . Cos[rVals k]);
ThetaN[z]*nR[k]+ThetaN[-z]*ThetaN[-epsp+z]*chi+ThetaN[epsp-z]*nL[k]];

(* =====HydCharge/HydCurrent=====*)
qPlus[k_,r_?NumericQ]:=2 Cos[r k];
qMinus[k_,r_?NumericQ]:=-2 Sin[r k];

HydCharge[r_?NumericQ,z_?NumericQ,sign_:"+",rVals_List,aVals_List]:=
Module[{fpos,fneg},fpos[k_?NumericQ]:=If[sign==="-",qMinus[k,r],qPlus[k,r]]*nZetaPos[k,z,rVals,aVals];
fneg[k_?NumericQ]:=If[sign==="-",qMinus[k,r],qPlus[k,r]]*nZetaNeg[k,z,rVals,aVals];
NIntegrate[fpos[k],{k,0,Pi},Method->"GlobalAdaptive",WorkingPrecision->wp,AccuracyGoal->12,PrecisionGoal->12,
MinRecursion->4,MaxRecursion->60]+NIntegrate[fneg[k],{k,-Pi,0},Method->"GlobalAdaptive",WorkingPrecision->wp,AccuracyGoal->12,PrecisionGoal->12,MinRecursion->4,MaxRecursion->60]];

HydCurrent[r_?NumericQ,z_?NumericQ,sign_:"+",rVals_List,aVals_List]:=
Module[{fpos,fneg},fpos[k_?NumericQ]:=(2 Jhop Sin[k])*If[sign==="-",qMinus[k,r],qPlus[k,r]]*nZetaPos[k,z,rVals,aVals];
fneg[k_?NumericQ]:=(2 Jhop Sin[k])*If[sign==="-",qMinus[k,r],qPlus[k,r]]*nZetaNeg[k,z,rVals,aVals];
NIntegrate[fpos[k],{k,0,Pi},Method->"GlobalAdaptive",WorkingPrecision->wp,AccuracyGoal->12,PrecisionGoal->12,
MinRecursion->4,MaxRecursion->60]+NIntegrate[fneg[k],{k,-Pi,0},Method->"GlobalAdaptive",WorkingPrecision->wp,AccuracyGoal->12,PrecisionGoal->12,MinRecursion->4,MaxRecursion->60]];

HydCurrentCSV[r_?NumericQ,z_?NumericQ,sign_:"+",rVals_List,aVals_List] :=
  0.5 HydCurrent[r, z, sign, rVals, aVals];



(* TEST *)
signTest = "+";
beta = 1.0;


sol = SolveThermalSystemDirectTherm[M, beta, gamma, Jhop, deltaRtest, 4000, wp];
(*Print["Head[sol] = ", Head[sol]];*)
(*Print["sol = ", InputForm[sol]];*)
If[!MatchQ[sol, {_List, _List}],
  Print["SolveThermalSystemDirectTherm did not return {rVals, aVals}."];
  Abort[];
];
{rVals, aVals} = sol;

qSingle = HydCharge[rTest, zetaTest, signTest, rVals, aVals];
jSingle = HydCurrent[rTest, zetaTest, signTest, rVals, aVals];

Print["zetaTest = ", zetaTest];
Print["qSingle = ", qSingle];
Print["jSingle = ",  jSingle];




(* LONG RUN: r = 1,3,5 *)

(*sol = SolveThermalSystemDirectTherm[M, beta, gamma, Jhop, deltaRtest, 4000, wp];
If[!MatchQ[sol, {_List, _List}],
  Print["SolveThermalSystemDirectTherm did not return {rVals, aVals}."];
  Abort[];
];
{rVals, aVals} = sol;

gammaTag = StringReplace[
  ToString @ NumberForm[N[gamma], {Infinity, 2}, NumberPadding -> {"", "0"}],
  " " -> ""
];

betaTag = StringReplace[
  ToString @ NumberForm[N[beta], {Infinity, 1}, NumberPadding -> {"", "0"}],
  " " -> ""
];

Do[
  {rval, signval} = rs;

  Print["Running ", deltaTag, " r=", rval, " sign=", signval];

  qVals = HydCharge[rval, #, signval, rVals, aVals] & /@ zetasVal;
  (*jVals = HydCurrent[rval, #, signval, rVals, aVals] & /@ zetasVal;*)
  jVals = HydCurrentCSV[rval, #, signval, rVals, aVals] & /@ zetasVal;


  badQ = Flatten @ Position[qVals, _?(Not @* finiteRealQ), {1}, Heads -> False];
  badJ = Flatten @ Position[jVals, _?(Not @* finiteRealQ), {1}, Heads -> False];

  If[badQ =!= {} || badJ =!= {},
    Print["ERROR: non-numeric values found. badQ=", badQ, " badJ=", badJ];
    Continue[];
  ];

  dataVals = N @ Transpose[{zetasVal, qVals, jVals}];

  plot = ListLinePlot[
    {
      Transpose[{zetasVal, qVals}],
      Transpose[{zetasVal, jVals}]
    },
    PlotLegends -> {"q", "0.5 J"},
    Frame -> True,
    FrameLabel -> {"zeta", "value"},
    PlotLabel -> Row[{
      "THERM beta, ", deltaTag,
      ", r=", rval,
      ", sign=", signval,
      ", beta=", betaTag,
      ", gamma=", gammaTag,
      ", M=", M,
      ", wp=", wp
    }],
    ImageSize -> 700
  ];

  fileStem = StringTemplate[
    "GHD_THERM2_Beta_``_r``_sign``_beta``_gamma``_M``_wp``_seggrid"
  ][deltaTag, rval, signval, betaTag, gammaTag, M, wp];

  csvOut = FileNameJoin[{csvDir, fileStem <> ".csv"}];
  pngOut = FileNameJoin[{pngDir, fileStem <> ".png"}];

  Export[csvOut, Prepend[dataVals, {"zeta", "q", "J"}], "CSV"];
  Export[pngOut, plot, "PNG"];

  Print["Saved CSV: ", csvOut];
  Print["Saved PNG: ", pngOut];

, {rs, {{1, "+"}, {3, "+"}, {5, "+"}}}]


*)
