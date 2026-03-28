(* ::Package:: *)

(*-- settings*)
M=350;
Jhop =1.0;
gamma =1.0;
beta= 1.0;
wp =80;
kVals = Subdivide[-Pi, Pi, 800];

deltaRtest = {
  -1.740005463361740112*^-1,
   0.0,
  -1.344909554952755659*^-1,
   0.0,
  -3.211390040814876556*^-2,
   0.0,
  -1.395109202712774277*^-2,
   0.0,
  -6.847018492408096790*^-3,
   0.0,
  -4.494369844906032085*^-3,
   0.0,
  -3.257162490626797080*^-3,
   0.0,
  -2.508738019969314337*^-3,
   0.0,
  -2.009734649618621916*^-3,
   0.0,
  -1.646561759116593748*^-3,
   0.0,
  -1.358057539619039744*^-3,
   0.0,
  -1.113314126996556297*^-3,
   0.0,
  -8.924128833314171061*^-4,
   0.0,
  -6.848142588751215953*^-4,
   0.0,
  -4.860054077653330751*^-4,
   0.0,
  -2.927731093222973868*^-4
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
rTest = 3;
zetasTest=Join[Subdivide[-2.5,-1.,4],Rest@Subdivide[-1.,1.,4],
Rest@Subdivide[1.,2.5,4]];
zetaTest = 0.0;



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
{rVals, gMat, A, rhs, a, DeltaPad, res, nList},

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

nList=Table[nCoeffMatsubara[r, betaVal, nMax, jVal],{r, 1, mVal}];
gMat=Table[gMatElem[r,n,jVal],{r,1,mVal},{n,1,mVal}];
(* delta_r *)
DeltaPad= PadRight[N@delta,mVal,0.0];
A = gMat + DiagonalMatrix[Table[N[Pi*gammaVal*jVal, wpVal], {r, 1, mVal}]];

rhs=Table[N[-(2 gammaVal/Pi)*nList[[r]],wpVal],{r,1,mVal}] -DeltaPad;
(* CHECK*)
Print["Dimensions[gMat] = ", Dimensions[gMat]];
Print["Length[nList] = ", Length[nList]];
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

HydCharge[r_?NumericQ,z_?NumericQ,sign_:"+",rVals_List,aVals_List]:=Module[{fpos,fneg},fpos[k_?NumericQ]:=If[sign==="-",qMinus[k,r],qPlus[k,r]]*nZetaPos[k,z,rVals,aVals];
fneg[k_?NumericQ]:=If[sign==="-",qMinus[k,r],qPlus[k,r]]*nZetaNeg[k,z,rVals,aVals];
NIntegrate[fpos[k],{k,0,Pi},Method->"GlobalAdaptive",WorkingPrecision->wp,AccuracyGoal->12,PrecisionGoal->12,MinRecursion->4,MaxRecursion->60]+NIntegrate[fneg[k],{k,-Pi,0},Method->"GlobalAdaptive",WorkingPrecision->wp,AccuracyGoal->12,PrecisionGoal->12,MinRecursion->4,MaxRecursion->60]];

HydCurrent[r_?NumericQ,z_?NumericQ,sign_:"+",rVals_List,aVals_List]:=Module[{fpos,fneg},fpos[k_?NumericQ]:=(2 Jhop Sin[k])*If[sign==="-",qMinus[k,r],qPlus[k,r]]*nZetaPos[k,z,rVals,aVals];
fneg[k_?NumericQ]:=(2 Jhop Sin[k])*If[sign==="-",qMinus[k,r],qPlus[k,r]]*nZetaNeg[k,z,rVals,aVals];
NIntegrate[fpos[k],{k,0,Pi},Method->"GlobalAdaptive",WorkingPrecision->wp,AccuracyGoal->12,PrecisionGoal->12,MinRecursion->4,MaxRecursion->60]+NIntegrate[fneg[k],{k,-Pi,0},Method->"GlobalAdaptive",WorkingPrecision->wp,AccuracyGoal->12,PrecisionGoal->12,MinRecursion->4,MaxRecursion->60]];


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
Print["jSingle = ", jSingle];



