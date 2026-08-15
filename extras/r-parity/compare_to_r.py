import json, warnings, sys
import numpy as np, pandas as pd
import empiricalcalibration as ec
from empiricalcalibration import _rmath as rm
from empiricalcalibration.likelihoods import gridLlApproximation
from empiricalcalibration.llr_calibration import computePFromLlr, computeLlrFromP
from empiricalcalibration.maxsprt import (samplePoissonMaxLrr, sampleBinomialMaxLrr,
                                          samplePoissonRegressionMaxLrr)

warnings.simplefilter("ignore")
G = json.load(open("gold.json"))
got = {}
def g(k, v):
    got[k] = np.atleast_1d(np.asarray(v, dtype=float)).ravel()

sccs = ec.datasets.sccs()
neg = sccs[sccs.groundTruth == 0]; pos = sccs[sccs.groundTruth == 1]
null = ec.fitNull(neg.logRr, neg.seLogRr)
g("null", [null[0], null[1]])
g("ease", ec.computeExpectedAbsoluteSystematicError(null))

ec.set_seed(101)
ctrl = ec.simulateControls(n=50*3, mean=0.25, sd=0.25, trueLogRr=np.log([1,2,4]))
g("sim.logRr", ctrl.logRr); g("sim.seLogRr", ctrl.seLogRr); g("sim.trueLogRr", ctrl.trueLogRr)
m = ec.fitSystematicErrorModel(ctrl.logRr, ctrl.seLogRr, ctrl.trueLogRr)
g("model", m)
ml = ec.fitSystematicErrorModel(ctrl.logRr, ctrl.seLogRr, ctrl.trueLogRr, legacy=True)
g("modelLegacy", ml)
mc = ec.fitSystematicErrorModel(ctrl.logRr, ctrl.seLogRr, ctrl.trueLogRr,
                                estimateCovarianceMatrix=True)
g("modelLB95", mc.attributes["LB95CI"]); g("modelUB95", mc.attributes["UB95CI"])
g("modelCov", mc.attributes["CovarianceMatrix"].T.ravel())

ec.set_seed(202)
nd = ec.simulateControls(n=15, mean=0.25, sd=0.25, trueLogRr=np.log([1,2,4]))
ci = ec.calibrateConfidenceInterval(nd.logRr, nd.seLogRr, m)
g("ci.logRr", ci.logRr); g("ci.lb", ci.logLb95Rr); g("ci.ub", ci.logUb95Rr); g("ci.se", ci.seLogRr)
cil = ec.calibrateConfidenceInterval(nd.logRr, nd.seLogRr, ml)
g("ciL.logRr", cil.logRr); g("ciL.lb", cil.logLb95Rr); g("ciL.ub", cil.logUb95Rr)

em = ec.convertNullToErrorModel(null); g("errModel", em)
ci2 = ec.calibrateConfidenceInterval(pos.logRr, pos.seLogRr, em)
g("ci2", np.concatenate([ci2.logRr, ci2.logLb95Rr, ci2.logUb95Rr, ci2.seLogRr]))
tci = ec.computeTraditionalCi(pos.logRr, pos.seLogRr)
g("tci", np.concatenate([tci.rr, tci.lb, tci.ub]))

g("llr.normal", ec.calibrateLlr(null, pos))
p2 = pos.copy(); p2.columns = ["drugName","mu","gamma","sigma"]
g("llr.custom", ec.calibrateLlr(null, p2))
p3 = pos.copy(); p3.columns = ["drugName","mu","alpha","sigma"]
g("llr.skew", ec.calibrateLlr(null, p3))
g("llr.negatives", ec.calibrateLlr(null, neg))
g("pFromLlr", computePFromLlr(np.array([0,0.5,2,10,20,32.9,33.1,50.]), mle=np.ones(8)))
g("pFromLlrLow", computePFromLlr(np.array([0,0.5,2,10.]), mle=-np.ones(4)))
g("llrFromP", [computeLlrFromP(p) for p in [0.6,0.5,0.4,0.05,1e-5,1e-15,1e-16,1e-20]])

ec.set_seed(123)
gd = ec.simulateControls(n=20, mean=0.2, sd=0.2, trueLogRr=0, seLogRr=0.1)
point = np.linspace(np.log(0.1), np.log(10), 200)
grids = [pd.DataFrame({"point": point,
                       "value": rm.dnorm(point, gd.logRr.iloc[i], gd.seLogRr.iloc[i])})
         for i in range(len(gd))]
gnull = ec.fitNullNonNormalLl(grids)
g("gridNull", [gnull[0], gnull[1]])
g("gridLlr", ec.calibrateLlr(gnull, grids[:3]))
g("gridLlApprox", gridLlApproximation(np.array([-3,-0.5,0,0.5,3.]), grids[0]))

ec.set_seed(555)
mn = ec.fitMcmcNull(neg.logRr, neg.seLogRr, iter=2000)
g("mcmcNull", [mn[0], mn[1]])
g("mcmcChainHead", mn.mcmc["chain"][:5, :].ravel())
g("mcmcAcc", np.mean(mn.mcmc["acc"]))
cp = ec.calibrateP(mn, pos.logRr, pos.seLogRr)
g("mcmcCalP", np.concatenate([cp.p, cp.lb95ci, cp.ub95ci]))
e = ec.computeExpectedAbsoluteSystematicError(mn)
g("mcmcEase", [e.ease.iloc[0], e.ciLb.iloc[0], e.ciUb.iloc[0]])

ec.set_seed(777)
n1 = ec.simulateControls(n=30)
rng = ec.get_generator()
n2 = n1.copy(); n2["logRr"] = n2.logRr + rng.rnorm(len(n2), mean=0.1, sd=0.1)
de = ec.compareEase(n1.logRr, n1.seLogRr, n2.logRr, n2.seLogRr, sampleSize=100)
g("compareEase", [de.delta.iloc[0], de.ciLb.iloc[0], de.ciUb.iloc[0], de.p.iloc[0]])
g("compareEase1", de.attrs["ease1"].to_numpy().ravel())
g("compareEase2", de.attrs["ease2"].to_numpy().ravel())

ec.set_seed(303)
ev = ec.evaluateCiCalibration(ctrl.logRr.values[:30], ctrl.seLogRr.values[:30],
                              ctrl.trueLogRr.values[:30])
ev = ev.sort_values(["trueRr", "label", "Confidence interval calculation", "ciWidth"],
                    kind="stable")
g("evalCoverage", ev.coverage); g("evalNrow", len(ev))

ec.set_seed(1); g("cvPois", ec.computeCvPoisson([1.]*10, sampleSize=10000))
ec.set_seed(1); g("cvPoisAlpha", ec.computeCvPoisson([1.]*10, sampleSize=10000).alpha)
ec.set_seed(2); g("cvBinom", ec.computeCvBinomial([1.]*10, z=4, sampleSize=10000))
ec.set_seed(3); g("cvPoisReg", ec.computeCvPoissonRegression([1.]*10, z=4, sampleSize=10000))
ec.set_seed(4); g("cvPoisNull", ec.computeCvPoisson([1.]*10, sampleSize=10000, nullMean=0.2, nullSd=0.2))
ec.set_seed(5); g("cvPoisZero", ec.computeCvPoisson([0.,1,2,3], sampleSize=10000))
ec.set_seed(6); g("samplePois", samplePoissonMaxLrr([1.]*5, 1, 20, 0, 0))
ec.set_seed(7); g("sampleBinom", sampleBinomialMaxLrr([1.]*5, 0.2, 1, 20, 0, 0))
ec.set_seed(8); g("samplePoisReg", samplePoissonRegressionMaxLrr([1.]*5, 4, 1, 20))

ec.set_seed(909)
sm = ec.simulateMaxSprtData(n=200, numberOfNegativeControls=2, numberOfPositiveControls=1)
g("smNrow", len(sm)); g("smTime", sm.time.values[:20]); g("smOutcomeSum", sm.outcome.sum())
g("smExpSum", sm.exposure.sum()); g("smLook", sm.lookTime.values[:5])

sr = ec.datasets.southworthReplication()
sn = sr[sr.trueLogRr == 0]
snull = ec.fitNull(sn.logRr, sn.seLogRr); g("southNull", [snull[0], snull[1]])
grd = ec.datasets.grahamReplication(); grd = grd[~grd.trueLogRr.isna()]
gm = ec.fitSystematicErrorModel(grd.logRr, grd.seLogRr, grd.trueLogRr)
g("grahamModel", gm)

# ---- report
print(f"{'key':<18} {'n':>5} {'maxrel':>11} {'exact':>9}  status")
print("-" * 62)
nfail = 0
for k, exp in G.items():
    exp = np.array([np.nan if v is None else v for v in exp], dtype=float)
    if k not in got:
        print(f"{k:<18} MISSING"); nfail += 1; continue
    v = got[k]
    if v.shape != exp.shape:
        print(f"{k:<18} SHAPE py={v.shape} R={exp.shape}"); nfail += 1; continue
    both_nan = np.isnan(v) & np.isnan(exp)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(exp != 0, np.abs(v - exp) / np.abs(exp), np.abs(v - exp))
    rel = np.where(both_nan, 0.0, rel)
    mx = np.nanmax(rel) if rel.size else 0.0
    nex = int(np.sum((v == exp) | both_nan))
    status = "OK" if mx < 1e-9 else ("CLOSE" if mx < 1e-5 else "FAIL")
    if status == "FAIL": nfail += 1
    print(f"{k:<18} {len(exp):>5} {mx:>11.3g} {nex:>4}/{len(exp):<4} {status}")
print(f"\n{nfail} failing keys")
sys.exit(1 if nfail else 0)
