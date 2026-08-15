library(EmpiricalCalibration)
suppressWarnings(suppressMessages({
data(sccs); data(caseControl); data(cohortMethod)
data(southworthReplication); data(grahamReplication)
}))
out <- list()
g <- function(k, v) out[[k]] <<- as.numeric(v)

neg <- sccs[sccs$groundTruth == 0, ]; pos <- sccs[sccs$groundTruth == 1, ]
null <- fitNull(neg$logRr, neg$seLogRr)
g("null", c(null[1], null[2]))
g("ease", computeExpectedAbsoluteSystematicError(null))

# --- systematic error model
set.seed(101)
ctrl <- simulateControls(n = 50*3, mean = 0.25, sd = 0.25, trueLogRr = log(c(1,2,4)))
g("sim.logRr", ctrl$logRr)
g("sim.seLogRr", ctrl$seLogRr)
g("sim.trueLogRr", ctrl$trueLogRr)
m <- fitSystematicErrorModel(ctrl$logRr, ctrl$seLogRr, ctrl$trueLogRr)
g("model", as.numeric(m))
ml <- fitSystematicErrorModel(ctrl$logRr, ctrl$seLogRr, ctrl$trueLogRr, legacy = TRUE)
g("modelLegacy", as.numeric(ml))
mc <- fitSystematicErrorModel(ctrl$logRr, ctrl$seLogRr, ctrl$trueLogRr,
                              estimateCovarianceMatrix = TRUE)
g("modelLB95", attr(mc, "LB95CI")); g("modelUB95", attr(mc, "UB95CI"))
g("modelCov", as.numeric(attr(mc, "CovarianceMatrix")))

set.seed(202)
nd <- simulateControls(n = 15, mean = 0.25, sd = 0.25, trueLogRr = log(c(1,2,4)))
ci <- calibrateConfidenceInterval(nd$logRr, nd$seLogRr, m)
g("ci.logRr", ci$logRr); g("ci.lb", ci$logLb95Rr); g("ci.ub", ci$logUb95Rr); g("ci.se", ci$seLogRr)
cil <- calibrateConfidenceInterval(nd$logRr, nd$seLogRr, ml)
g("ciL.logRr", cil$logRr); g("ciL.lb", cil$logLb95Rr); g("ciL.ub", cil$logUb95Rr)

em <- convertNullToErrorModel(null)
g("errModel", as.numeric(em))
ci2 <- calibrateConfidenceInterval(pos$logRr, pos$seLogRr, em)
g("ci2", c(ci2$logRr, ci2$logLb95Rr, ci2$logUb95Rr, ci2$seLogRr))
tci <- computeTraditionalCi(pos$logRr, pos$seLogRr)
g("tci", c(tci$rr, tci$lb, tci$ub))

# --- LLR calibration
g("llr.normal", calibrateLlr(null, pos))
p2 <- pos; colnames(p2) <- c("drugName","mu","gamma","sigma")
g("llr.custom", calibrateLlr(null, p2))
p3 <- pos; colnames(p3) <- c("drugName","mu","alpha","sigma")
g("llr.skew", calibrateLlr(null, p3))
g("llr.negatives", calibrateLlr(null, neg))
g("pFromLlr", EmpiricalCalibration:::computePFromLlr(c(0,0.5,2,10,20,32.9,33.1,50), mle = rep(1, 8)))
g("pFromLlrLow", EmpiricalCalibration:::computePFromLlr(c(0,0.5,2,10), mle = rep(-1, 4)))
g("llrFromP", sapply(c(0.6,0.5,0.4,0.05,1e-5,1e-15,1e-16,1e-20), EmpiricalCalibration:::computeLlrFromP))

# --- grid likelihood
set.seed(123)
gd <- simulateControls(n = 20, mean = 0.2, sd = 0.2, trueLogRr = 0, seLogRr = 0.1)
point <- seq(log(0.1), log(10), length.out = 200)
grids <- lapply(split(gd, 1:nrow(gd)), function(row)
  data.frame(point = point, value = dnorm(point, mean = row$logRr, sd = row$seLogRr)))
gnull <- fitNullNonNormalLl(grids)
g("gridNull", c(gnull[1], gnull[2]))
g("gridLlr", calibrateLlr(gnull, grids[1:3]))
g("gridLlApprox", EmpiricalCalibration:::gridLlApproximation(
   c(-3, -0.5, 0, 0.5, 3), grids[[1]]))

# --- MCMC (fixed seed)
set.seed(555)
mn <- fitMcmcNull(neg$logRr, neg$seLogRr, iter = 2000)
g("mcmcNull", c(mn[1], mn[2]))
g("mcmcChainHead", as.numeric(t(attr(mn,"mcmc")$chain[1:5, ])))
g("mcmcAcc", mean(attr(mn,"mcmc")$acc))
cp <- calibrateP(mn, pos$logRr, pos$seLogRr)
g("mcmcCalP", c(cp$p, cp$lb95ci, cp$ub95ci))
e <- computeExpectedAbsoluteSystematicError(mn)
g("mcmcEase", c(e$ease, e$ciLb, e$ciUb))

# --- compareEase
set.seed(777)
n1 <- simulateControls(n = 30)
n2 <- n1; n2$logRr <- n2$logRr + rnorm(nrow(n2), mean = 0.1, sd = 0.1)
de <- compareEase(n1$logRr, n1$seLogRr, n2$logRr, n2$seLogRr, sampleSize = 100)
g("compareEase", c(de$delta, de$ciLb, de$ciUb, de$p))
g("compareEase1", as.numeric(attr(de,"ease1")))
g("compareEase2", as.numeric(attr(de,"ease2")))

# --- evaluation
set.seed(303)
ev <- evaluateCiCalibration(ctrl$logRr[1:30], ctrl$seLogRr[1:30], ctrl$trueLogRr[1:30])
ev <- ev[order(ev$trueRr, ev$label, ev$`Confidence interval calculation`, ev$ciWidth), ]
g("evalCoverage", ev$coverage)
g("evalNrow", nrow(ev))

# --- MaxSPRT
set.seed(1); g("cvPois", computeCvPoisson(rep(1,10), sampleSize = 10000))
set.seed(1); g("cvPoisAlpha", attr(computeCvPoisson(rep(1,10), sampleSize = 10000), "alpha"))
set.seed(2); g("cvBinom", computeCvBinomial(rep(1,10), z = 4, sampleSize = 10000))
set.seed(3); g("cvPoisReg", computeCvPoissonRegression(rep(1,10), z = 4, sampleSize = 10000))
set.seed(4); g("cvPoisNull", computeCvPoisson(rep(1,10), sampleSize = 10000, nullMean = 0.2, nullSd = 0.2))
set.seed(5); g("cvPoisZero", computeCvPoisson(c(0,1,2,3), sampleSize = 10000))
set.seed(6); g("samplePois", EmpiricalCalibration:::samplePoissonMaxLrr(rep(1,5), 1, 20, 0, 0))
set.seed(7); g("sampleBinom", EmpiricalCalibration:::sampleBinomialMaxLrr(rep(1,5), 0.2, 1, 20, 0, 0))
set.seed(8); g("samplePoisReg", EmpiricalCalibration:::samplePoissonRegressionMaxLrr(rep(1,5), 4, 1, 20))

# --- simulateMaxSprtData
set.seed(909)
sm <- simulateMaxSprtData(n = 200, numberOfNegativeControls = 2, numberOfPositiveControls = 1)
g("smNrow", nrow(sm)); g("smTime", head(sm$time, 20)); g("smOutcomeSum", sum(sm$outcome))
g("smExpSum", sum(sm$exposure)); g("smLook", head(sm$lookTime, 5))

# --- southworth / graham
sn <- southworthReplication[southworthReplication$trueLogRr == 0 &
                            !is.na(southworthReplication$trueLogRr), ]
snull <- fitNull(sn$logRr, sn$seLogRr); g("southNull", c(snull[1], snull[2]))
gr <- grahamReplication[!is.na(grahamReplication$trueLogRr), ]
gm <- fitSystematicErrorModel(gr$logRr, gr$seLogRr, gr$trueLogRr)
g("grahamModel", as.numeric(gm))

con <- file("gold.json", "w")
writeLines("{", con)
keys <- names(out)
for (i in seq_along(keys)) {
  v <- out[[keys[i]]]
  s <- ifelse(is.na(v), "null", sprintf("%.17g", v))
  writeLines(sprintf('"%s": [%s]%s', keys[i], paste(s, collapse=", "),
                     if (i < length(keys)) "," else ""), con)
}
writeLines("}", con)
close(con)
cat("wrote", length(keys), "entries\n")
