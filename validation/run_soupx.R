# R SoupX reference run on the inputs written by prepare_hgmm.py.  Usage: Rscript run_soupx.R OUTDIR
suppressPackageStartupMessages(library(Matrix))
if (nzchar(Sys.getenv("SOUPX_SRC"))) source(Sys.getenv("SOUPX_SRC")) else suppressPackageStartupMessages(library(SoupX))
d <- commandArgs(TRUE)[1]
f <- function(...) file.path(d, paste0(...))
rd <- function(n) {
  dm <- scan(f(n, "_dim.txt"), quiet = TRUE)
  sparseMatrix(i = readBin(f(n, "_i.bin"), "integer", dm[3]), p = readBin(f(n, "_p.bin"), "integer", dm[2] + 1),
               x = as.double(readBin(f(n, "_x.bin"), "integer", dm[3])), dims = dm[1:2], index1 = FALSE)
}
wr <- function(m, n) {
  m <- as(as(m, "CsparseMatrix"), "generalMatrix")
  writeBin(m@i, f(n, "_i.bin")); writeBin(m@p, f(n, "_p.bin")); writeBin(m@x, f(n, "_x.bin"))
  writeLines(as.character(c(dim(m), length(m@x))), f(n, "_dim.txt"))
}
genes <- readLines(f("genes.txt")); cells <- readLines(f("cells.txt"))
toc <- rd("toc"); dimnames(toc) <- list(genes, cells)
tod <- rd("tod"); dimnames(tod) <- list(genes, paste0("drop", seq_len(ncol(tod))))
cl <- setNames(readLines(f("clusters.txt")), cells)

sc <- setClusters(SoupChannel(tod, toc), cl)
write.csv(sc$soupProfile, f("r_soup.csv"))
write.csv(quickMarkers(toc, cl, N = Inf), f("r_markers.csv"), row.names = FALSE)
tm <- c(); t <- system.time(sc <- autoEstCont(sc, doPlot = FALSE))[["elapsed"]]; tm["autoEstCont"] <- t
message(sprintf("autoEstCont: rho = %.3f (%.1fs)", sc$fit$rhoEst, t))
write.csv(sc$fit$dd, f("r_dd.csv"), row.names = FALSE)
write.csv(data.frame(post = sc$fit$posterior), f("r_post.csv"), row.names = FALSE)
writeLines(as.character(sc$fit$rhoEst), f("r_rho.txt"))
for (m in c("subtraction", "soupOnly", "multinomial")) {
  set.seed(1)  # multinomial initialises with stochastic rounding
  t <- system.time(out <- adjustCounts(sc, method = m, verbose = 0))[["elapsed"]]
  wr(out, paste0("r_", m)); tm[m] <- t; message(sprintf("adjustCounts %s: %.1fs", m, t))
}
write.csv(data.frame(step = names(tm), seconds = tm), f("r_timings.csv"), row.names = FALSE)
writeLines(c(R.version.string, paste("SoupX", tryCatch(as.character(packageVersion("SoupX")), error = function(e) "(sourced)"))), f("r_session.txt"))
