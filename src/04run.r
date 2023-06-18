################################################################################
##
## File: vWAP-SPY-ModelDev-20091205.R
##
## Purpose:
##  - Estimation of daily-intradaily turnover.
##
## Created: 2008.10.12
##
## Version: 2023.02.26
##
## Author:
##  Fabrizio Cipollini
##
################################################################################


################################################################################
## PRELIMINARIES
################################################################################

 rm(list = ls(all=TRUE))
options(warn = 1)
source("~/CMEM/vWAPMain_20081010.R")


.dummyBin <-
function(binL, include)
{
  ## FUNCTION:

  ##############################################################################
  ## Part 1: PRELIMINARS
  ##############################################################################

  #### Sort
  include <- sort(unique(include))
  bin <- sort(unique(binL))

  #### Remove values greater than nBin
  include <- include[include %in% bin]

  #### Again, returns if include is empty
  if (NROW(include) == 0)
  {
    return(NULL)
  }

  #### Not all values can be included (for collinearity issues)
  if ( (NROW(include) == NROW(bin)) && all(include == bin))
  {
    include <- include[-1] ## First bin as reference
  }


  ##############################################################################
  ## Part 2: MAKE
  ##############################################################################

  #### Make
  x <- A.op.tx(binL, include, "==")

  #### as.numeric
  x <- matrix(data = as.numeric(x), nrow = NROW(binL), ncol = NROW(include))

  #### add colnames
  colnames(x) <- include


  ##############################################################################
  ## Part 3: Answer
  ##############################################################################

  #### Answer
  x
}
# ------------------------------------------------------------------------------


.dummyBin.include <-
function(model)
{
  ## FUNCTION:

  #### Find possible dummy elements
  x <- grep(x = model[,1], pattern = "deltaM", fixed = TRUE, value = TRUE)

  #### Retrieve indexes
  if (NROW(x) > 0)
  {
    x <- gsub(x = x, pattern = "[^[:digit:]]", replacement = "")
    x <- as.numeric(x)
  }
  else
  {
    x <- NULL
  }

  #### Answer
  x
}
# ------------------------------------------------------------------------------


################################################################################
## INPUTS
################################################################################

#### File data


# filein1 <- "~/CMEM/Test_SPY-15m-adj-20091201.txt"
# rangeDate <- c(20020101, 20061231)

# filein1 <- "~/CMEM/A.txt"
# filein1 <- "~/CMEM/AAL.txt"
# rangeDate <- c(20170705, 20171228)



dir_path <- "/Users/kang/CMEM/data/02_r_input/"
# dir_path <- "/Users/kang/CMEM/data/02_r_input_10/"
file_names <- list.files(dir_path)


for (i in seq_along(file_names))
{
  cat("++++++++++++++++++++ i is :", i, "\n")
  filein1 <- paste0(dir_path, file_names[i])
  rangeDate <- c(20170705, 20171228)

  #### Fileout
  fileout1 <- "~/CMEM/SPY-residuals-LBstats-20091201.txt"

  #### Parameters
  model <- c(
              "omegaE",     0.03,
              "alphaE[1]",  0.2,
  #            "alphaE[2]",  0.0,
              "gammaE[1]",  0.0,
              "betaE[1]" ,  0.6,
              "alphaM[1]",  0.3,
  #           "gammaM[1]",  0.0,
  #           "alphaM[2]",  0.0,
              "betaM[1]" ,  0.4,
  #           "deltaM[1]",  0.0,
              "seasF[1]" ,  0.0,
              "seasF[2]" ,  0.0,
              "seasF[3]" ,  0.0,
              "seasF[4]" ,  0.0,
              "seasF[5]" ,  0.0,
              "seasF[6]" ,  0.0,
              "seasF[7]" ,  0.0,
              "seasF[8]" ,  0.0,
              "seasF[9]" ,  0.0,
              "seasF[10]",  0.0,
              "seasF[11]",  0.0,
              "seasF[12]",  0.0,
              "seasF[13]",  0.0,
              "seasF[14]",  0.0,
              "seasF[15]",  0.0,
              "seasF[16]",  0.0,
              "seasF[17]",  0.0,
              "seasF[18]",  0.0,
              "seasF[19]",  0.0,
              "seasF[20]",  0.0,
              "seasF[21]",  0.0,
              "seasF[22]",  0.0,
              "seasF[23]",  0.0,
              "seasF[24]",  0.0,
              "seasF[25]",  0.0
              )
  model <- matrix(data = model, ncol = 2, byrow = TRUE)


  ################################################################################
  ## READ DATA
  ################################################################################

  #### Read
  data1 <- read.table(file = filein1, header = TRUE, sep = "\t", quote = "",
                      na.strings = "..", colClasses = "numeric", comment.char = "")

  #### Only interesting days
  rangeDate <- range(rangeDate)
  date1 <- data1[,"date"]
  ind   <- (rangeDate[1] <= date1 & date1 <= rangeDate[2])
  data1 <- data1[ ind, , drop = FALSE]

  #### Compose data
  nBin  <- max(data1[,"bin"])

  #### Add dummies
  include.dummyBin <- .dummyBin.include(model)
  if ( NROW(include.dummyBin) > 0 )
  {
    data1 <- cbind(data1, .dummyBin(binL = data1[,"bin"], include = include.dummyBin) )
  }


  ################################################################################
  ## ESTIMATES
  ################################################################################

  #### Set diControl
  diControl            <- list()       ## Initialize control settings
  diControl$method     <- "estimation" ## Method
  diControl$nBin       <- nBin         ## Number of bins for each day
  diControl$intraAdj   <- FALSE        ## Intradaily adjustment
  diControl$E$nmTrace    <- 1          ## Eta: Nelder-Mead: trace option
  diControl$E$nmMaxIter  <- 0          ## Eta: Nelder-Mead: maximum fct evaluations
  diControl$E$nrTrace    <- 1          ## Eta: Newton-Raphson: trace any ... iterations
  diControl$E$nrMaxIter  <- 200        ## Eta: Newton-Raphson: maximum iterations
  diControl$E$nrGradTol  <- 1e-2       ## Eta: Newton-Raphson: gradient tolerance
  diControl$M$nmTrace    <- 1          ## Mu: Nelder-Mead: trace option
  diControl$M$nmMaxIter  <- 0          ## Mu: Nelder-Mead: maximum fct evaluations
  diControl$M$nrTrace    <- 1          ## Mu: Newton-Raphson: trace any ... iterations
  diControl$M$nrMaxIter  <- 20         ## Mu: Newton-Raphson: maximum iterations
  diControl$M$nrGradTol  <- 1e-2       ## Mu: Newton-Raphson: gradient tolerance
  diControl$EM$nmTrace   <- 1          ## EtaMu: Nelder-Mead: trace option
  diControl$EM$nmMaxIter <- 0          ## EtaMu: Nelder-Mead: maximum fct evaluations
  diControl$EM$nrTrace   <- 1          ## EtaMu: Newton-Raphson: trace any ... iterations
  diControl$EM$nrMaxIter <- 100        ## EtaMu: Newton-Raphson: maximum iterations
  diControl$EM$nrGradTol <- 1e-3       ## EtaMu: Newton-Raphson: gradient tolerance

  #### Estimation
  inference <- vWAP(data1, model, diControl)
  inference <- inference$inference
  #.print.diMEM(inference, diControl, fileout = "")


  ################################################################################
  ## FORECASTS
  ################################################################################

  #### Set diControl
  diControl$method     <- "forecast" ## Method
  # diControl$nDFor      <- c(400,402) ## Days at which to make forecasts
  # diControl$nDFor      <- c(10,120) ## Days at which to make forecasts
  # diControl$nDFor      <- c(120,122) ## Days at which to make forecasts
  diControl$nDFor      <- c(2,123) ## Days at which to make forecasts

  #### Set model based on estimates
  modelFor <- cbind(inference$parmName, inference$parmEst, deparse.level = 0)

  #### Forecasts
  forecasts <- vWAP(data1, modelFor, diControl)
  df <- do.call(rbind, forecasts)
  # write the all_forecasts data frame to a single file
  # out_dir_path <- "/Users/kang/CMEM/r_output/r_output_raw_data/"
  # out_dir_path <- "/Users/kang/CMEM/r_output/r_output_raw_data_10/"
  out_dir_path <- "/Users/kang/CMEM/r_output/04_r_output_raw_data/"
  filename <- paste0(out_dir_path, "forecasts_", substr(file_names[i], 1,nchar(file_names[i])-4), ".csv")
  write.table(df, file = filename, sep = ",", col.names = FALSE, append = FALSE)
}


stop("Stop here")


################################################################################
## DIAGNOSTICS
################################################################################

#### Data settings
timeL   <- index(data$x)
x       <- as.numeric(data$x)

#### indices
nBin  <- diControl$nBin
calendar <- .calendar(timeL, nBin)
binL <- calendar$binL
bin <- calendar$bin


#### Plot diagnostics
.plotDiagnostics(x, inference, calendar)


################################################################################
## LJUNG-BOX STATISTICS
################################################################################

#### Time series decomposition
xDec <- .decomposeSeries(inference$residuals, calendar)

#### Selected lags
lags.x   <- c(1, 7, 14, 66)
lags.d   <- c(1, 5, 10, 20)

#### Selected info
colSel <- c("lag","stat")

#### Overall time series
tmp    <- .compute.portmanteau(epsMatrix = xDec$x, parmVal = NULL, lagMax = 100)
lb.x   <- cbind( type = "x", tmp[ tmp[,"lag"] %in% lags.x, colSel] )

#### Daily time series
tmp    <- .compute.portmanteau(epsMatrix = xDec$eta, parmVal = NULL, lagMax = 100)
lb.eta <- cbind( type = "eta", tmp[ tmp[,"lag"] %in% lags.d, colSel] )

#### Intradaily periodic time series
tmp    <- .compute.portmanteau(epsMatrix = xDec$seasMu, parmVal = NULL, lagMax = 100)
lb.seasMu <- cbind( type = "seasMu", tmp[ tmp[,"lag"] %in% lags.x, colSel] )

#### Intradaily non-periodic time series
tmp   <- .compute.portmanteau(epsMatrix = xDec$mu, parmVal = NULL, lagMax = 100)
lb.mu <- cbind( type = "mu", tmp[ tmp[,"lag"] %in% lags.x, colSel] )

#### Compose
out   <- rbind( lb.x, lb.eta, lb.seasMu, lb.mu )
write.table(x = out, file = fileout1, quote = FALSE, sep = "\t", row.names = FALSE)

