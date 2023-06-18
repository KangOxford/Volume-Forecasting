################################################################################
##
## File: vWAPMain_20081010.R
##
## Purpose:
##  R main for handling daily-intradaily data.
##
## Created: 2008.10.10
##
## Version: 2009.12.05
##
## Author:
##  Fabrizio Cipollini <cipollini@ds.unifi.it>
##
################################################################################

################################################################################
## LOAD FUNCTIONS
################################################################################

#### Formulation functions
# source("~/Cipo/Volatility/package/vWAP/R/dev/vWAPFormulation_20081010.R")
source("~/CMEM/vWAPFormulation-20091205.R")

#### Filtering functions
# source("~/Cipo/Volatility/package/vWAP/R/dev/vWAPFiltering_20081023.R")
source("~/CMEM/vWAPFiltering-20091205.R")

#### Estimation functions
#source("~/Cipo/Volatility/package/vWAP/R/dev/vWAPEstimation_20081120.R")
source("~/CMEM/vWAPEstimation-20091205.R")

#### Forecasts functions
#source("~/Cipo/Volatility/package/vWAP/R/dev/vWAPForecast_20081108.R")
source("~/CMEM/vWAPForecast-20091210.R")

#### Time functions
#source("~/Cipo/Volatility/package/vWAP/R/dev/vWAPTime_20090301.R")
source("~/CMEM/vWAPTime-20091205.R")

#### Utilities functions
source("~/CMEM/vWAPUtilities_20081220.R")


################################################################################
## LOAD GLOBAL CONSTANTS and PACKAGES
################################################################################

require(xts)
.constants()


################################################################################
## MAIN FUNCTION
################################################################################

vWAP <-
function(x, model, diControl, fileout = "")
{
  ##############################################################################
  ## Description:
  ##  Simulate or estimate a daily-intradaily MEM.
  ##
  ## Arguments:
  ##  x: (data.frame) time series of data or time series of simulated residuals 
  ##   (if diControl$method == "simulation"). It must have at least columns 
  ##   'date' and 'bin' in order to convert it in a xts object.
  ##  model: (character) a matrix with two columns:
  ##   - "parm": specifies parameters in the model in this form:
  ##     - "omegaE": constant of the daily component;
  ##     - "alphaE[l]": MA part of the daily component at lag l;
  ##     - "betaE[l]": AR part of the daily component at lag l;
  ##     - "alphaM[l]": MA part of the intradaily component at lag l;
  ##     - "betaM[l]": AR part of the intradaily component at lag l;
  ##     - "deltaM[l]": Intradaily dummy for lag l;
  ##     - "seasF[i]": i-th seasonal parameter for the intradaily periodic 
  ##       component.
  ##   - "start": specifies starting values for each one of the parameters.
  ##   If colnames are not provided, parameters must be specified in the first
  ##   column, starting values in the second one.
  ##  diControl: (list) model and estimation control settings. See functions
  ##   .diControl(), .nrControl(), .nmControl() for details.
  ##  fileout: (character) output filename.
  ##
  ## Value:
  ##  Controlled by the option 'diControl$method':
  ##   - if "settings", returns 'diControl' settings only;
  ##   - if "simulation", returns a simulated time series;
  ##   - if "estimation", returns inferences from the daily-intradaily MEM;
  ##   - if "forecast", returns forecasts from the daily-intradaily MEM.
  ##
  ## Author:
  ##  Fabrizio Cipollini
  ##############################################################################

  ## FUNCTION:
  
  ##############################################################################
  ## Part 1: Settings.
  ##############################################################################

  #### Adjust 'model'
  model <- .extract.model(model)
  model <- .check.model.internal(model)

  #### Adjust 'diControl'
  diControl <- .diControl(diControl)

  #### Adjust 'model' again
  model <- .check.model.vs.diControl(model, diControl)
  model <- .sort.model(x = model, vars = c("parm", "lag"))

  #### Extract
  parms   <- model[,"parm"]
  lags    <- model[,"lag"]
  parmVal <- model[,"start"]

  #### Make infoFilter
  infoFilter <- .infoFilter(parms, lags, diControl)
  
  #### Transform data to an xts object
  x <- .data.2.xts(x)


  ##############################################################################
  ## Part 2: Make.
  ##############################################################################

  #### Settings only
  if (diControl$method == "settings")
  {
    return(diControl)
  }

  #### Simulation
  else if (diControl$method == "simulation")
  {
    #### Simulate
    x <- .r.diMEM(parmVal, infoFilter, x, diControl)

    #### Answer
    return( x )
  }

  #### Estimation
  else if (diControl$method == "estimation")
  {
    #### Make inferences
    inference <- .fit.diMEM( parmVal, infoFilter, x, diControl )

    #### Print inferences
    .print.diMEM(inference, diControl, fileout)

    #### Return
    return( list(diControl = diControl, inference = inference) )
  }

  #### Forecasts
  else if (diControl$method == "forecast")
  {
    #### Make forecasts
    return( .forecast.diMEM( parmVal, infoFilter, x, diControl ) )
  }


  ##############################################################################
  ## Part 3: Answer.
  ##############################################################################

  NULL
}
# ------------------------------------------------------------------------------
