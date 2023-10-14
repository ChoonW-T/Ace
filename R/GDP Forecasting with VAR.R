library(vars)
library(mFilter)
library(forecast)
library(tseries)
library(tidyverse)

okun <- read.csv("GDP_dataset.csv",head=TRUE)

ggplot(data = okun) + geom_point(mapping = aes(x = unem, y = real_gdp_growth))

gdp <- ts(okun$real_gdp_growth, start= c(1993,3), frequency = 4)
unem <- ts(okun$unem, start = c(1993,3), frequency=4)

autoplot(cbind(gdp,unem))

OLS1 <- lm(gdp ~ unem)

acf(gdp, main = "ACF for Real GDP Growth")
pacf(gdp, main = "PACF for Real GDP Growth")

acf(unem, main = "ACF for Unemployment")
pacf(unem, main = "PACF for Unemployment")

okun.bv <- cbind(gdp,unem)
colnames(okun.bv) <- cbind("GDP","Unemployment")

logselect <- VARselect(okun.bv, lag.max=10, type="const")
logselect$selection

modelOkun <- VAR(okun.bv, p=4, type = "const", season = NULL, exog= NULL)
summary(modelOkun)

Serial <- serial.test(modelOkun, lags.pt=12, type="PT.asymptotic")
Serial

Arch <- arch.test(modelOkun, lags.multi=12, multivariate.only = TRUE)
Arch

Norm <- normality.test(modelOkun, multivariate.only = TRUE)
Norm

Stability <- stability(modelOkun, type = "OLS-CUSUM")
plot(Stability)

GrangerGDP <- causality(modelOkun, cause = "GDP")
GrangerGDP

GrangerUnemployment <- causality(modelOkun, cause = "Unemployment")
GrangerUnemployment

GDPirf <- irf(modelOkun, impulse="Unemployment", response="GDP", n.ahead=20, boot=TRUE)
plot(GDPirf, ylab="GDP",main="Shock from Unemployment")

Unemploymentirf <- irf(modelOkun, impulse="GDP", response="Unemployment", n.ahead=20, boot=TRUE)
plot(Unemploymentirf, ylab="Unemployment",main="Shock from GDP")

FEVD <- fevd(modelOkun, n.ahead=10)
plot(FEVD)

forecast <- predict(modelOkun, n.ahead=4, ci=0.95)
fanchart(forecast, names="GDP")
fanchart(forecast, names="Unemployment")

