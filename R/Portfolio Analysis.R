## 1. LOAD ALL LIBRARIES RELEVANT TO THIS PORTFOLIO ASSIGNMENT. 
library(zoo)
library(reshape2)
library(plyr)
library(abind)
library(data.table) 
library(dplyr)

## 2. LOAD ALL DATA RELEVANT TO THIS PORTFOLIO ASSIGNMENT.
#open the data files
factor.5 <- read.csv("factors2022-23-exam.csv",head=TRUE)
assets.25 <- read.csv("testassets2022-23-exam.csv",head=TRUE)

## ------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------

#remove the unnecessary column
assets.25 <- select(assets.25, -X)
factor.5 <- select(factor.5, -X)

#creating the table with the size-deciles in the rows, and the B/M-deciles in columns.
col.var <- expand.grid(1:5,1:5) ##(first number = ME, second number = Book to Market)
col.var <- paste(col.var[,1],col.var[,2],sep=".")

colnames(assets.25) <- c("month",col.var)
assets.25$month <- as.Date(as.yearmon(as.character(assets.25$month),format="%Y%m"))
assets.25 <- zoo(assets.25[,2:NCOL(assets.25)],assets.25$month)

colnames(factor.5) <- c("month","rm.rf","smb","hml","rmw","cma","rf")
factor.5$month <- as.Date(as.yearmon(as.character(factor.5$month),format="%Y%m"))
factor.5 <- zoo(factor.5[,2:NCOL(factor.5)],factor.5$month)

#function to create a table to store avg returns, std. dev and sharpe ratio
pretty.table <- function(x){
  ## Assume names(x) is 1.1, ... ,10.10
  z <- strsplit(names(x),split="[.]")
  z <- data.frame(do.call("rbind",z))
  z <- apply(z,2,function(x) as.numeric(as.character(x)))
  colnames(z) <- c("column","rows")
  z <- data.frame(z)
  z$values <- x
  z$values <- round(z$values,3)
  res <- acast(z,rows ~ column, value.var="values")
  return(res)
}

## 3. PART A

## cut the data for the factor model and assets from 1963-07 to 2021-10 
assets.25 <- window(assets.25,start="1963-07-01",end="2021-10-01") 
factor.5 <- window(factor.5,start="1963-07-01",end="2021-10-01") 

## 1(a) Average monthly returns
avg.ret <- apply(window(assets.25,start="1963-07-01",end="2021-10-01"),2,mean,na.rm=TRUE)
avg.ret <- pretty.table(avg.ret)


## 1(b) Std. Dev and sharpe.rat Ratio calculations
std.dev <- apply(window(assets.25,start="1963-07-01",end="2021-10-01"),2,sd,na.rm=TRUE)
std.dev <- pretty.table(std.dev)

# compute risk-free rate 
rf <-factor.5$rf 

# compute excess returns
eret <- assets.25 - rf  

sharpe.rat <- apply(eret,2,function(x){mean(x,na.rm=TRUE)/sd(x,na.rm=TRUE)})
sharpe.rat <- pretty.table(sharpe.rat)
sharpe.rat


#function to create the beta & GRS function
run.factor.model <- function(lhs,rhs){
  alphas <- NULL
  betas <- NULL
  resids <- zoo(rep(NA,nrow(lhs)),index(lhs))
  
  ## Check they are of the same window
  start.win <- max(start(lhs),start(rhs))
  end.win <- min(end(lhs),end(rhs))
  
  lhs <- window(lhs,start=start.win,end=end.win)
  rhs <- window(rhs,start=start.win,end=end.win)
  
  ## Store residuals and alphas from regressions
  for(i in 1:NCOL(lhs)){
    m <- lm(lhs[,i] ~ rhs)
    alphas <- c(alphas,coef(m)[1])
    b <- coef(m)[2:length(coef(m))]
    betas <- rbind(betas,b)
    resids <- merge(resids,residuals(m),all=TRUE)
    colnames(resids) <- NULL
  }
  
  resids <- resids[,2:NCOL(resids)]
  colnames(resids) <- names(alphas) <- 1:NCOL(lhs)
  colnames(betas) <- colnames(rhs)
  rownames(betas) <- 1:NCOL(lhs)
  
  ## Various variables for the GRS statistic
  T <- nrow(lhs)
  N <- ncol(lhs)
  K <- ncol(rhs)
  Sigma <- var(resids,na.rm=TRUE)
  demeaned <- apply(rhs,2,function(x){x - mean(x,na.rm=TRUE)})
  Omega <- (t(demeaned)%*%demeaned)/T
  f <- apply(rhs,2,mean,na.rm=TRUE)
  GRS <- ((T-N-K)/N)*solve(1+t(f)%*%solve(Omega)%*%f)*t(alphas)%*%solve(Sigma)%*%alphas
  GRS <- as.numeric(GRS)
  
  #compute the p-value for each GRS statistic
  p.value <- pf(GRS, N, T-N-1, 0, lower.tail = FALSE)
  p.value <- as.numeric(p.value)
  finres <- list(beta=betas,residuals=resids,GRS=GRS,pval=p.value,T=T,N=N,K=K,Sigma=Sigma,Omega=Omega)
  return(finres)
}

## 1(c.i) FMB Regression Stage 1

#splitting the data for 5 factors
rhs.f5 <- factor.5[,1:5]

#splitting the data for 3 factors
rhs.f3 <- factor.5[,1:3]

#getting the GRS for 5 factors
GRS.f5 <- run.factor.model(lhs=eret,rhs=rhs.f5)

#getting the GRS for 3 factors
GRS.f3 <- run.factor.model(lhs=eret,rhs=rhs.f3)

#function to calculate risk premia and t-statistics
second.step <- function(lhs,rhs,mu.m,var.m){
  lambda <- NULL
  for(i in 1:NCOL(lhs)){
    m <- lm(lhs[,i] ~ rhs)
    tmp <- t(summary(m)$coefficients[,c(1,3)])
    lambda <- rbind(lambda,tmp[1,])
  }
  
  c.alpha <- lambda[,1]
  lambda <- lambda[,2:NCOL(lambda)]
  mean.alpha <- mean(c.alpha)
  mean.lambda <- apply(lambda,2,mean)
  
  var.alpha <- (1/(NCOL(lhs)*(NCOL(lhs)-1)))*sum((c.alpha - mean.alpha)^2)
  var.lambda <- (1/(NCOL(lhs)*(NCOL(lhs)-1)))*apply((lambda - mean.lambda)^2,2,sum)
  tstats <- c(mean.alpha,mean.lambda)/c(sqrt(var.alpha),sqrt(var.lambda))
  df = NCOL(lhs)-1
  
  #compute the p-value for each t statistic
  p.value <- pmin(2*pt(q=tstats,df=df,lower.tail = FALSE),1)
  p.value <- as.numeric(p.value)
  
  res <- rbind(c(mean.alpha,mean.lambda),tstats,p.value)
  
  rownames(res) <- c("lambda","t(lambda)","p-value")
  colnames(res) <- c("alpha",colnames(res)[2:NCOL(res)])
  return(res)
}

## 1(c.ii) FMB Regression Stage 2

#getting the lambda and its t-statistic for each of the 5 factors
fmb.f5 <- second.step(lhs=t(eret),rhs= GRS.f5$beta)

#getting the lambda and its t-statistic for each of the 3 factors
fmb.f3 <- second.step(lhs=t(eret),rhs= GRS.f3$beta)


## 2(a) Rolling factor betas (3-year).
#setting up the conditions needed to run a rolling window 
window_size <- 36
num <- (nrow(rhs.f3) - window_size + 1)
window_beta_upd <- array(0, c(25, 3, num))

#function to compute the betas at each rolling window
for (i in 1:num) {
  window_data.rhs <- rhs.f3[i:(i + window_size - 1),]
  window_data.eret <- eret[i:(i + window_size - 1),]
  
  window_beta <- run.factor.model(window_data.eret,window_data.rhs)
  collect.beta <- window_beta$beta
  
  window_beta_upd[,,i] <- collect.beta
}

window_start <- window_beta_upd[,,1]
window_diff <- window_beta_upd[,,1]-window_beta_upd[,,665]

## 2(b) FMB Regression Stage 2, predictive estimates.
model.fit <- function(lhs,rhs){
  predval <- list()
  for(i in 37:NCOL(lhs)){
    m <- lm(lhs[,i] ~ rhs[,,i-36], na.action = na.exclude)
    predval[[i-36]] <- predict(m)
  }
  predictions <- do.call("rbind",predval)
  corr.table <- cor(predictions,t(lhs)[37:NROW(eret)])
  
  predictions.mean <- apply(predictions,2,mean,na.rm=TRUE)
  actuals <- apply(lhs,1,mean,na.rm=TRUE)
  finres <- list(df=data.frame(actuals,predictions.mean),cor=corr.table)
  return(finres)
}

p1 <- model.fit(lhs=t(eret),rhs=window_beta_upd)
corr.table  <- matrix(round(t(p1$cor),3),ncol=5,nrow=5,byrow=TRUE)

#assembling the array as the 5x5 table
colnames(corr.table) <- c(1:5)
rownames(corr.table) <- c(1:5)



data <- read.csv("prediction-exercise-2022-23.csv",head=TRUE)
data <- select(data,-X)

variables <- c("D12","E12","b.m","tbl","AAA","BAA","lty","ntis","Rfree","infl","ltr","corpr","svar","csp")

for(i in variables) data[, paste0(i,"_lag")] <- shift(data[,i],n=1,type="lag")

data <- data[-1:-2,]

## 3 (a)
mean.returns <- rep(NA,1174) 
for(t in 1175:nrow(data)){
  ## Compute mean until data point t-1: 
  mr <- mean(data$return[1:(t-1)],na.rm=TRUE)
  mean.returns <- c(mean.returns,mr)
}

data$mean.return <- mean.returns
data$oos_meanret <- (data$return - data$mean.return)^2

data$yyyymm <- as.Date(as.yearmon(as.character(data$yyyymm),format="%Y%m"))
data <- zoo(data[,2:NCOL(data)],data$yyyymm)

forecast.model <- function(y,x,initial.sample.end="2019-02-01"){
  fulldat <- merge(y,x)
  months <- index(fulldat)[120:nrow(fulldat)]
  
  oos.forecasterror <- as.numeric(rep(NA,(nrow(fulldat)-length((months)))))
  
  for(t in 1:length(months)){
    
    ## create the dataset for regression:
    tmp.d <- window(fulldat,end=(as.yearmon(months[t])-1/12))
    
    # Run a linear regression:
    m <- lm(y ~ x,data=tmp.d)
    
    ## Predict return for time t:
    pt <- data.frame("x"= x[months[t]])
    haty <- predict(m,newdata=pt)
    
    ## Compute out of sample forecast error: 
    oose <- as.numeric(y[as.character(months[t])]-haty)
    oos.forecasterror <- c(oos.forecasterror,oose)
  }
  return(oos.forecasterror^2) 
}

predictors <- c("tbl_lag","AAA_lag","lty_lag","ntis_lag","Rfree_lag")

plot.vars <- paste0(predictors,"_perf")

results <- NULL 
for(i in predictors){
  cat("Predicting using:",i,"\n") 
  tmp <- forecast.model(y=data$return,x=data[,i])
  results <- cbind(results,tmp)
}

colnames(results) <- paste0(predictors,"_oosse")
data <- merge(data,results)

perf <- NULL
for(i in predictors){
  tmp <- c(rep(NA,1174),cumsum(na.omit(data$oos_meanret - na.omit(data[,paste0(i,"_oosse")]))))
  perf <- cbind(perf,tmp)
}

colnames(perf) <- predictors
perf <- zoo(perf,order.by=index(data))

toplot <- na.omit(perf) 

for(i in 1:length(plot.vars)){
  plot(toplot[,i],
       xlab="",ylab="Cumulative SSE Difference",
       main=paste0("Variable: ",plot.vars[i]))
  grid()
} 

overall.diff <- rowSums(toplot)
overall.diff <- zoo(overall.diff,order.by=index(toplot))

plot(overall.diff,xlab="Time",ylab="Cumulative SSE Difference",main=paste0("Model (tbl_lag, AAA_lag, lty_lag, ntis_lag, Rfree_lag)"),type="l")

pdf("model1.pdf") ## Store your best 1 month model cumulative SSE here. 

#GRAPH HERE
plot(overall.diff,xlab="Time",ylab="Cumulative SSE Difference",main=paste0("Model (tbl_lag, AAA_lag, lty_lag, ntis_lag, Rfree_lag)"),type="l")

dev.off()

