
vselect<-function(X,Y,alpha=0.05,p_total=0,
                  selmtd="dlasso",
                  precmtd="sqrtlasso",
                  FCD=TRUE){
  p = dim(X)[2]
  n = dim(X)[1]
  # Debiased lasso
  if (selmtd=="dlasso"){
    # obtain debiased lasso estimate, recycle Mhat if possible
    dblt = debiased_lasso(X,Y,precmtd = precmtd,nlam=100)
    Ts = dblt$T_i
    if (FCD==FALSE){
      # select by individual tests
      return(list(selected = (2*(1-pnorm(abs(Ts)))<=alpha),
                  res = Y-X%*%dblt$coef))
    }
    # FCD procedure:
    t_p = sqrt(2*log(p)-2*log(log(p)))
    ts = sort(abs(Ts)[abs(Ts)<=t_p])
    ps = 2*p*(1-pnorm(ts))/
      pmax(1,sapply(ts, function(l)sum(abs(Ts)>l)))
    # compute FCD cutoff
    if (!length(which(ps<=(alpha)))){
      t0 <- sqrt(2*log(p))
    } else {
      t0 <- max(ts[which(ps<=(alpha))])
    }
    return(list(selected = (abs(Ts)>=t0),
                res = Y-X%*%dblt$coef))
  } else if (selmtd=="lasso"){
    # lasso with fixed penalty
    p_total=ifelse(p_total>0,p_total,p)
    lam = 1/sqrt(n)*qnorm(1-alpha/2/p_total/(p-1))
    lassom = coefficients(glmnet::glmnet(X,Y,lambda=lam))[-1]
    return(list(selected = (lassom!=0),res = Y-X%*%lassom))
  } else if (selmtd=="cvlasso"){
    # lasso with cv penalty
    lassom = coefficients(glmnet::cv.glmnet(X,Y))[-1]
    return(list(selected = (lassom!=0),res = Y-X%*%lassom))
  } else if (selmtd=="scallasso"){
    # scaledlasso
    lassom = coef(scalreg::scalreg(X,Y))
    return(list(selected = (lassom!=0),res = Y-X%*%lassom))
  } else if (selmtd=="adalasso"){
    # adaptive lasso with fixed penalty initialization
    p_total=ifelse(p_total>0,p_total,p)
    lam = 1/sqrt(n)*qnorm(1-alpha/2/p_total/(p-1))
    lassom = coefficients(glmnet::glmnet(X,Y,lambda=lam))[-1]
    if (all(lassom==0)){
      return(list(selected=rep(0,p), res=Y))
    } else {
      lassom = coefficients(
        glmnet::glmnet(X,Y,lambda=lam,
                       penalty.factor =
                         pmax(1,1/abs(lassom)!=0)))[-1]
      return(list(selected = (lassom!=0),res = Y-X%*%lassom))
    }
  }
}

debiased_lasso<-function(X,Y,precmtd="cv",nlam=100){
  # Compute debiased lasso fit
  # using the algorithm from Van der Geer 2014
  # X: n x p Y: 1 x p
  # precmtd: chooses how to compute the debiasing matrix:
  #   "cv": joint cross-validated node-wise lasso
  #   "sqrtlasso": square-root lasso (no tune)
  # return coefficients, test statistics for variable selection
  p = dim(X)[2]
  n = dim(X)[1]
  # Compute debiasing matrix
  if (p>2){
    if (precmtd=="cv"){
      # lambda_max, implemented same as in glmnet
      lmax = max(sapply(1:p, function(i)
        max( abs(t(X[,i] - mean(X[,i])*
                     (1-mean(X[,i]))) %*% X[,-i] ) )/n))
      # lambda sequence
      lams =
        exp(seq(from = log(0.001*lmax),
                to = log(lmax),length.out = nlam))
      # CV for best lambda
      lamj = lams[which.min(rowSums(
        sapply(1:p, function(i)
          glmnet::cv.glmnet(X[,-i],X[,i],lambda=lams)$cvm)))]
      # matrix of coefs
      ghat = sapply(1:p, function(i)
        coefficients(glmnet::glmnet(X[,-i],X[,i],
                                    lambda=lamj))[-1])
    } else if (precmtd == "sqrtlasso"){
      ghat = sapply(1:p, function(i)
        RPtests::sqrt_lasso(X[,-i],X[,i]))
    }
    # construct debiasing matrix as in VdG2014
    Chat = diag(rep(1,p))
    for (i in 1:p){Chat[i,-i]=-ghat[,i]}
    tauhat = sapply(1:p, function(i)
      t((X[,i]-X[,-i]%*%
           ghat[,i]))%*%X[,i]/n)
    Mhat = diag(1/tauhat)%*%Chat
  } else {
    Mhat = solve(cov(X))
  }
  
  # get lasso fit
  lmod= scalreg::scalreg(X,Y)
  # get estimated error noise level by refitting
  s = seq(p)[lmod$coefficients!=0]
  if (length(s)){
    theta_hat = sqrt(sum(resid(lm(Y~X[,s]))^2)/n)
  } else {
    theta_hat = sd(Y)
  }
  # debias
  dlfit = coefficients(lmod)+Mhat%*%t(X)%*%(Y-predict(lmod,X))/n
  # variable selection test stat
  T_i = sqrt(n)*dlfit/theta_hat/
    sqrt(diag(Mhat%*%cov(X)%*%t(Mhat)))
  return(list(coef=dlfit,T_i=T_i))
}



DAG_from_Ordering<-function(X,TO,mtd="ztest",alpha=0.05,
                            threshold=1e-1,FCD=NULL,precmtd=NULL){
  n=dim(X)[1]
  p=dim(X)[2]
  if (p!=length(TO)){stop("length mismatch")}
  if (mtd=="ztest"){
    # sidak
    C=cor(X)
    adj=matrix(0,p,p)
    for (i in 2:p){
      u=TO[i]
      for (j in 1:(i-1)){
        v = TO[j]
        s = setdiff(TO[seq(i-1)],v)
        pval = 1-(2*pnorm(abs(pcalg::zStat(u,v,s,C=C,n=n)))-1)^(p*(p-1)/2)
        adj[v,u]=ifelse(pval<alpha,1,0)
      }
    }
    return(adj!=0)
  }
  if (mtd=="chol"){
    Sigma=cov(X)
    B = solve(chol(Sigma[TO,TO])[order(TO),order(TO)])
    gm = diag(p)-B%*%diag(1/diag(B))
    return(gm*(abs(gm)>threshold)!=0)
  }
  if (mtd=="rls"){
    gm = upper.tri(matrix(0,p,p))[order(TO),order(TO)]
    colnames(gm)=rownames(gm)=colnames(X)
    return(abs(t(ggm::fitDag(gm,cov(X),dim(X)[1])$Ahat))-diag(p)>threshold)
  } else {
    # dblasso
    if (is.null(FCD)){FCD="T"}
    if (is.null(precmtd)){precmtd="sqrtlasso"}
    gm = matrix(0,p,p)
    gm[TO[1],TO[2]]=anova(lm(X[,TO[2]]~X[,TO[1]]))$`Pr(>F)`[1]<alpha
    if(p==2){return(gm)}
    for (i in 3:p){
      gm[TO[1:(i-1)],TO[i]]=
        vselect(X[,TO[1:i-1]],X[,TO[i]],alpha=alpha,p_total = p,
                selmtd = mtd,FCD = FCD,precmtd = precmtd)$selected
    }
    return(gm!=0)
  }
}


# Copyright (c) 2018 - 2020  Wenyu Chen [wenyuc@uw.edu]
# All rights reserved.  See the file COPYING for license terms.

###############
### Main method with high-dimensional bottom-up CLIME approach
###############
#' Estimate topological ordering and DAG using high dimensional bottom-up CLIME approach
#' Estimate  DAG using topological ordering
#' @param X,Y: n x p and 1 x p matrix
#' @param alpha: desired selection significance level
#' @param mtd: methods for learning DAG from topological orderings.
#'  "ztest": (p<n) [Multiple Testing and Error Control in Gaussian Graphical Model Selection. Drton and Perlman.2007]
#'  "rls": (p<n) fit recursive least squares using ggm package and threshold the regression coefs
#'  "chol": (p<n) perform cholesky decomposition and threshold the regression coefs
#'  "dlasso": debiased lasso (default with FCD=True and precmtd="sqrtlasso");
#'   "lasso": lasso with fixed lambda from [Penalized likelihood methods for estimation of sparse high-dimensional directed acyclic graphs. Shojaie and Michailidis. 2010];
#'   "adalasso": adaptive lasso with fixed lambda from [Shojaie and Michailidis. 2010];
#'   "cvlasso": cross-validated lasso from glmnet;
#'    "scallasso": scaled lasso.
#' @param threshold: for rls and chol, the threshold level.
#' @param FCD: for debiased lasso, use the FCD procedure [False Discovery Rate Control via Debiased Lasso. Javanmard and Montanari. 2018]
#' or use individual tests to select support.
#' @param precmtd: for debiased lasso, how to compute debiasing matrix
#'               "cv": node-wise lasso w/ joint 10 fold cv
#'               "sqrtlasso": square-root lasso (no tune, default)
#' @return Adjacency matrix with ADJ[i,j]!=0 iff i->j, and topological ordering
#' @examples
#' X1<-rnorm(100)
#' X2<-X1+rnorm(100)
#' EqVarDAG_HD_TD(cbind(X1,X2),2)
#'
#' #$adj
#' #[,1] [,2]
#' #[1,]    0    1
#' #[2,]    0    0
#' #
#' #$TO
#' #[1] 1 2
EqVarDAG_HD_CLIME<-function(X,mtd="dlasso",alpha=0.05,
                            threshold=1e-1,FCD=TRUE,precmtd="sqrtlasso"){
  # Input
  # X : n by p matrix of data
  # cv: if true, use cv-ed lambda, else use lambdafix,default True
  # lambdafix: customized lambda, default 0.1
  # Output
  # adj: estimated adjacency matrix
  # TO : estimated topological ordering
  n<-dim(X)[1]
  p<-dim(X)[2]
  lambda = 6.214*2
  TO=EqVarDAG_HD_CLIME_internal(X,lambda)
  adj=DAG_from_Ordering(X,TO,mtd,alpha,threshold,FCD,precmtd)
  return(list(adj=adj,TO=TO))
}


###############
### helper functions
###############
EqVarDAG_HD_CLIME_internal<-function(X,lam=NULL){
  # (i,j)=1 in fixedorder means i is ancestral to j
  n=dim(X)[1]
  p=dim(X)[2]
  Sigma=cov(X)
  if (is.null(lam)){lam = 4/sqrt(n)*sqrt(log(p/sqrt(0.05)))}
  Theta=clime_theta(X)
  TO=NULL
  while (length(TO)<p-1){
    sink=which.min(diag(Theta))
    TO=c(TO,sink)
    s = setdiff(which(Theta[,sink]!=0),sink)
    for (j in s){
      sj = unique(c(j,setdiff(which(Theta[,j]!=0),sink),s))
      if (length(sj)==1){
        Theta[j,sj]=Theta[sj,j]=1/Sigma[sj,sj]
      } else {
        Theta[j,sj]=Theta[sj,j]=clime_lp(Sigma[sj,sj],lam,1)
      }
    }
    Theta[,sink]=Theta[sink,]=rep(0,p)
    Theta[sink,sink]=Inf
  }
  return(rev(unname(c(TO,setdiff(seq(p),TO)))))
}

# clime utilities
unit_vec<-function(p,i){v=rep(0,p);v[i]=1;return(v)}
clime_lp<-function(Sigma,lam,j){
  # Sigma: cov(X)
  # lam: tuning parameter
  # j: the j-th problem
  p = dim(Sigma)[2]
  f.obj = rep(1,p*2) # sum (u+v), u=max(x,0), v=max(-x,0), x=u-v
  const.mat = rbind(
    cbind(Sigma,-Sigma), # Sigma*(u-v) >= lam +ej
    cbind(-Sigma,Sigma), # -Sigma*(u-v) >= lam-ej
    cbind(diag(p),matrix(0,p,p)), # u>0
    cbind(matrix(0,p,p),diag(p))  # v>0
  )
  const.dir = c(
    rep("<=",2*p),rep(">=",2*p)
  )
  const.rhs = c(
    rep(lam,p)+unit_vec(p,j),
    rep(lam,p)-unit_vec(p,j),
    rep(0,2*p)
  )
  lpout=lpSolve::lp(direction = "min",objective.in = f.obj,
                    const.mat = const.mat,const.dir = const.dir,
                    const.rhs = const.rhs)
  return(lpout$solution[1:p]-lpout$solution[(p+1):(2*p)])
}

clime_theta<-function(X,lam=NULL){
  p=dim(X)[2]
  n=dim(X)[1]
  Sigma = cov(X)
  if (is.null(lam)){lam = 4/sqrt(n)*sqrt(log(p/sqrt(0.05)))}
  Omega = sapply(1:p, function(i)clime_lp(Sigma,lam,i))
  Omega = (abs(Omega)<=abs(t(Omega)))*Omega+
    (abs(Omega)>abs(t(Omega)))*t(Omega)
  return(Omega)
}

cv_clime<-function(X){
  p=dim(X)[2]
  n=dim(X)[1]
  Sigma = cov(X)
  ws = sqrt(diag(Sigma))
  lams=exp(seq(log(1e-4),log(0.8),length.out = 100))
  ebics=sapply(1:100, function(j){
    Theta = sapply(1:p, function(i)clime_lp(Sigma,lams[j],i))
    Theta = (abs(Theta)<=abs(t(Theta)))*Theta+
      (abs(Theta)>abs(t(Theta)))*t(Theta)
    loglikGGM(S=Sigma,Theta=Theta)-sum(Theta!=0)/2*(log(p)+0.5*log(n))/n
  })
  return(clime_theta(X,lam = lams[which.max(ebics)]))
}


# library(QICD)
library(huge)
library(igraph)