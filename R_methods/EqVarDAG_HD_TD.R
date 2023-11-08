
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

EqVarDAG_HD_TD<-function(X,J=3,mtd="dlasso",alpha=0.05,
                         threshold=1e-1,FCD=TRUE,precmtd="sqrtlasso"){
  # Input
  # X : n by p matrix of data
  # Output
  # adj: estimated adjacency matrix
  # TO : estimated topological ordering
  n<-dim(X)[1]
  p<-dim(X)[2]
  TO=getOrdering(X,J)
  adj=DAG_from_Ordering(X,TO,mtd,alpha,threshold,FCD,precmtd)
  return(list(adj=adj,TO=TO))
}

###############
### helper functions
###############
# compute best subset search
helper.func <- function(z, Y, Theta, J,mtd="exhaustive"){
  leaps::regsubsets(x = Y[, Theta, drop = F],
                    y = Y[, z, drop = F],
                    method=mtd,
                    nbest = 1,
                    nvmax = min(J, sum(Theta > 0 )), really.big = T)
}
# compute topological ordering
getOrdering <- function(Y, J){
  p <- dim(Y)[2]
  variances <- apply(Y, MAR = 2, sd)
  Theta <- rep(0, p)
  Theta[1] <- which.min(variances)
  out <- sapply(setdiff(1:p, Theta[1]),
                function(z){
                  sum(resid(RcppEigen
                            ::fastLm(Y[, z] ~ Y[, Theta[1], drop = F]) )^2)})
  Theta[2] <- setdiff(1:p, Theta[1])[which.min(out)]
  for(i in 3:p){
    out <- lapply(setdiff(1:p, Theta),
                  function(jj)helper.func(jj, Y, Theta[seq(i-1)], J))
    nextRoot <- which.min(sapply(out,function(x){min(x$rss)}))
    Theta[i] <- setdiff(1:p, Theta)[nextRoot]
  }
  return(Theta)
}

library(QICD)
library(huge)
library(igraph)