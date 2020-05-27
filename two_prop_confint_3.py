
import scipy.stats as st
import numpy as np
import warnings

def two_prop_confint(p1,n1,p2,n2,confidence=0.95,method='normal'):
    """
    Confidence interval for the difference between two population proportions;
    given the sample proportions (p1,p2) and sample sizes (n1,n2) of two independent, simple random 
    samples estimate the difference between the two population proportions and the corresponding confidence interval;
    each sample should include at least 10 successes and 10 failures, otherwise provide warning; 

    Parameters
    ----------
    p1, p2 : float
        Proportions of successes in each sample. 

    n1, n2 : int
        Number of observations in each sample.

    confidence : float, optional
        Desired level of confidence (default: 0.95).
        
    method: {'normal','Continuity', 'AgrestiCaffo', 'CarlinLouis'}, optional
        default: 'normal', 
        method to use for confidence interval estimate,
        available methods : 
        - 'normal' : asymmptotic normal approximation (Wald two-sided confidence interval)
        - 'continuity' : asymptotic normal approximation with continuity correction
        - 'AgrestiCaffo' : Agresti-Caffo interval 
        - 'CarlinLouis' : Carlin-Louis interval 
        

    Returns
    -------
    diff : float
        Difference between the two proportions (p1-p2)

    conf_int : 1d ndarray
        Confidence interval for the difference of two proportions.
        
    
    Raises
    ------
    ValueError
        If proportions or confidence are not in range [0,1].
        If sample sizes are not greater than zero. 
        If method can not be recognized.
    TypeError
        If sample sizes are not integers.
        
        
    Warns
    ------
    Raise warning if normality assumption is violated (i.e. less than 10 successes and failures per sample).
    This warning is raised regardless of the method used.
    
        
    Notes
    -----
    Unpooled sample variances are used to calculate the confidence interval. 
    If method='Continuity', apply continuity correction insofar it does not exceed 
    the absolute difference between the two sample proportions. 
        
    
    References
    ----------
    Wilson, E. B. "Probable Inference, the Law of Succession,
    and Statistical Inference,"Journal of the American Statistical Association, 22, 209-212 (1927).
    Newcombe, Robert G. "Interval Estimation for the Difference Between Independent
    Proportions: Comparison of Eleven Methods," Statistics in Medicine, 17, 873-890 (1998).
    Agresti and Caffo (2000), "Simple and Effective Confidence Intervals for Proportions and
    Differences of Proportions Result From Adding Two Successes and Two Failures",
    The American Statistician, Vol. 54, No. 4, pp. 280-288. 
    Carlin and Louis (1996), "Bayes and Empirical Bayes Methods for Data Analysis", Chapman and Hall. 
    
    
    Example
    --------
    >>> import scipy.stats as st 
    >>> import numpy as np
    >>> import warnings
    >>> p1=0.5 # proportion of successes in sample_1
    >>> n1=210  # number of observations in sample_1
    >>> p2=0.35 # proportion of successes in sample_2
    >>> n2=130  # number of observations in sample_2
    >>> two_prop_confint(p1,n1,p2,n2,confidence=0.95,method="normal")
    (0.15, array([0.04371868, 0.25628132]))
    
    """
     
    if ((0<=p1<=1) & (0<=p2<=1))==False:
        raise ValueError("Proportions must be in range [0,1]")
        
    if ((0<n1) & (0<n2))==False:
        raise ValueError("Sample sizes must be greater than zero")
    
    if (0<=confidence<=1)==False:
        raise ValueError("Confidence must be in range [0,1], default: 0.95")

    if ((type(n1)==int) & (type(n2)==int))==False:
        raise TypeError ("Sample sizes must be integers.")
    
    if ((p1*n1<10) | ((1-p1)*n1<10) | (p2*n2<10) | ((1-p2)*n2<10))==True: 
        warnings.warn('Violation of normality assumption!\n''At least 10 successes and 10 failures are required per sample.\n'
                      'Results may be unreliable!')
        
    
    z = st.norm(loc = 0, scale = 1).ppf(confidence + (1-confidence)/2)
    diff=np.round(p1-p2,7)
    
    if method=="normal": 
        se1= np.sqrt(p1 * (1 - p1) / n1)
        se2= np.sqrt(p2 * (1 - p2) / n2)
        se = np.sqrt(se1**2 + se2**2) 
        lcb = diff - z*se
        ucb = diff + z*se
        conf_int=np.array([lcb,ucb])
        
    elif method=="Continuity":
        se1= np.sqrt(p1 * (1 - p1) / n1)
        se2= np.sqrt(p2 * (1 - p2) / n2)
        se = np.sqrt(se1**2 + se2**2) 
        correction_term= (1/n1 + 1/n2)/2
        correction_term_=np.min([np.abs(diff),correction_term])
        lcb = diff - z*se - correction_term_
        ucb = diff + z*se + correction_term_
        conf_int=np.array([lcb,ucb])
        
    elif method=="AgrestiCaffo":
        p1_= (p1*n1+1)/(n1+2)
        p2_= (p2*n2+1)/(n2+2)
        se1= np.sqrt(p1_ * (1 - p1_) / (n1+2))
        se2= np.sqrt(p2_ * (1 - p2_) / (n2+2))
        se = np.sqrt(se1**2 + se2**2) 
        diff_=np.round(p1_-p2_,7)
        lcb = diff_ - z*se
        ucb = diff_ + z*se
        conf_int=np.array([lcb,ucb])
        
    elif method=="CarlinLouis":
        p1_= (p1*n1+1)/(n1+2)
        p2_= (p2*n2+1)/(n2+2)
        se1= np.sqrt(p1_ * (1 - p1_) / (n1+3))
        se2= np.sqrt(p2_ * (1 - p2_) / (n2+3))
        se = np.sqrt(se1**2 + se2**2) 
        diff_=np.round(p1_-p2_,7)
        lcb = diff_ - z*se
        ucb = diff_ + z*se
        conf_int=np.array([lcb,ucb])
        
    else:
         raise ValueError('Method not recognized.\n'
                         'Should be "normal", "Continuity", "AgrestiCaffo" or "CarlinLouis".')
    
    return diff, conf_int



