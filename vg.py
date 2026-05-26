
from numpy import *
from scipy import special
from scipy.integrate import quad
from scipy import optimize
from matplotlib import pyplot as plt


# TODO: Assuming risk free rate is 0 
def calculateCallOptPrice(vgParams, strike, spot, t):
    return quad(lambda R: max(spot * exp(R) - strike, 0) * pdf_one_point(R, *vgParams, t),-0.99,2)[0]

def calculatePutOptPrice(vgParams, strike, spot, t):
    return quad(lambda R: max(strike - spot * exp(R), 0) * pdf_one_point(R, *vgParams, t),-0.99,1)[0]


# Slightly modified https://github.com/dlaptev/VarGamma/blob/master/VarGamma.py
def pdf_one_point(x=0.0, c=0.0, sigma=1.0, theta=0.0, nu=1.0, t=1):
	x2 = (x - c*t - (t/nu*log(1-theta*nu-(sigma**2 *nu)/2)))
	temp1 = 2.0 / ( sigma*(2.0*pi)**0.5*nu**(t/nu)*special.gamma(t/nu) )
	temp2 = ((2*sigma**2/nu+theta**2)**0.5)**(0.5-t/nu)
	temp3 = exp(theta*(x2)/sigma**2) *  (x2**2)**(t/(2*nu)-0.25)
	temp4 = special.kv(t/nu - 0.5,  abs(x2)*(2*sigma**2/nu+theta**2)**0.5/sigma**2)
	return temp1*temp2*temp3*temp4

def pdf(x=0.0, c=0.0, sigma=1.0, theta=0.0, nu=1.0):
	if isinstance(x, (int, float, double)): 
		return pdf_one_point(x, c, sigma, theta, nu)
	else:
		return [pdf_one_point(xi, c, sigma, theta, nu) for xi in x]

def fit_moments(x):
	''' fits the parameters of VarGamma distribution to a given list of points
		via method of moments, assumes that theta is small (so theta^2 = 0)
		see: Seneta, E. (2004). Fitting the variance-gamma model to financial data.
	 '''
	mu = mean(x)
	sigma_squared = mean( (x-mu)**2 )
	beta = mean( (x-mu)**3 ) / mean( (x-mu)**2 )**1.5
	kapa = mean( (x-mu)**4 ) / mean( (x-mu)**2 )**2
	# solve combined equations
	sigma = sigma_squared**0.5
	nu = kapa/3.0 - 1.0
	theta = sigma*beta / (3.0*nu)
	c = mu - theta
	return (c, sigma, theta, nu)

def neg_log_likelihood(data, par):
	''' negative log likelihood function for VarGamma distribution '''
	# par = array([c, sigma, theta, nu])
	if (par[1] > 0) & (par[3] > 0):
		return -sum(log( pdf(data, c=par[0], sigma=par[1], theta=par[2], nu = par[3]) ))
	else:
		return Inf

def fit(data):
	''' fits the parameters of VarGamma distribution to a given list of points
		via Maximizing the Likelihood functional (=minimizing negative log likelihood)
		the initial point is chosen with fit_moments(x),
		optimization is performed using Nedler-Mead method '''
	par_init = array( fit_moments(data) )
	par = optimize.fmin(lambda x: neg_log_likelihood(data, x), par_init, maxiter=1000)
	return tuple(par)

def plotVG(bins: list[float], params: tuple[float], t):
    probs = []

    for i in range(1, len(bins)): 
        x1 = bins[i - 1]
        x2 = bins[i]
        prob = quad(lambda x: pdf_one_point(x, *params, t), x1, x2)[0]
        plt.plot([x1, x2], [prob, prob], color="red")  
        probs.append(prob)     
		
    return probs