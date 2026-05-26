import scipy 
import scipy.stats as st
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats._continuous_distns import _distn_names
import warnings

# Test various probability distributions and see which best fits 

def plotHist(data: list[float], color, bins=250, normalize=True, alpha=1): 
    weights = np.ones_like(data) / len(data)
    values, bins, patches = plt.hist(data,color=color,weights=weights if normalize else None, bins=bins, density=False,alpha=alpha)
    return values, bins, patches


def plotDistrib(bins: list[float], distrib, params: tuple[float] = None): 
    probs = []

    for i in range(1, len(bins)): 
        x1 = bins[i - 1]
        x2 = bins[i]

        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        prob = distrib.cdf(x2, *arg, loc=loc, scale=scale) - distrib.cdf(x1, *arg, loc=loc, scale=scale)
        plt.plot([x1, x2], [prob, prob], color="black")  
        probs.append(prob)     

    return probs 


def bestFitPdf(data: list[float], nbins=250): 
    y,x = np.histogram(data, bins=nbins, density=True)
    # values, bins, patches = plotHist(data, nbins)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    best_distributions = []   

    # https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python/37616966#37616966

    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                # if axis pass in add to plot
                # try:
                #     if ax:
                #         pd.Series(pdf, x).plot(ax=ax)
                #     end
                # except Exception:
                #     pass

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))
        
        except Exception as e:
            print("Skipped distribution: ", distribution)
            print("Error: ", e)
            pass

    best_distributions.sort(key=lambda x:x[2])

    # plot the best distribution
    best_distribution = best_distributions[0]
    distribution = best_distribution[0]
    params = best_distribution[1]
    pdf = distribution.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

    print("Best fit distribution: ", distribution.name)

    # Print top 5
    print("Top 5 distributions: ")
    for i in range(5): 
        print(best_distributions[i][0].name, best_distributions[i][2])

    # print parameters
    print("Parameters: ", params)

    plt.show()