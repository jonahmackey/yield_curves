import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import math


# Q4. (a)
def price_equation1(r, price, coupon, maturity):
    time = 0.5
    result = 0
    
    # all cash flows except final
    while time < maturity:
        result += coupon / ((1 + (r / 2)) ** (2 * time))
        time += 0.5
    
    # final cash flow (coupon + face value)
    result += (coupon + 100) / ((1 + (r / 2)) ** (2 * maturity))
    
    return result - price


def compute_ytm(price, coupon, maturity):
    sol = root_scalar(price_equation1, args=(price, coupon, maturity), bracket=[0.0001, 2], method='brentq')
    return sol.root
    

def plot_yield_curve(time_to_maturity, coupons, price_data):
    X = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    yields = []
    
    plt.figure()
    for date in price_data.columns:
        prices = price_data[date].to_numpy()
        
        # include day-count
        time_to_maturity2 = time_to_maturity - (int(date.split('/')[1]) - 1)/ 365
        
        # calculate ytm
        ytm = []
        
        for i in range(len(prices)):
            ytm.append(compute_ytm(price=prices[i], coupon=coupons[i], maturity=time_to_maturity2[i]))
        
        yields.append(ytm)
        plt.plot(X, ytm, '-', label=date)
        
    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    plt.xlabel('Year')
    plt.ylabel('Yield to Maturity')
    plt.title('0-5 Year Yield Curves')
    plt.legend(loc='upper right')
    plt.savefig('./Images/yield_curves.png')
    
    return np.array(yields)


# Q4. (b)
def price_equation2(r, price, coupon, spot_rates):
    result = 0
    
    # all cash flows except final
    for i in range(len(spot_rates)):
        result += coupon * math.exp(-spot_rates[i] * (i+1) * 0.5)
        
    # final cash flow (coupon + face value)
    result += (coupon + 100) * math.exp(-r * ((len(spot_rates) + 1) * 0.5))
    
    return result - price


def compute_spot_rate(price, coupon, spot_rates):
    sol = root_scalar(price_equation2, args=(price, coupon, spot_rates), bracket=[0.0001, 2], method='brentq')
    return sol.root


def plot_spot_rate_curve(coupons, price_data):
    X = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    spot_rates = []
    
    plt.figure()
    
    for date in price_data.columns:
        prices = price_data[date].to_numpy()
        
        # calculate spot_rate
        spot_rate = []
        spot_rate.append(-math.log(prices[0] / (100 + coupons[0])) / 0.5)
        
        for i in range(len(prices[1:])):
            spot_rate.append(compute_spot_rate(price=prices[i+1], coupon=coupons[i+1], spot_rates=spot_rate))
        
        spot_rates.append(spot_rate[1:])
        plt.plot(X, spot_rate[1:], '-', label=date)
        
    plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    plt.xlabel('Year')
    plt.ylabel('Spot Rate')
    plt.title('Spot Rate Curves')
    plt.legend(loc='upper right')
    plt.savefig('./Images/spot_rate_curves.png')

    return np.array(spot_rates)


# Q4. (c)
def plot_forward_rate_curve(spot_rates, dates):
    X = np.array(['1-1', '1-2', '1-3', '1-4'])
    forward_rates = []
    
    plt.figure()
    
    for i in range(len(spot_rates)): 
        forward_rate = []
        
        for j in range(len(spot_rates[i]) - 1): 
            forward_rate.append((((1 + spot_rates[i][j+1]) ** (j+2)) / (1 + spot_rates[i][0])) - 1)
            
        forward_rates.append(forward_rate)
        plt.plot(X, forward_rate, '-', label=dates[i])
    
    plt.xticks(['1-1', '1-2', '1-3', '1-4'])
    plt.xlabel('Year')
    plt.ylabel('Forward Rate')
    plt.title('Forward Rate Curves')
    plt.legend()
    plt.savefig('./Images/forward_rate_curves.png')
    
    return np.array(forward_rates)
            

# Q5, Q6
def log_return_cov(X):
    log_returns = []
    
    for x in X:
        log_return = []
        for i in range(len(x) - 1):
            log_return.append(math.log(x[i+1] / x[i]))
        log_returns.append(log_return)
        
    log_returns = np.array(log_returns) # (maturity, daily log return) = (5, 10) 
    log_returns = log_returns - np.mean(log_returns, axis=0) # centering
    
    covariance = (1 / 5) * (np.transpose(log_returns) @ log_returns) # (10, 10)
    
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[order]
    
    return covariance, eigenvalues, eigenvectors


if __name__=="__main__":
    data = pd.read_csv('./data.csv')
    
    maturity_dates = list(data['maturity date'])
    time_to_maturity = []
    
    for date in maturity_dates:
        date = date.split('/')
        time_to_maturity.append(int(date[2]) - 2023 + (int(date[0]) - 1) / 12)
    
    time_to_maturity = np.array(time_to_maturity)
    coupons = data['coupon'].to_numpy()
    price_data = data[data.columns[4:]]
    
    # Q4 (a)
    yields = plot_yield_curve(time_to_maturity, coupons, price_data) # (11, 10)
    
    # Q4 (b)
    spot_rates = plot_spot_rate_curve(coupons, price_data) # (11, 9)
    
    # Q4 (c)
    spot_rates = spot_rates[:, np.array([0, 2, 4, 6, 8])] # (11, 5)
    forward_rates = plot_forward_rate_curve(spot_rates, price_data.columns) # (11, 4)
    
    # Q5, Q6
    yields = np.transpose(yields)[np.array([1, 3, 5, 7, 9])] # (5, 11)
    covariance, eigenvalues, eigenvectors = log_return_cov(yields) # (10, 10), (10,), (10, 10)
    
    print('-' * 30 + 'LOG-RETURN OF YIELD' + '-' * 30)
    print(f'Covariance Matrix: \n{covariance}\n')
    print(f'Eigenvalues: \n{eigenvalues}\n')
    print(f'Eigenvectors: \n{eigenvectors}\n')
    
    forward_rates = np.transpose(forward_rates) # (4, 11)
    covariance, eigenvalues, eigenvectors = log_return_cov(forward_rates) # (10, 10), (10,), (10, 10)
    
    print('-' * 30 + 'LOG-RETURN OF FORWARD RATE' + '-' * 30)
    print(f'Covariance Matrix: \n{covariance}\n')
    print(f'Eigenvalues: \n{eigenvalues}\n')
    print(f'Eigenvectors: \n{eigenvectors}\n')
