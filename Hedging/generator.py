import numpy as np

np.random.seed(111)

def generate_path(asset_model, p, m, n, time_step, R_f):
    #  return n * 1 / time_step * p
    #  p: number of assets, 5, 50, 100
    #  m >=p : dim of random noise
    #  n: total number of experiments(path)
    #  time_step: \delta t
    # print('Generate asset_returns under the {} model with the time step {} and the interest rate {}'.format(asset_model, time_step, r_f))

    if asset_model == 'BS':
        r_f = 0.03 # risk free ret
        if m == p: # complete market
            if p == 5:
                lamb = 0.1 * np.ones(p)
                lamb[2:] = 0.2 # lambda p*1
                sig = 0.01 * np.ones([p, p])
                np.fill_diagonal(sig, 0.15) # Sigma p*p
            elif p == 50:
                lamb = 0.01 * np.ones(p)
                lamb[25:] = 0.05
                sig = 0.001 * np.ones([p, p])
                np.fill_diagonal(sig, 0.15)
#            elif p == 100:
#                lamb = 0.01 * np.ones(p)
#                lamb[50:] = 0.05
#                sig = 0.0025 * np.ones([p, p])
#                np.fill_diagonal(sig, 0.15)

            K = int(1 / time_step)

            randomness = np.random.randn(n, K, p) # epsilon n * K * p

            tmp1 = (r_f + np.matmul(sig, lamb) - 0.5 * np.diag(np.matmul(sig, np.transpose(sig)))) * time_step
            tmp2 = np.sqrt(time_step) * np.matmul(randomness, sig)
            rs = np.exp(tmp1 + tmp2) - R_f # BSM MODEL

        elif m > p: # incomplete market
            if p == 5:
                lamb = 0.1 * np.ones(m)
                lamb[2:] = 0.2 # lambda p*1
                sig = 0.01 * np.ones([p, m])
                np.fill_diagonal(sig, 0.15) # Sigma p * m
            elif p == 50:
                lamb = 0.01 * np.ones(m)
                lamb[25:] = 0.05
                sig = 0.001 * np.ones([p, m])
                np.fill_diagonal(sig, 0.15)
            #elif p == 100:
            #    lamb = 0.01 * np.ones(m)
            #    lamb[50:] = 0.05
            #    sig = 0.0025 * np.ones([p, m])
            #    np.fill_diagonal(sig, 0.15)

            K = int(1 / time_step)

            randomness = np.random.randn(n, K, m) # epsilon n * K * m

            tmp1 = (r_f + np.matmul(sig, lamb) - 0.5 * np.diag(np.matmul(sig, np.transpose(sig)))) * time_step
            tmp2 = np.sqrt(time_step) * np.matmul(randomness, np.transpose(sig))
            rs = np.exp(tmp1 + tmp2) - R_f # BSM MODEL
        else:
            print('m needs to be greater than or equal to p !!!')
    else:
        print("Please set asset_model=='BS'!!!")


    return rs







