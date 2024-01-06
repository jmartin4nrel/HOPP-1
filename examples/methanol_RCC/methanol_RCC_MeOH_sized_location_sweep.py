import numpy as np
import matplotlib.pyplot as plt

pt1a = 'methanol_RCC_pt1_MeOH_sized_mid.py'
pt1b = 'methanol_RCC_pt1_MeOH_sized_lo.py'
pt1c = 'methanol_RCC_pt1_MeOH_sized_hi.py'
multi = 'multi_location_RCC_NGCC_sized_location_sweep.py'
pt2 = 'methanol_RCC_pt2_NGCC_sized.py'


from methanol_RCC_aspen_fcn import try_H2_ratio
from methanol_RCC_pt1_MeOH_sized_lo_fcn import try_H2_ratio as try_H2_ratio_lo
from methanol_RCC_pt1_MeOH_sized_hi_fcn import try_H2_ratio as try_H2_ratio_hi
from methanol_RCC_pt2_NGCC_sized import try_H2_price

plt.clf
H2_ratios = [3.358236210,0.335099771,3.053578151,0.331063017,1.266865819,0.248240577,1.008276917,0.253104021]
#[0.3265325,1.0023537,0.3265325,0.3265325,0.3265325,0.3161153,0.3057350,0.2955746,0.2800000,0.2400000,0.2000000]
H2_mults = [1,1,1,1,1,1,1,1,1,1,1]
CO2_feed_mt_yrs = [5067429,1048657,4658568,1039624,2139935,871546,1882485,880250]
#[1049531,4109672,1049531,1049531,1049531,1035886,1021990,1008477,1049531,1049531,1049531]
ASPEN_MeOH_cap_mt_yrs = [115104,115104,115104,115104,115104,115104,115104,115104,115104,115104,115104]
ASPEN_capexs = [69898147.52,34659745.44,62807644.75,33598733.38,34872041.88,24167459.75,31442263.88,24789650.06]
#[33303653,60397722,31680188,31680188,30677975,32392895,31415107,30389563,33303653,33303653,33303653]
ASPEN_Fopexs = [26.31,13.47,24.64,13.29,16.03,11.51,15.00,11.63]
#[13.28,23.36,13.05,13.05,12.91,13.11,12.94,12.76,13.28,13.28,13.28] 
ASPEN_Vopex_cats = [494.18,102.27,340.73,76.04,102.02,41.55,80.24,37.52]
#[102.35,400.78,71.86,55.37,45.03,101.02,99.66,98.35,102.35,102.35,102.35]
ASPEN_Vopex_others = [-1444.79,-71.42,-1308.02,-69.64,-490.18,-29.48,-374.82,-32.21]
#[-66.5374818,-355.1076575,-66.5374818,-66.5374818,-66.5374818,-61.56203064,-56.58869893,-51.69084291,-66.5374818,-66.5374818,-66.5374818]
ASPEN_elec_uses = [1.159362271,0.794807345,1.064284552,0.786466327,0.482940411,0.460142294,0.421340209,0.489793618]
#[0.721669808,0.892844636,0.721669808,0.721669808,0.721669808,0.673099203,0.624139417,0.575637102,0.721669808,0.721669808,0.721669808]
ASPEN_H2O_uses = [31.41290814,1.689291115,28.36035461,1.641945643,10.52488995,0.678666004,8.036898151,0.749000491]
#[1.54767215,7.63090881,1.53578830,1.52935933,1.52533052,1.42478882,1.30166200,1.17999767,1.54767215,1.54767215,1.54767215]
ASPEN_MeOH_CO2s = [0.378428177,0.078312183,0.260921395,0.058228283,0.078128125,0.03181976,0.061446216,0.028732272]
#[0.07837749,0.30690417,0.05503105,0.04240101,0.03448619,0.07735845,0.07632072,0.07531160,0.07837749,0.07837749,0.07837749]

H2_prices = [1,1,1,1,1,1,1,1,1,1,1]#1.80615878
for DAC_cost_mt in [0]:
    for H2_idx, H2_ratio in enumerate(H2_ratios[:1]):
        # for file in [pt1a,pt1b,pt1c,multi]:#,pt2]:
        try_H2_ratio(H2_ratio*H2_mults[H2_idx],
                                CO2_feed_mt_yrs[H2_idx],
                                ASPEN_MeOH_cap_mt_yrs[H2_idx],
                                ASPEN_capexs[H2_idx],
                                ASPEN_Fopexs[H2_idx],
                                ASPEN_Vopex_cats[H2_idx],
                                ASPEN_Vopex_others[H2_idx],
                                ASPEN_elec_uses[H2_idx],
                                ASPEN_H2O_uses[H2_idx],
                                ASPEN_MeOH_CO2s[H2_idx],
                                )
        # try_H2_ratio_lo(H2_ratio)
        # try_H2_ratio_hi(H2_ratio)
        
        # with open(multi) as f:
        #     exec(f.read())

        old_price_diff = -1
        breakeven_price = 2
        price_diff = try_H2_price(H2_prices[H2_idx], H2_idx, plotting=False, DAC_cost_mt=DAC_cost_mt, run_idx=H2_idx)
        # for H2_price in np.arange(1.8,1.5,-0.01):
        #     price_diff = try_H2_price(H2_price, plotting=False)
        #     if (price_diff > 0) and (old_price_diff < 0):
        #         breakeven_price = H2_price
        #         break
        #     old_price_diff = price_diff
        # H2_prices.append(breakeven_price)

    # plt.plot(H2_ratios,H2_prices)
    plt.show()