import numpy as np
import matplotlib.pyplot as plt

pt1a = 'methanol_RCC_pt1_MeOH_sized_mid.py'
pt1b = 'methanol_RCC_pt1_MeOH_sized_lo.py'
pt1c = 'methanol_RCC_pt1_MeOH_sized_hi.py'
multi = 'multi_location_RCC_NGCC_sized.py'
pt2 = 'methanol_RCC_pt2_NGCC_sized.py'


from methanol_RCC_aspen_fcn import try_H2_ratio
from methanol_RCC_pt1_MeOH_sized_lo_fcn import try_H2_ratio as try_H2_ratio_lo
from methanol_RCC_pt1_MeOH_sized_hi_fcn import try_H2_ratio as try_H2_ratio_hi
from methanol_RCC_pt2_NGCC_sized import try_H2_price

plt.clf
H2_ratios = [0.238756, 0.238756, 0.238756, 0.238756, 0.44, 0.44, 0.44, 0.44, 0.4, 0.36, 0.32]
H2_mults = [1/1.037454, 1/1.037454, 1/1.037454, 1/1.037454, 1.002007, 1.002007, 1.002007, 1.002007, 0.995345, 0.989257, 0.982099]
CO2_feed_mt_yrs = [869253, 869253, 869253, 869253, 1596153, 1596153, 1596153, 1596153, 1596153, 1596153, 1596153]
ASPEN_MeOH_cap_mt_yrs = [115104, 115104, 115104, 115104, 115104, 115104, 115104, 115104, 115104, 115104, 115104]
ASPEN_capexs = [27283412, 25090441, 24107589, 23525040, 33339802, 30732435, 29419107, 28608350, 33339802, 33339802, 33339802]
ASPEN_Fopexs = [11.95, 11.64, 11.50, 11.42, 14.62, 14.25, 14.07, 13.96, 14.62, 14.62, 14.62] 
ASPEN_Vopex_cats = [223.79, 111.90, 74.60, 55.95, 410.94, 205.47, 136.98, 102.73, 410.94, 410.94, 410.94]
ASPEN_Vopex_others = [-0.55, -0.55, -0.55, -0.55, -90.4, -90.4, -90.4, -90.4, -90.4, -90.4, -90.4]
H2_prices = [1,1,1,1,1,1,1,1,1,1,1,1,1]#1.80615878
for DAC_cost_mt in [0,400,1000]:
    for H2_idx, H2_ratio in enumerate(H2_ratios):
        # for file in [pt1a,pt1b,pt1c,multi]:#,pt2]:
        try_H2_ratio(H2_ratio*1.037454*H2_mults[H2_idx],
                                CO2_feed_mt_yrs[H2_idx],
                                ASPEN_MeOH_cap_mt_yrs[H2_idx],
                                ASPEN_capexs[H2_idx],
                                ASPEN_Fopexs[H2_idx],
                                ASPEN_Vopex_cats[H2_idx],
                                ASPEN_Vopex_others[H2_idx])
        # try_H2_ratio_lo(H2_ratio)
        # try_H2_ratio_hi(H2_ratio)
        
        with open(multi) as f:
            exec(f.read())

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