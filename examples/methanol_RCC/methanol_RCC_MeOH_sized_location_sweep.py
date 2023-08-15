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
H2_ratios = [0.3265325,1.0023537,0.3265325,0.3265325,0.3265325,0.3161153,0.3057350,0.2955746,0.2800000,0.2400000,0.2000000]
H2_mults = [1]
CO2_feed_mt_yrs = [1049531,4109672,1049531,1049531,1049531,1035886,1021990,1008477,1049531,1049531,1049531]
ASPEN_MeOH_cap_mt_yrs = [115104,115104,115104,115104,115104,115104,115104,115104,115104,115104,115104]
ASPEN_capexs = [33303653,60397722,31680188,31680188,30677975,32392895,31415107,30389563,33303653,33303653,33303653]
ASPEN_Fopexs = [13.28,23.36,13.05,13.05,12.91,13.11,12.94,12.76,13.28,13.28,13.28] 
ASPEN_Vopex_cats = [102.35,400.78,71.86,55.37,45.03,101.02,99.66,98.35,102.35,102.35,102.35]
ASPEN_Vopex_others = [-66.54,-355.11,-66.54,-66.54,-66.54,-61.56,-56.59,-51.69,-66.54,-66.54,-66.54]
H2_prices = [1]#1.80615878
for DAC_cost_mt in [0]:
    for H2_idx, H2_ratio in enumerate(H2_ratios[:1]):
        # for file in [pt1a,pt1b,pt1c,multi]:#,pt2]:
        try_H2_ratio(H2_ratio*H2_mults[H2_idx],
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