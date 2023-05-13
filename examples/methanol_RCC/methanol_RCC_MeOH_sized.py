import numpy as np
import matplotlib.pyplot as plt

pt1a = 'methanol_RCC_pt1_MeOH_sized_mid.py'
pt1b = 'methanol_RCC_pt1_MeOH_sized_lo.py'
pt1c = 'methanol_RCC_pt1_MeOH_sized_hi.py'
multi = 'multi_location_RCC_NGCC_sized.py'
pt2 = 'methanol_RCC_pt2_NGCC_sized.py'


from methanol_RCC_pt1_MeOH_sized_mid_fcn import try_H2_ratio
from methanol_RCC_pt1_MeOH_sized_lo_fcn import try_H2_ratio as try_H2_ratio_lo
from methanol_RCC_pt1_MeOH_sized_hi_fcn import try_H2_ratio as try_H2_ratio_hi
from methanol_RCC_pt2_NGCC_sized import try_H2_price

plt.clf
H2_ratios = [0.29,0.25,0.21]
H2_prices = [1,1,1,1.08,0.92,0.8,0.71,0.64,0.58]#1.80615878
for H2_idx, H2_ratio in enumerate(H2_ratios):
    # for file in [pt1a,pt1b,pt1c,multi]:#,pt2]:
    try_H2_ratio(H2_ratio)
    # try_H2_ratio_lo(H2_ratio)
    # try_H2_ratio_hi(H2_ratio)
    with open(multi) as f:
        exec(f.read())

    old_price_diff = -1
    breakeven_price = 2
    price_diff = try_H2_price(H2_prices[H2_idx], H2_idx, plotting=True)
    # for H2_price in np.arange(1.8,1.5,-0.01):
    #     price_diff = try_H2_price(H2_price, plotting=False)
    #     if (price_diff > 0) and (old_price_diff < 0):
    #         breakeven_price = H2_price
    #         break
    #     old_price_diff = price_diff
    # H2_prices.append(breakeven_price)

# plt.plot(H2_ratios,H2_prices)
plt.show()