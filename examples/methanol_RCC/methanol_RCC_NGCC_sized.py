pt1a = 'methanol_RCC_pt1_NGCC_sized_mid.py'
pt1b = 'methanol_RCC_pt1_NGCC_sized_lo.py'
pt1c = 'methanol_RCC_pt1_NGCC_sized_hi.py'
multi = 'multi_location_RCC_NGCC_sized.py'
pt2 = 'methanol_RCC_pt2_NGCC_sized.py'

for file in [pt1a,pt1b,pt1c,multi,pt2]:
    with open(file) as f:
        exec(f.read())