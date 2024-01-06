import numpy as np
import matplotlib.pyplot as plt

recycle_coeffs = np.array([0.121311587,-0.363971753,-0.246876268,0.130576677,0.521401591,0.125834842])
recycle_coeffs = np.array([0.052877898,0,-0.680921662,0.15845236,0.437972757,0.125834842])
recycle_coeffs = np.array([0.156322702,-0.556284973,0,0.11727409,0.559221956,0.125834842])
#recycle_coeffs = np.array([-0.707629889,0.541599049,-2.687339055,0.965669103,0.806770062,-0.118067538])
recycle_coeffs = np.array([0.004344351,-0.031728817,0.11289189,0.226684858,0.166269883,0.123772965])
# recycle_coeffs = np.array([-0.912158271,0.468715564,-4.567673613,0.447004034,0.932597335,1.375229872,0.003151389,-0.0985312,0.137097221,0.125834842])
recycle_coeffs = np.array([0,-0.862625104,0,0.34919482,0.75844123,0.044813135])


meth_sel = np.arange(0,101)
co2_conv = np.arange(0,101)

a = [1-i/100 for i in meth_sel]
b = [1-i/100 for i in co2_conv]

[X,Y] = np.meshgrid(meth_sel,co2_conv)
[A,B] = np.meshgrid(a,b)

Avec = np.reshape(A,[-1,1])
Bvec = np.reshape(B,[-1,1])

# ABmat = np.hstack([np.multiply(Avec,np.square(Avec)),
#                    np.multiply(Bvec,np.square(Avec)),
#                    np.multiply(Avec,np.square(Bvec)),
#                    np.multiply(Bvec,np.square(Bvec)),
ABmat = np.hstack([np.square(Avec),
                   np.multiply(Avec,Bvec),
                   np.square(Bvec),
                   Avec,
                   Bvec,
                   np.ones(np.shape(Avec))])

ratios = np.matmul(ABmat,recycle_coeffs)

max_ratio = 0.34
min_ratio = 0.14
ratios = [max(i,min_ratio) for i in ratios]
ratios = [min(i,max_ratio) for i in ratios]

ratio_mat = np.reshape(ratios,np.shape(A))

plt.contourf(X,Y,ratio_mat)
plt.colorbar(label='H2:MeOH Ratio')

cat_sels = [23.0,27.8,31.0,46.4,58.1]
cat_effs = [86.4,73.1,71.2,94.4,84.5]
plt.plot(cat_sels[0],cat_effs[0],'.',label='CZA Jun 23')
plt.plot(cat_sels[1],cat_effs[1],'.',label='CZA Nov 23')
plt.plot(cat_sels[2],cat_effs[2],'.',label='Ca/CZA')
plt.plot(cat_sels[3],cat_effs[3],'.',label='K/CZA')
plt.plot(cat_sels[4],cat_effs[4],'.',label='Na/CZA')
plt.ylim([60,100])
plt.legend()
plt.xlabel('Methanol selectivity %')
plt.ylabel('Net CO2 conversion %')
plt.show()