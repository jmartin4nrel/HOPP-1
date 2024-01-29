import numpy as np
import matplotlib.pyplot as plt

recycle_coeffs = np.array([0.121311587,-0.363971753,-0.246876268,0.130576677,0.521401591,0.125834842])
recycle_coeffs = np.array([0.052877898,0,-0.680921662,0.15845236,0.437972757,0.125834842])
recycle_coeffs = np.array([0.156322702,-0.556284973,0,0.11727409,0.559221956,0.125834842])
recycle_coeffs = np.array([-0.707629889,0.541599049,-2.687339055,0.965669103,0.806770062,-0.118067538])
recycle_coeffs = np.array([0.004344351,-0.031728817,0.11289189,0.226684858,0.166269883,0.123772965])
recycle_coeffs = np.array([-0.912158271,0.468715564,-4.567673613,0.447004034,0.932597335,1.375229872,0.003151389,-0.0985312,0.137097221,0.125834842])
recycle_coeffs = np.array([0,-0.862625104,0,0.34919482,0.75844123,0.044813135])
recycle_coeffs = np.array([5.888074305,4.706015913,0.490968032,-1.292476742,0.358803576,0.119962526])

recycle_coeffs = np.array([5.727062617,2.149454309,8.180333752,-1.009180563,-0.397734837,0.117147289])

recycle_coeffs = np.array([5.344973406,3.319247523,7.791814942,-0.810991271,-0.932391716,0.117047199])






recycle_coeffs = np.array([0,-0.140722005,0,0.202260591,0.330659156,0.12583702])







meth_sel = np.arange(20,101,2)
co2_conv = np.arange(60,101)

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

max_ratio = 5.0
min_ratio = 0

max_ratio = 0.37
min_ratio = 0.12

max_ratio = 13.0
min_ratio = 0

max_ratio = 1.2
min_ratio = 0.3

# max_ratio = 7.0
# min_ratio = 0

# max_ratio = 1.4
# min_ratio = 0.0


# Convert to LCOM
ratios = [0.1353*i**5-1.1868*i**4+3.7065*i**3-4.9465*i**2+4.9105*i-0.2194 for i in ratios]

# # Convert to CI
# ratios = [0.0897*i**5+0.7853*i**4+2.4453*i**3-3.2317*i**2+3.229*i-0.2525 for i in ratios]

ratios = [max(i,min_ratio) for i in ratios]
ratios = [min(i,max_ratio) for i in ratios]



ratio_mat = np.reshape(ratios,np.shape(A))

# map = plt.contourf(X,Y,ratio_mat, np.arange(0.12,0.39,0.02))#np.arange(0.0,5.5,0.5))# 
map = plt.contourf(X,Y,ratio_mat, np.arange(0.3,1.2,0.1))#np.arange(0.0,14,1))# 
# map = plt.contourf(X,Y,ratio_mat, np.arange(0.1,0.7,0.05))#np.arange(0.0,7.5,1.0))# 

# map.set_cmap('plasma_r')

# plt.colorbar(label='H2:MeOH Ratio')
plt.colorbar(label='LCOM [$/kg-MeOH]')
# plt.colorbar(label='CI [kg-CO2e/kg-MeOH]')

plt.contour(X,Y,ratio_mat,[0.79,100],colors=[[1,0,0]],linestyles=['--'],label='Baseline')

cat_sels = [23.0,27.8,31.0,46.4,58.1]
cat_effs = [86.4,73.1,71.2,94.4,84.5]

cat_sels = [23.0,27.8,31.0,46.4,58.1]
cat_effs = [86.4,73.1,71.2,94.4,84.5]


# # plt.plot(cat_sels[0],cat_effs[0],'.',label='CZA Jun 23')
# # plt.plot(cat_sels[1],cat_effs[1],'.',label='CZA Nov 23')
plt.plot(cat_sels[1],cat_effs[1],'o',label='CZA',color=[1,0,0],markeredgecolor='w')
plt.plot(cat_sels[2],cat_effs[2],'v',label='Ca/CZA',color=[.5,0,1],markeredgecolor='w')
plt.plot(cat_sels[3],cat_effs[3],'s',label='K/CZA',color=[1,0,1],markeredgecolor='w')
plt.plot(cat_sels[4],cat_effs[4],'D',label='Na/CZA',color=[1,.5,0],markeredgecolor='w')
# plt.ylim([60,100])
plt.xlabel('Methanol selectivity %')
plt.ylabel('Net CO2 conversion %')
plt.title('H2:MeOH Ratio, Recycle',pad=30)
plt.title('Levelized Cost of Methanol, Recycle',pad=30)
# plt.title('Carbon Intensity, Recycle\n')
plt.legend(bbox_to_anchor=(1.45,1.25),ncol=4,handletextpad=0.0,columnspacing=0.1)
# plt.grid('on')
plt.gcf().set_size_inches(3.4,3)
plt.gcf().set_tight_layout(True)
plt.show()