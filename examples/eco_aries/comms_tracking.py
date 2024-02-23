import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def setup_tracking(plotting):
    
    # Make time series - "hopp_time" with one point per hour, "hopp_time2" with two points per hour
    hopp_time = pd.date_range('2019-01-05 14:00', periods=25, freq='1 h')
    aries_time = pd.date_range('2019-01-05 14:00', periods=0, freq='100 ms')
    aries_xdata = {'wind':[],'wave':[],'solar':[],'batt':[],'elyzer':[],'soc':[]}
    if plotting:
        hopp_time2 = np.vstack([hopp_time,hopp_time])
        hopp_time2 = np.reshape(np.transpose(hopp_time2),25*2)
        hopp_time2 = hopp_time2[1:-1]
        # hopp_time = hopp_time[:-1]

        lines = np.empty([3,2],object)
        for i in range(3):
            for j in range(2):
                lines[i,j] = []

        fig,ax=plt.subplots(3,2)
        fig.set_figwidth(15.0)
        fig.set_figheight(7.0)

        lines[0,0].append(ax[0,0].plot(aries_time,aries_xdata['wave'],label=None,color=[0,0,1])[0])
        lines[0,0].append(ax[0,0].plot(aries_time,aries_xdata['solar'],label=None,color=[1,.5,0])[0])
        lines[0,0].append(ax[0,0].plot(hopp_time2,np.zeros(len(hopp_time2)),label="Wave Generation",color=[0,0,1],alpha=0.5)[0])
        lines[0,0].append(ax[0,0].plot(hopp_time2,np.zeros(len(hopp_time2)),label="Solar Generation",color=[1,.5,0],alpha=0.5)[0])
        ax[0,0].legend()
        ax[0,0].set_ylabel('Generation [MW]')
        ax[0,0].set_ylim([0,5])

        lines[0,1].append(ax[0,1].plot(aries_time,aries_xdata['wave'],label=None,color=[0,0,1])[0])
        lines[0,1].append(ax[0,1].plot(aries_time,aries_xdata['solar'],label=None,color=[1,.5,0])[0])
        lines[0,1].append(ax[0,1].plot(hopp_time2,np.zeros(len(hopp_time2)),label="Wave Generation",color=[0,0,1],alpha=0.5)[0])
        lines[0,1].append(ax[0,1].plot(hopp_time2,np.zeros(len(hopp_time2)),label="Solar Generation",color=[1,.5,0],alpha=0.5)[0])
        ax[0,1].legend()
        ax[0,1].set_ylabel('Generation [MW]')
        ax[0,1].set_ylim([0,5])


        lines[1,0].append(ax[1,0].plot(hopp_time2,np.zeros(len(hopp_time2)),label=None,color=[0,.5,0])[0])
        lines[1,0].append(ax[1,0].plot(hopp_time2,np.zeros(len(hopp_time2)),label=None,color=[.5,0,0])[0])
        lines[1,0].append(ax[1,0].plot(hopp_time2,np.zeros(len(hopp_time2)),label=None,color=[.5,0,1])[0])
        lines[1,0].append(ax[1,0].plot(hopp_time2,np.zeros(len(hopp_time2)),label="Wind Generation",color=[0,.5,0],alpha=0.5)[0])
        lines[1,0].append(ax[1,0].plot(hopp_time2,np.zeros(len(hopp_time2)),label="Battery Generation",color=[.5,0,0],alpha=0.5)[0])
        lines[1,0].append(ax[1,0].plot(hopp_time2,np.zeros(len(hopp_time2)),label="Output to Electrolyzer",color=[.5,0,1],alpha=0.5)[0])
        ax[1,0].legend(ncol=2)
        ax[1,0].set_ylabel('Generation [MW]')
        ax[1,0].set_ylim([-120,620])
        
        lines[1,1].append(ax[1,1].plot(hopp_time2,np.zeros(len(hopp_time2)),label=None,color=[0,.5,0])[0])
        lines[1,1].append(ax[1,1].plot(hopp_time2,np.zeros(len(hopp_time2)),label=None,color=[.5,0,0])[0])
        lines[1,1].append(ax[1,1].plot(hopp_time2,np.zeros(len(hopp_time2)),label=None,color=[.5,0,1])[0])
        lines[1,1].append(ax[1,1].plot(hopp_time2,np.zeros(len(hopp_time2)),label="Wind Generation",color=[0,.5,0],alpha=0.5)[0])
        lines[1,1].append(ax[1,1].plot(hopp_time2,np.zeros(len(hopp_time2)),label="Battery Generation",color=[.5,0,0],alpha=0.5)[0])
        lines[1,1].append(ax[1,1].plot(hopp_time2,np.zeros(len(hopp_time2)),label="Output to Electrolyzer",color=[.5,0,1],alpha=0.5)[0])
        ax[1,1].legend(ncol=2)
        ax[1,1].set_ylabel('Generation [MW]')
        ax[1,1].set_ylim([-120,620])

        
        lines[2,0].append(ax[2,0].plot(aries_time,np.zeros(len(aries_time)),'k-',label='"ARIES Simulation"')[0])
        lines[2,0].append(ax[2,0].plot(hopp_time,np.zeros(len(hopp_time)),'k-',label="HOPP Simulation",alpha=0.5)[0])
        ax[2,0].legend()
        ax[2,0].set_ylabel('Battery SOC [%]')
        ax[2,0].set_ylim([0,100])

        lines[2,1].append(ax[2,1].plot(aries_time,np.zeros(len(aries_time)),'k-',label='"ARIES Simulation"')[0])
        lines[2,1].append(ax[2,1].plot(hopp_time,np.zeros(len(hopp_time)),'k-',label="HOPP Simulation",alpha=0.5)[0])
        ax[2,1].legend()
        ax[2,1].set_ylabel('Battery SOC [%]')
        ax[2,1].set_ylim([0,100])

        
        sim_start = '2019-01-05 14:00:00.0'
        sim_end = '2019-01-06 14:00:00.0'
        ax[0,0].set_xlim(pd.DatetimeIndex((sim_start,sim_end)))
        ax[1,0].set_xlim(pd.DatetimeIndex((sim_start,sim_end)))
        ax[2,0].set_xlim(pd.DatetimeIndex((sim_start,sim_end)))

         
        return hopp_time, aries_time, aries_xdata, hopp_time2, fig, ax, lines
    
    else:

        return hopp_time, aries_time, aries_xdata
    

def update_trackers(trackers, HOPPdict, ARIESdict, plotting=False):

    
    gen_dict = HOPPdict['gen']
    batt_soc = HOPPdict['soc']

    if plotting:
        hopp_time, aries_time, aries_xdata, hopp_time2, fig, ax, lines = trackers
    else:
        hopp_time, aries_time, aries_xdata = trackers

    aries_time = aries_time.append(pd.DatetimeIndex(ARIESdict['aries_time']))
    
    for i, gen_type in enumerate(["wind", "wave", "solar", "batt", "elyzer"]):
        aries_xdata[gen_type].extend(ARIESdict[gen_type])
    
    if plotting:

        # Double up the generation timepoints to make stepped plot with hopp_time2
        gen2_dict = {}
        for i, gen_type in enumerate(["wind", "wave", "pv", "batt", "elyzer"]):
            gen2 = np.vstack([gen_dict[gen_type],gen_dict[gen_type]])
            gen2 = np.reshape(np.transpose(gen2),24*2)
            gen2_dict[gen_type] = gen2
        
        lines[0,0][0].set_xdata(aries_time)
        lines[0,0][0].set_ydata([i/1000 for i in aries_xdata['wave']])
        lines[0,0][1].set_xdata(aries_time)
        lines[0,0][1].set_ydata([i/1000 for i in aries_xdata['solar']])
        lines[0,0][2].set_ydata(gen2_dict['wave']/1000)
        lines[0,0][3].set_ydata(gen2_dict['pv']/1000)
        
        lines[0,1][0].set_xdata(aries_time)
        lines[0,1][0].set_ydata([i/1000 for i in aries_xdata['wave']])
        lines[0,1][1].set_xdata(aries_time)
        lines[0,1][1].set_ydata([i/1000 for i in aries_xdata['solar']])
        lines[0,1][2].set_ydata(gen2_dict['wave']/1000)
        lines[0,1][3].set_ydata(gen2_dict['pv']/1000)
        
        
        lines[1,0][0].set_xdata(aries_time)
        lines[1,0][0].set_ydata([i/1000 for i in aries_xdata['wind']])
        lines[1,0][1].set_xdata(aries_time)
        lines[1,0][1].set_ydata([i/1000 for i in aries_xdata['batt']])
        lines[1,0][2].set_xdata(aries_time)
        lines[1,0][2].set_ydata([i/1000 for i in aries_xdata['elyzer']])
        lines[1,0][3].set_ydata(gen2_dict['wind']/1000)
        lines[1,0][4].set_ydata(gen2_dict['batt']/1000)
        lines[1,0][5].set_ydata(gen2_dict['elyzer']/1000)

        lines[1,1][0].set_xdata(aries_time)
        lines[1,1][0].set_ydata([i/1000 for i in aries_xdata['wind']])
        lines[1,1][1].set_xdata(aries_time)
        lines[1,1][1].set_ydata([i/1000 for i in aries_xdata['batt']])
        lines[1,1][2].set_xdata(aries_time)
        lines[1,1][2].set_ydata([i/1000 for i in aries_xdata['elyzer']])
        lines[1,1][3].set_ydata(gen2_dict['wind']/1000)
        lines[1,1][4].set_ydata(gen2_dict['batt']/1000)
        lines[1,1][5].set_ydata(gen2_dict['elyzer']/1000)
        

        aries_start_index = np.max([0,len(aries_time)-100])
        ax[0,1].set_xlim(pd.DatetimeIndex((aries_time[aries_start_index],aries_time[-1])))
        ax[1,1].set_xlim(pd.DatetimeIndex((aries_time[aries_start_index],aries_time[-1])))
        ax[2,1].set_xlim(pd.DatetimeIndex((aries_time[aries_start_index],aries_time[-1])))

        fig.canvas.draw() 
        fig.canvas.flush_events() 
        plt.pause(.000001)

        return hopp_time, aries_time, aries_xdata, hopp_time2, fig, ax, lines
    
    else:

        return hopp_time, aries_time, aries_xdata
    
def updateSOCplot(trackers, HOPPdict):

    hopp_time, aries_time, aries_xdata, hopp_time2, fig, ax, lines = trackers

    batt_soc = HOPPdict['soc']
    
    lines[2,0][0].set_xdata(aries_time[np.arange(1,len(aries_time),2)])
    lines[2,0][0].set_ydata(aries_xdata['soc'][1:])
    lines[2,0][1].set_ydata(batt_soc)

    lines[2,1][0].set_xdata(aries_time[np.arange(1,len(aries_time),2)])
    lines[2,1][0].set_ydata(aries_xdata['soc'][1:])
    lines[2,1][1].set_ydata(batt_soc)

    fig.canvas.draw() 
    fig.canvas.flush_events() 
    plt.pause(.000001)

    return hopp_time, aries_time, aries_xdata, hopp_time2, fig, ax, lines
        
        