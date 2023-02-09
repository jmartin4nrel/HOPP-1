MeOH_hybrid = {
    'NGCC Power Plant':{
        'Sizing Inputs':{'MWe out':[100,200,300]},
        'Sizing Model': (NETL_baseline,func6),
        'Sized Streams':['CNG in','kt/y CO2 out','etc.']
    },
    'MeOH Plant':{
        'Sizing Inputs':{'kt/y CO2 out':'NGCC Power Plant'},
        'Sizing Model': (Nyari_paper,func5),
        'Sized Streams':['ky/t H2 in','kt/y MeOH out']
    },
    'PEM Electrolyzer':{
        'Sizing Inputs':{'kt/y H2 in':'MeOH PSRs'},
        'Sizing Model': (H2A_scenarios,func1b),
        'Sized Streams':['MWe in','H2O in','etc.']
    },
    'PV Array':{
        'Sizing Inputs':{'MWe in':'PEM Electrolyzer',
                         '% PV':[20,40,60]},
        'Sizing Model': (ATB_scenarios,func2),
        'Sized Streams':['MWe capacity','MWe out est.']
    },
    'Wind Farm':{
        'Sizing Inputs':{'MWe in':'PEM Electrolyzer',
                         '% PV':'PV Array'},
        'Sizing Model': (ATB_scenarios,func3),
        'Sized Streams':['MWe capacity','MWe out est.']
    },
    'Grid Interconnection':{
        'Sizing Inputs':{'MWe capacity':'PV Array',
                         'MWe capacity':'Wind Farm',
                         'MWe in':'PEM Electrolyzer'},
        'Sizing Model': (Interconn_size_model,func4),
        'Sized Streams':['MWe capacity','etc.']
    },
}

Green_H2_hybrids = {
    'PEM Electrolyzer':{
        'Sizing Inputs':{'kt/y H2 out':365},
        'Sizing Model': (H2A_scenarios,func1),
        'Sized Streams':['MWe in','H2O in','etc.']
    },
    'PV Array':{
        'Sizing Inputs':{'MWe in':'PEM Electrolyzer',
                         '% PV':[20,40,60]},
        'Sizing Model': (ATB_scenarios,func2),
        'Sized Streams':['MWe capacity','MWe out est.']
    },
    'Wind Farm':{
        'Sizing Inputs':{'MWe in':'PEM Electrolyzer',
                         '% PV':'PV Array'},
        'Sizing Model': (ATB_scenarios,func3),
        'Sized Streams':['MWe capacity','MWe out est.']
    },
    'Grid Interconnection':{
        'Sizing Inputs':{'MWe capacity':'PV Array',
                         'MWe capacity':'Wind Farm',
                         'MWe in':'PEM Electrolyzer'},
        'Sizing Model': (Interconn_size_model,func4),
        'Sized Streams':['MWe capacity','etc.']
    },
}

Wind_add_PV_H2 = {
    'PEM Electrolyzer':{
        'Sizing Inputs':{'MWe in': ['PV Array',
                                    'Wind Farm',
                                    'Grid Interconnection']},
        'Sizing Model': (H2A_scenarios,func1a),
        'Sized Streams':['kt/y H2 out','H2O in','etc.']
    },
    'PV Array':{
        'Sizing Inputs':{'MWe capacity':[100,200,300]},
        'Sizing Model': (ATB_scenarios,func2a),
        'Sized Streams':['MWe out est.']
    },
    'Wind Farm':{
        'Sizing Inputs':{'MWe capacity':365},
        'Sizing Model': (ATB_scenarios,func3a),
        'Sized Streams':['MWe out est.']
    },
    'Grid Interconnection':{
        'Sizing Inputs':{'MWe capacity':'PV Array',
                         'MWe capacity':'Wind Farm',
                         'MWe in':'PEM Electrolyzer'},
        'Sizing Model': (Interconn_size_model,func4),
        'Sized Streams':['MWe capacity','etc.']
    },
}