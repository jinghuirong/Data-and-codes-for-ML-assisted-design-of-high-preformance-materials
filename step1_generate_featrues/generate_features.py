import numpy as np
import pymatgen.core as mg
from pymatgen.ext.matproj import MPRester
import pandas as pd
from time import sleep
from CBFV import composition
from pymatgen.core.structure import Structure
import os
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester
import pymatgen.core as mg
import strfoctor as sf
import decomposition as dc

df = pd.read_excel('new-data.xlsx')
decomp_list=[]
num_data=len(df)
for i in range(len(df)):
    decomp_list.append(dc.get_Decomp(df.iloc[i,3]))
df[['n0','n1','n2','n3']] = ('none','none','none','none')
df[['f0','f1','f2','f3']] = (0,0,0,0)
for i in range(len(decomp_list)):
    a=len(decomp_list[i][0])
    for j in range(a):
        if j<1:
            df['n0'].iloc[i]=decomp_list[i][0][0]
            df['f0'].iloc[i]=decomp_list[i][1][0]
        if 1<=j<2:
            df['n0'].iloc[i]=decomp_list[i][0][0]
            df['n1'].iloc[i]=decomp_list[i][0][1]
            df['f0'].iloc[i]=decomp_list[i][1][0]
            df['f1'].iloc[i]=decomp_list[i][1][1]
        if 2<=j<3:
            df['n0'].iloc[i]=decomp_list[i][0][0]
            df['n1'].iloc[i]=decomp_list[i][0][1]
            df['n2'].iloc[i]=decomp_list[i][0][2]
            df['f0'].iloc[i]=decomp_list[i][1][0]
            df['f1'].iloc[i]=decomp_list[i][1][1]
            df['f2'].iloc[i]=decomp_list[i][1][2]
        if 3<=j<4:
            df['n0'].iloc[i]=decomp_list[i][0][0]
            df['n1'].iloc[i]=decomp_list[i][0][1]
            df['n2'].iloc[i]=decomp_list[i][0][2] 
            df['n3'].iloc[i]=decomp_list[i][0][3]
            df['f0'].iloc[i]=decomp_list[i][1][0]
            df['f1'].iloc[i]=decomp_list[i][1][1]
            df['f2'].iloc[i]=decomp_list[i][1][2] 
            df['f3'].iloc[i]=decomp_list[i][1][3]

(name,value)=([],[])
for i in range(len(df)):
    structure = Structure.from_file(df['name'][i]) 
    lattice=structure.lattice
    l1=lattice.abc[0]
    l2=lattice.abc[1]
    l3=lattice.abc[2]
    s1=round(l1/3.128,0)
    s2=round(l2/3.128,0)
    s3=round(l3/5.017,0)
    if s1==2 and s1==2 and s3==1:
        structure.make_supercell([1,1,2])   
    name.append(df['name'][i])
    value.append(sf.struct_foctor(0,1,structure))
dfname=pd.DataFrame(name)
df_str=pd.DataFrame(value)
df_str.columns=['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','AB1','AB2','AB3','AB4','AB5'
                ,'u1','u2','u3','u4','u5','u6']
X,y,formulae,skipped = composition.generate_features(df,elem_prop='magpie',extend_features=False)
Nd_mean=pd.DataFrame(X, columns=['avg_NdValence','avg_GSvolume_pa','avg_NpValence','avg_CovalentRadius'])
df_str=pd.concat([df_str,Nd_mean],axis=1)
df_str.columns=['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','AB1','AB2','AB3','AB4','AB5'
                ,'u1','u2','u3','u4','u5','u6','Nd','V','Np','CR']

import json
with open( "wf.json") as f:
    _pt_data = json.load(f)
with open( "decomp_FE.json") as f:
    _FE_data = json.load(f)
with open( "ionic_radii.json") as f:
    _ir_data = json.load(f)
with open( "Vec-Z.json") as f:
    _vecZ_data = json.load(f)
with open( "polar.json") as f:
    _polar_data = json.load(f)
with open( "decomp_BG.json") as f:
    _BG_data = json.load(f)
with open( "decomp_BG.json") as f:
    _BG_data = json.load(f)

def features_generator(string):
    #elements, ratio = preprocessing(chemical)
    comp = mg.Composition(string)
    num_elements=len(comp.element_composition)
    elements=[]   # elements
    ratio=[]      # fraction
    oxydation=[]  # oxidation states
    tempcount=1

    for ii in range(num_elements):
        etemp=str(comp.elements[ii])
        elements.append(etemp)
        etemp=mg.Element(etemp)
        
        rtemp=comp.get_atomic_fraction(etemp)
        ratio.append(rtemp)
        
        oxydation.append(rtemp*np.array(etemp.common_oxidation_states))
        tempcount=tempcount*len(oxydation[ii])
    
    # composition = np.array(ratio)/sum(ratio)
    # for (a)
    # L0 = len(elements)
    # L2 = np.linalg.norm(composition,ord=2)
    (SM,SM_list,SMD_list,BG,BG_list,BGD_list)=([],[],[],[],[],[])
    (BM,BM_list,BMD_list,F_list,EA,EA_list,EAD_list,firstIE,firstIE_list,firstIED_list)=([],[],[],[],[],[],[],[],[],[])   
    (WF,WF_list,WFD_list,DS,DS_list,DSD_list,DS,DS_list,DSD_list)=([],[],[],[],[],[],[],[],[])  
    (FE,FE_list,FED_list,AR,AR_list,ARD_list,EN,EN_list,END_list)=([],[],[],[],[],[],[],[],[])  
    (IR,IR_list,IRD_list,VEZ,VEZ_list,VEZD_list,P,P_list,PD_list)=([],[],[],[],[],[],[],[],[])  
    
    for ii in range(0,num_elements):
        e1=str(comp.elements[ii])
        e2=mg.Element(e1)
        f1=comp.get_atomic_fraction(e1)
        
        L1=e2.electron_affinity        # mean electron_affinity character
        m1=e2.ionization_energies[0]   # mean 1st ionization energy character
        n1=_pt_data[e1]                # mean work function character
        o1=e2.density_of_solid         # mean Density of solid character
        r1=e2.atomic_radius_calculated # mean atomic_radius_calculated
        q1=e2.X                        # mean electronegativity difference character
        s1=_ir_data[e1]
        u1=_polar_data[e1]
        o1=e2.density_of_solid         # mean Density of solid character
        n1=_pt_data[e1]
        t1=_vecZ_data[e1]
        
        EA.append(L1)
        EA_list.append(L1*f1)

        firstIE.append(m1)
        firstIE_list.append(m1*f1)

        WF.append(n1)
        WF_list.append(n1*f1)

        DS.append(o1)
        DS_list.append(o1*f1)

        AR.append(r1)
        AR_list.append(r1*f1)
        
        
        EN.append(3.04-q1)
        EN_list.append((3.04-q1)*f1)
        END_list.append(3.04-q1)
        
        IR.append(s1/1.46)
        IR_list.append((s1/1.46)*f1)
        IRD_list.append(s1/1.46)       

        P.append(u1)
        P_list.append(u1*f1)
        PD_list.append(u1)  
        
        DS.append(o1)
        DS_list.append(o1*f1)  
        
        WF.append(n1)
        WF_list.append(n1*f1)

        VEZ.append(t1)
        VEZ_list.append(t1*f1)
        
    avg_firstIE=sum(firstIE_list)
    avg_WF=sum(WF_list)          
    avg_EN=sum(EN_list)
    avg_TF=sum(IR_list)
    avg_P=sum(P_list)
    avg_DS=sum(DS_list)  
    avg_WF=sum(WF_list)
    avg_EA=sum(EA_list) 
    avg_AR=sum(AR_list)
    avg_VEZ=sum(VEZ_list)
    return_list=[avg_firstIE, avg_TF, avg_EN, avg_P,avg_WF, avg_AR,avg_VEZ]
    return return_list
feature_names=['firstIE','TF','EN','P','WF','AR','VEZ']
feature_list=[]
num_data=len(df)
for i in range(num_data):
    feature_list.append(features_generator(df.iloc[i,1]))
df1 = pd.DataFrame(data=feature_list, columns = feature_names)
a=[df['n0'],df['n1'],df['n2'],df['n3']]
b=[df['f0'],df['f1'],df['f2'],df['f3']]
(df['FED'],df['BGD'])=(0,0)
for i in range(len(df)):
    df['FED'].iloc[i]=_FE_data[a[0].iloc[i]]*b[0].iloc[i]+_FE_data[a[1].iloc[i]]*b[1].iloc[i]
    +_FE_data[a[2].iloc[i]]*b[2].iloc[i]+_FE_data[a[3].iloc[i]]*b[3].iloc[i]

    df['BGD'].iloc[i]=_BG_data[a[0].iloc[i]]*b[0].iloc[i]+_BG_data[a[1].iloc[i]]*b[1].iloc[i]
    +_BG_data[a[2].iloc[i]]*b[2].iloc[i]+_BG_data[a[3].iloc[i]]*b[3].iloc[i]
df2=pd.concat([df[['name','formula','target']],df_str,df[['FED','BGD']],df1],axis=1)
df2['Axy']=df2[['u1','u2']].mean(axis=1)
df2['Bxy']=df2[['u4','u5']].mean(axis=1)
df2['Az']=df2['u3']
df2['Bz']=df2['u6']
df2['FIE']=df2['firstIE']
df2.drop(['u1','u2','u3','u4','u5','u6'],axis=1, inplace=True)
df2.to_csv('generated_features.csv',index=False)
