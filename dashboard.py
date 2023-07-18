import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, re, math
from matplotlib.lines import Line2D
import streamlit as st
from st_aggrid import AgGrid,GridOptionsBuilder  

from utils import *
st.set_page_config(layout="wide")

main_df_path='main_df.csv'
verify_df_path = 'verify_df.csv'
p2_sp_df_path='2p_smt_s1732_2352_3556_r1.csv'
main_df = get_df_cash(main_df_path)
verify_df = get_df_cash(verify_df_path)
p2_sp = get_df_cash(p2_sp_df_path)
# verify_df = pd.read_csv('verify_df.csv')
# p2_sp = pd.read_csv('2p_smt_s1732_2352_3556.csv')
# p2_sp = p2_sp[p2_sp['cl_ml_ratio']==1]
# # objective lm-lp
# p2_sp['obj_value']=p2_sp['lambda_minus']-p2_sp['lambda_plus']
# main_df['obj_value'] = main_df['lp_lambda_minus']-main_df['lp_lambda_plus']

#
kappa_set=np.sort(main_df['kappa'].unique()).tolist()
data_set = list(main_df['data'].unique())+['all']
seed_set = list(main_df['seed'].unique())+['all']

st.title("Pareto Optimal Report")
st.subheader('pivot table')
AgGrid(main_df, height=400)
st.write('#')

# f1_plot_tab, f1_df_tab = st.tabs(["visual", "pivot table"])
### --- Plot the ARI vs. Kappa plot
st.subheader("Pareto OptimalARI vs. Kappa plot")
p1_container = st.container()



with p1_container:
    p1_col1,p1_col2,p1_col3 = st.columns([1,3,4])
    with p1_col1:
        data_slct = st.radio("data", data_set, key="data_slct_p1", horizontal=True)
        seed_slct = st.radio("seed", seed_set, key="seed_slct_p1", horizontal=True)
        # po_line_2p(main_df,verify_df, target_data='iris', target_seed=1732)
        data_slct=data_slct if data_slct!='all' else None
        seed_slct=seed_slct if seed_slct!='all' else None
    with p1_col2:
        ari_po_2p(main_df,p2_sp, target_data=data_slct, target_seed=seed_slct)
        st.write('figure (1-a)')
        po_MinMax_obj_2p(main_df,p2_sp, target_data=data_slct, target_seed=seed_slct)
        st.write('figure (1-c)')
    with p1_col3:
        if data_slct!=None and seed_slct!=None:
            po_line_2p(main_df,verify_df, target_data=data_slct, target_seed=seed_slct, p2_sp=p2_sp)
            st.write('figure (1-b)')
        else:
            st.write('figure (1-b) pareto optimal line requires specific data and seed')

# with p1_df_container:
#     AgGrid(ptab_df, height=400)


### --- Plot the pareto optimal line
st.markdown('#')
st.markdown('--')
st.subheader("Pareto Optimal line")
container = st.container()
with container:
    col1,col2,col3 = st.columns([1,3,1])
    with col1:
        p2_data_slct = st.radio("data", data_set[:-1], key="data_slct_p2", horizontal=True)
        p2_seed_slct = st.radio("seed", seed_set[:-1], key="seed_slct_p2", horizontal=True)
    with col2:
        po_line_2p(main_df,verify_df, target_data=p2_data_slct, target_seed=p2_seed_slct, p2_sp=p2_sp)
        st.write('figure (2)')

### --- Plot the pareto optimal line
st.markdown('#')
st.markdown('--')
st.subheader("Pareto Optimal obj value vs. Kappa")
st.write(r"$\text{object value} = \lambda^- - \lambda^+$")
p3_container = st.container()
with p3_container:
    p3_col1,p3_col2,p3_col3 = st.columns([1,3,2])
    with p3_col1:
        p3_data_slct = st.radio("data", data_set, key="data_slct_p3", horizontal=True)
        p3_seed_slct = st.radio("seed", seed_set, key="seed_slct_p3", horizontal=True)
    with p3_col2:
        po_sum_time_2p(main_df, p2_sp,target_data=p3_data_slct, target_seed=p3_seed_slct)
        
        st.write('figure (3)')

