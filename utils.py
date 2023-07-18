
import numpy as np
import pandas as pd
import os,sys,re,math
import matplotlib.pyplot as plt
import streamlit as st


def query_main_verify_df(df, target_data, target_seed, target_kappa):
  return df[(df['data']==target_data)\
          &(df['seed']==target_seed)\
          &(df['kappa']==target_kappa)].copy()

def po_line_2p(main_df, verify_df, target_data, target_seed, p2_sp=None):
  kappa_set=np.sort(main_df['kappa'].unique()).tolist()
  # create fig
  fig, axs = plt.subplots(3, 3, figsize=(10, 10))
  axs = axs.ravel()
  for ax_ind, ax in enumerate(axs):
    curr_maindf = query_main_verify_df(main_df,target_data, target_seed, target_kappa=kappa_set[ax_ind])
    curr_verifydf = query_main_verify_df(verify_df,target_data, target_seed, target_kappa=kappa_set[ax_ind])
    curr_verify_lm = curr_verifydf[curr_verifydf['verify']=='lm']
    curr_verify_lp = curr_verifydf[curr_verifydf['verify']=='lp']

    ax.scatter(curr_maindf['lp_lambda_minus'], curr_maindf['lp_lambda_plus'], label=f'opt_lambda_plus',marker='o', facecolors='none',edgecolors='b',linewidths=1.5, zorder=10)
    ax.scatter(curr_maindf['lm_lambda_minus'], curr_maindf['lm_lambda_plus'], label=f'opt_lambda_minus',marker='x', color='darkorange', zorder=5)
    ax.scatter(curr_verify_lm['p2_lm_b0'], curr_verify_lm['p2_lm_b1'], label=f'MinLm', marker='^', color='g',zorder=5)
    ax.scatter(curr_verify_lp['p2_lp_b0'], curr_verify_lp['p2_lp_b1'], label=f'MaxLp',marker='v', color='r',zorder=5)
    ## add 2p
    #
    if p2_sp is not None:
      print(f"********************** {p2_sp is not None}   ----------- {p2_sp.shape}")
      curr_2pdf = query_main_verify_df(p2_sp,target_data, target_seed, target_kappa=kappa_set[ax_ind])
      ax.scatter(curr_2pdf['lambda_minus'], curr_2pdf['lambda_plus'], marker='+', color='black',zorder=15,s=100, label='2p')

    # Adjust y-axis limits with padding
    lp_list = [*curr_maindf['lm_lambda_plus'],*curr_maindf['lp_lambda_plus'],*curr_verifydf['p2_lm_b1'],*curr_verifydf['p2_lp_b1']]
    lp_min = min(lp_list) if len(lp_list)>0 else 0
    lp_max = max(lp_list) if len(lp_list)>0 else 0
    y_padding = (lp_min*0.05+ lp_max*0.05)/2
    if y_padding>0:
      ax.set_ylim(lp_min - y_padding, lp_max + y_padding)

    # Set labels and title for each subplot
    ax.set_title(f'kappa={kappa_set[ax_ind]}', fontsize=8)

  # add label
  for xlab_ind in [6,7,8]:
    axs[xlab_ind].set_xlabel('lambda_minus')
  for ylab_ind in [0,3,6]:
    axs[ylab_ind].set_ylabel('lambda_plus')

  # add legend
  axs[2].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
  axs[5].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
  axs[8].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

  # Add a big title to the entire set of plots
  fig.suptitle(f'{target_data} seed: {target_seed}  lambda_minus vs, lambda_plus', fontsize=16,y=1)
  # Adjust spacing and layout
  plt.tight_layout()

  # Display the plot
#   plt.show()
  st.pyplot(fig)


## ARI line PO
def groupby_df_ari(df, target_data=None, target_seed=None):
  tmp_df = df.copy()
  target_col_dict = {k: v for k, v in zip(['data','seed'], [target_data, target_seed]) if v!=None}
  for k,v in target_col_dict.items():
    tmp_df = tmp_df[tmp_df[k]==v]
  res =  tmp_df.groupby(by=list(target_col_dict.keys())+['kappa'], as_index=False)\
  .agg({'ARI':[('_avg','mean'), ('_min', 'min'), ('_max', 'max')]}).copy()
  res.columns = ['{}{}'.format(col, agg) for col, agg in res.columns]
  return res

def groupby_df_general_avg_ari(df,target_data=None, target_seed=None):
  tmp_df = df.copy()
  target_col_dict = {k: v for k, v in zip(['data','seed'], [target_data, target_seed]) if v!=None}
  for k,v in target_col_dict.items():
    tmp_df = tmp_df[tmp_df[k]==v]
  res = tmp_df.groupby(['kappa','data','seed'],as_index=False)\
    .agg({'ARI':[('_avg','mean'), ('_min', 'min'), ('_max', 'max')]})
  res.columns = ['{}{}'.format(col, agg) for col, agg in res.columns]
  return res.groupby(by=list(target_col_dict.keys())+['kappa'],as_index=False).mean(numeric_only=True)

def pareto_sol_count(df,target_data,target_seed ):
  tmp_df = df.copy()
  tmp_df['pareto_index'] = tmp_df['pareto_index']+1
  target_col_dict = {k: v for k, v in zip(['data','seed'], [target_data, target_seed]) if v!=None}
  for k,v in target_col_dict.items():
      tmp_df = tmp_df[tmp_df[k]==v]

  num_po_res = tmp_df.groupby(by=['kappa','data','seed'], as_index=False).max()
  return num_po_res.groupby(by=list(target_col_dict.keys())+['kappa'],as_index=False).mean(numeric_only=True)


def ari_po_2p(main_df,p2_sp,target_data,target_seed):

  kappa_set=np.sort(main_df['kappa'].unique()).tolist()
  if target_data!=None and target_seed!=None:
    # tmp_maindf_group_ari = groupby_df_ari(main_df, target_data, target_seed)
    tmp_maindf_group_ari = groupby_df_general_avg_ari(main_df,target_data ,target_seed)
  else:
    tmp_maindf_group_ari = groupby_df_general_avg_ari(main_df,target_data ,target_seed)
  tmp_p2_group_ari = groupby_df_ari(p2_sp, target_data, target_seed)

  fig, ax = plt.subplots()
  po_avg_ari, = ax.plot(tmp_maindf_group_ari['kappa'], tmp_maindf_group_ari['ARI_avg'],  color='darkblue', label='po_avg_ari',zorder=10)
  po_min_ari, = ax.plot(tmp_maindf_group_ari['kappa'], tmp_maindf_group_ari['ARI_min'], linestyle='--',color='lightsteelblue', label='po_min_ari',zorder=5)
  po_max_ari, = ax.plot(tmp_maindf_group_ari['kappa'], tmp_maindf_group_ari['ARI_max'], linestyle='--',color='lightsteelblue', label='po_max_ari',zorder=5)
  # add count of pareto optimal solution
  # add number of pareto optimal solution count
  pareto_sol_count_df = pareto_sol_count(main_df,target_data,target_seed)
  for ind in range(tmp_maindf_group_ari.shape[0]):
    ax.text(tmp_maindf_group_ari.loc[ind, 'kappa'],
    tmp_maindf_group_ari.loc[ind,'ARI_avg'],
    f'{round(pareto_sol_count_df.loc[ind, "pareto_index"],1)}',
    ha='center', va='bottom')
  
  # 2p sp avg
  p2_sp_avg, = ax.plot(tmp_p2_group_ari['kappa'], tmp_p2_group_ari['ARI_avg'],color='darkred', label='2p_avg')

  # Set the legend with different styles for each line
  ax.legend(handles=[po_avg_ari, po_max_ari,p2_sp_avg],
            labels=['po_avg_ari', 'po_max&min_ari','2p_avg_ari'],
            loc='lower right')
  # Add labels and title
  ax.set_xlabel('kappa')
  ax.set_ylabel('ARI')
  ax.set_title(f'ARI vs. Kappa -Data:{["all data avg",target_data][target_data!=None]}, Seed:{["all seeds avg",target_seed][target_seed!=None]}')

#   plt.show()
  st.pyplot(fig)




### solver time
def groupby_df_general_horrizontal(df,col_to_agg, target_data=None, target_seed=None):
  tmp_df = df.copy()
  target_col_dict = {k: v for k, v in zip(['data','seed'], [target_data, target_seed]) if v!=None}
  for k,v in target_col_dict.items():
    tmp_df = tmp_df[tmp_df[k]==v]
  res = tmp_df.groupby(['kappa','data','seed'],as_index=False)\
    .agg({col_to_agg:[('_avg','mean'), ('_min', 'min'), ('_max', 'max'),('_sum', 'sum')]})
  res.columns = ['{}{}'.format(col, agg) for col, agg in res.columns]
  return res.groupby(by=list(target_col_dict.keys())+['kappa'],as_index=False).mean(numeric_only=True)

def groupby_df_general_vertical(df, col_to_agg, target_data=None, target_seed=None):
  tmp_df = df.copy()
  target_col_dict = {k: v for k, v in zip(['data','seed'], [target_data, target_seed]) if v!=None}
  for k,v in target_col_dict.items():
    tmp_df = tmp_df[tmp_df[k]==v]
  res =  tmp_df.groupby(by=list(target_col_dict.keys())+['kappa'], as_index=False)\
  .agg({col_to_agg:[('_avg','mean'), ('_min', 'min'), ('_max', 'max'), ('_sum', 'sum')]}).copy()
  res.columns = ['{}{}'.format(col, agg) for col, agg in res.columns]
  return res

def po_sum_time_2p(main_df, p2_sp,target_data,target_seed ):
  kappa_set=np.sort(main_df['kappa'].unique()).tolist()

  tmp_maindf_group_ari = groupby_df_general_vertical(main_df,col_to_agg='total_slvr_time',target_data=target_data, target_seed=target_seed)
  tmp_p2_group_ari = groupby_df_general_vertical(p2_sp,col_to_agg='total_slvr_time', target_data=target_data, target_seed=target_seed)

  fig, ax = plt.subplots()
  po_avg_ari, = ax.plot(tmp_maindf_group_ari['kappa'], tmp_maindf_group_ari['total_slvr_time_sum'], color='darkblue', label='po_avg_ari')

  # 2p sp avg
  p2_sp_avg, = ax.plot(tmp_p2_group_ari['kappa'], tmp_p2_group_ari['total_slvr_time_avg'],color='darkred', label='2p_avg')

  # Set the legend with different styles for each line
  ax.legend(handles=[po_avg_ari,p2_sp_avg],
            labels=['po_sum_time','2p_sum_time'],
            loc='lower right')
  # Add labels and title
  ax.set_xlabel('kappa')
  ax.set_ylabel('time (s)')
  ax.set_title(f'Total Solver Time vs. Kappa -data:{["all data avg",target_data][target_data!=None]} seed:{["all seeds avg",target_seed][target_seed!=None]}')

#   plt.show()
  st.pyplot(fig)