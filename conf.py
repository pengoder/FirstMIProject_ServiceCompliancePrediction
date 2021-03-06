# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:51:11 2018

@author: pqian
"""

feature_list = [
'MEM_GNDR_CD',
'MEM_AGE',
'M_18',
'M_25',
'M_35',
'M_45',
'M_55',
'M_65',
'M_70',
'M_75',
'M_80',
'M_85',
'F_18',
'F_25',
'F_35',
'F_45',
'F_55',
'F_65',
'F_70',
'F_75',
'F_80',
'F_85',
'RACE_WHITE',
'RACE_BLACK',
'RACE_ASIAN',
'RACE_HISPANIC',
'IND_MONE',
'HH_INC_AVG',
'MEM_MTH',
'SML_GRP_IND',
'VEBA_IND',
'HFHS_IND',
'FEHB_IND',
'HFMG_IND',
'PART_A_RISK_ADJ_FCTR',
'TTL_LIABILITY',
'IND_HIGH_COST',
'AMT_IP',
'AMT_ER',
'COPD',
'DEPRESSION',
'CHRONIC_HEARTFAILURE',
'CHRONIC_KIDNEYDISEASE',
'DIABETES',
'HYPERTENSION',
'PCP_MEM_CNT',
'IPA_MEM_CNT',
'INDV_TIER_1_DDCTBL_AMT',
'PCP_TIER_1_COPAY_AMT',
'ER_TIER_1_COPAY_AMT',
#'PCP_COMM_ABA',
#'PCP_COMM_ART',
#'PCP_COMM_BCS',
#'PCP_COMM_COL',
#'PCP_COMM_CDC_A1C',
#'PCP_COMM_CDC_EYE',
#'PCP_COMM_CDC_NEPH',
#'PCP_COMM_OMW',
#'PCP_COMM_SPC',
'PCP_CARE_ABA',
#'PCP_CARE_ART',
#'PCP_CARE_BCS',
#'PCP_CARE_COL',
#'PCP_CARE_CDC_A1C',
#'PCP_CARE_CDC_EYE',
#'PCP_CARE_CDC_NEPH',
#'PCP_CARE_OMW',
#'PCP_CARE_SPC',
#'IPA_COMM_ABA',
#'IPA_COMM_ART',
#'IPA_COMM_BCS',
#'IPA_COMM_COL',
#'IPA_COMM_CDC_A1C',
#'IPA_COMM_CDC_EYE',
#'IPA_COMM_CDC_NEPH',
#'IPA_COMM_OMW',
#'IPA_COMM_SPC',
'IPA_CARE_ABA'
#'IPA_CARE_ART',
#'IPA_CARE_BCS'
#'IPA_CARE_COL',
#'IPA_CARE_CDC_A1C'
#'IPA_CARE_CDC_EYE',
#'IPA_CARE_CDC_NEPH',
#'IPA_CARE_OMW',
#'IPA_CARE_SPC'

]

in_logit_model = [
#    'MEM_GNDR_CD',
#    'MEM_AGE',
#    'M_18',
#    'M_25',
#    'M_35',
#    'M_45',
#    'M_55',
#    'M_65',
#    'M_70',
#    'M_75',
#    'M_80',
#    'M_85',
#    'F_18',
#    'F_25',
#    'F_35',
#    'F_45',
#    'F_55',
#    'F_65',
#    'F_70',
#    'F_75',
#    'F_80',
#    'F_85',
#    'RACE_WHITE',
#    'RACE_BLACK',
#    'RACE_ASIAN',
#    'RACE_HISPANIC',
#    'IND_MONE',
#    'HH_INC_AVG',
#    'MEM_MTH',
#    'SML_GRP_IND',
#    'VEBA_IND',
#    'HFHS_IND',
#    'FEHB_IND',
     'HFMG_IND',
#    'PART_A_RISK_ADJ_FCTR',
    'TTL_LIABILITY',
    'IND_HIGH_COST',
    'AMT_IP',
    'AMT_ER',
    'COPD',
#    'DEPRESSION',
    'CHRONIC_HEARTFAILURE',
#    'CHRONIC_KIDNEYDISEASE',
    'DIABETES',
    'HYPERTENSION',
#    'PCP_MEM_CNT',
#    'IPA_MEM_CNT',
    'INDV_TIER_1_DDCTBL_AMT',
#    'PCP_TIER_1_COPAY_AMT',
#    'ER_TIER_1_COPAY_AMT',
#    'PCP_COMM_ABA',
#    'PCP_COMM_ART',
#    'PCP_COMM_BCS',
#    'PCP_COMM_COL',
#    'PCP_COMM_CDC_A1C',
#    'PCP_COMM_CDC_EYE',
#    'PCP_COMM_CDC_NEPH',
#    'PCP_COMM_OMW',
#    'PCP_COMM_SPC',
#    'PCP_CARE_ABA',
#    'PCP_CARE_ART',
    'PCP_CARE_BCS',
#    'PCP_CARE_COL',
#    'PCP_CARE_CDC_A1C',
#    'PCP_CARE_CDC_EYE',
#    'PCP_CARE_CDC_NEPH',
#    'PCP_CARE_OMW',
#    'PCP_CARE_SPC',
#    'IPA_COMM_ABA',
#    'IPA_COMM_ART',
#    'IPA_COMM_BCS',
#    'IPA_COMM_COL',
#    'IPA_COMM_CDC_A1C',
#    'IPA_COMM_CDC_EYE',
#    'IPA_COMM_CDC_NEPH',
#    'IPA_COMM_OMW',
#    'IPA_COMM_SPC',
#    'IPA_CARE_ABA',
#    'IPA_CARE_ART',
    'IPA_CARE_BCS'
#    'IPA_CARE_COL'
#    'IPA_CARE_CDC_A1C'
#    'IPA_CARE_CDC_EYE',
#    'IPA_CARE_CDC_NEPH',
#    'IPA_CARE_OMW',
#    'IPA_CARE_SPC'
]
    
