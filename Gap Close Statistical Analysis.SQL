/* Formatted on 3/20/2018 4:14:49 PM (QP5 v5.267.14150.38573) */
/* This trial program is to, get data for a model to explain factors to close ABA gaps
Use CY2015(HEDIS2016) to fit the model
Use YTD201602 and HEDIS2016 data to test/validate the model
Regression Method:
    Logistics (this is a typical binary classification problem)
Dependent Variable
    Compliant or Not
Product Line:
    Combine all in the data pull;
    Will split into different product lines in analysis
Independent Variable/Attribute:
    A) Demongraphic
        1. Age/Sex
        2. Income
           -> external data source, average income based on ZIP
        3. Race
           -> indicator of Hispanic, according to data that we have
        4. Language
           -> Amount of languages the member speaks
        5. Living status
           -> Any dependent under this subscriber
        6. Enrollment
           -> Amount of enrolled months in the year
    B) Medical Claim (in the base year, here is CY2015)
        1. Amount of high cost claim utils (>=50,000)
        2. Amount of inpatient hospitalization
        3. Amount of ER/UC utils (in total)
        4. Amount of out-of-pocket pay (total liability)
        5. Heart Desease?
        6. Diabetes?
        7. Total cost
    B1) Pharmacy Claim (in order to look at chronical disease)
        1. 
    C) IPA/PCP
        1. Size of IPA
            -> Amount of members with HAP (standadized)
            -> Current measure' rates (standadized)
        2. Size of PCP
            -> Amount of members with HAP (standadized)
            -> Current measure' rates (standadized)
    D) Benefit: Plan/Group/Product
        1. Product
            -> PCP tier - sum of copay/coinsurance/deductible
            -> ER tier
            -> Individual tier
        2. Plan
        3. Group
            -> Size
                -> Amount of members with HAP
            -> Indicator of RMBT or HFHS
     E) PCP/IPA measure rates
        1. for selected measures
*/

CREATE OR REPLACE PACKAGE pkg_stat_mdl_gap_cls
AS
   /*
   ############################################
   Using HEDIS2017 data to test
   Standing at the beginning of 2016, to predict how members would close measure gaps
   for all data, look at 2015
   ###########################################
   */
   point_dt             INT := 20170301; -- the point of time when we want to do prediction
   var_en_bgn           DATE := TO_DATE ('20160101', 'YYYYMMDD'); -- enrollment span start date
   var_en_end           DATE := TO_DATE ('20161231', 'YYYYMMDD'); -- enrollment span end date
   num_en_bgn           INT := 20160101;
   num_en_end           INT := 20161231;
   var_rs_id            VARCHAR2 (10) := 'HEDIS2017'; -- last year (full year) results
   var_rs_yr_mnth       INT := 201712;
   var_fwd_rs_id        VARCHAR2 (10) := 'PRE201803';     -- this year results
   var_fwd_rs_yr_mnth   INT := 201803;
   batchid              INT;
   obj_nm               VARCHAR2 (10) := 'STAT_MDL';
   v_tbl                VARCHAR2 (50);

   PROCEDURE catch_batchid;

   PROCEDURE hit_demo;

   PROCEDURE hit_clm;

   PROCEDURE hit_clm_cc;

   PROCEDURE hit_pcpipa;

   PROCEDURE hit_bnft;

   PROCEDURE hit_msrrslt;

   PROCEDURE hit_master_full;

   PROCEDURE hit_master_selected;

   PROCEDURE hit_all;
END pkg_stat_mdl_gap_cls;

CREATE OR REPLACE PACKAGE BODY pkg_stat_mdl_gap_cls
AS
   PROCEDURE catch_batchid
   IS
   BEGIN
      SELECT MAX (batchid) + 1
        INTO batchid
        FROM log_debug
       WHERE objectname = obj_nm;

      IF batchid IS NULL
      THEN
         batchid := 1;
      END IF;
   END;

   PROCEDURE hit_demo
   IS
   BEGIN
      v_tbl := 'stat_mdl_gap_cls_demo';
      p_trunc_table (v_tbl);

      INSERT INTO stat_mdl_gap_cls_demo (mem_hap_id,
                                         mem_gndr_cd,
                                         mem_age,
                                         m_18,
                                         m_25,
                                         m_35,
                                         m_45,
                                         m_55,
                                         m_65,
                                         m_70,
                                         m_75,
                                         m_80,
                                         m_85,
                                         f_18,
                                         f_25,
                                         f_35,
                                         f_45,
                                         f_55,
                                         f_65,
                                         f_70,
                                         f_75,
                                         f_80,
                                         f_85,
                                         race_white,
                                         race_black,
                                         race_asian,
                                         race_hispanic,
                                         ind_mone,
                                         hh_inc_avg,
                                         mem_zipcd,
                                         mem_mth,
                                         sml_grp_ind,
                                         veba_ind,
                                         hfhs_ind,
                                         fehb_ind,
                                         hfmg_ind,
                                         part_a_risk_adj_fctr)
         -- Demongraphic
         --   -> Age/Gender, Race

         WITH mem_in_pool
              /*
           get members who are active at the beginning of the year (say Mar 1, 2016) have enrollment spans in 2015
           */
              AS (SELECT DISTINCT mem_key
                    FROM dm.cnfrmd_memshp mem
                   WHERE     point_dt BETWEEN mem.drvd_mem_elgblty_eff_dt
                                          AND mem.drvd_mem_elgblty_term_dt
                         AND mem.prod_cat_cd = 'M'
                         AND mem.mem_hap_id IN (SELECT mem.mem_hap_id
                                                  FROM dm.cnfrmd_memshp mem
                                                 WHERE     mem.prod_cat_cd =
                                                              'M'
                                                       AND mem.drvd_mem_elgblty_eff_dt <=
                                                              num_en_end
                                                       AND mem.drvd_mem_elgblty_term_dt >=
                                                              num_en_bgn)),
              mem_en
              AS (  SELECT elig.mem_key,
                           SUM (
                                (  FLOOR (
                                        (CASE
                                            WHEN elgblty_term_dt >=
                                                    TO_NUMBER (
                                                       TO_CHAR (var_en_end + 1,
                                                                'YYYYMMDD'))
                                            THEN
                                               num_en_end
                                            ELSE
                                               elgblty_term_dt
                                         END)
                                      / 100)
                                 - FLOOR (
                                        (CASE
                                            WHEN elgblty_eff_dt <=
                                                      TO_NUMBER (
                                                         TO_CHAR (
                                                            var_en_bgn + 1,
                                                            'YYYYMMDD'))
                                                    - 1
                                            THEN
                                               num_en_bgn
                                            ELSE
                                               elgblty_eff_dt
                                         END)
                                      / 100)
                                 + 1)
                              * (CASE
                                    WHEN elgblty_ind = 'Y' THEN 1
                                    WHEN elgblty_ind = 'N' THEN 0
                                 END))
                              AS mem_mth
                      FROM edw.mem_elgblty elig
                           INNER JOIN mem_in_pool pool
                              ON elig.mem_key = pool.mem_key
                           INNER JOIN edw.prod
                              ON     elig.prod_key = prod.prod_key
                                 AND prod.dw_cur_ind = 'Y'
                           INNER JOIN edw.lob
                              ON     prod.lob_key = lob.lob_key
                                 AND lob.dw_cur_ind = 'Y'
                                 AND lob.dw_lgcl_del_ind = 'N'
                                 AND lob.lob_cd NOT IN ('7000', '8000', '9999')
                     --                     INNER JOIN edw.class_plan cp
                     --                        ON     prod.prod_key = cp.prod_key
                     --                           AND SUBSTR (cp.class_id, 4, 1) <> 'W'
                     --                           AND cp.dw_cur_ind = 'Y'
                     WHERE     elig.elgblty_eff_dt <= num_en_end
                           AND elig.elgblty_term_dt >= num_en_bgn
                           AND elig.elgblty_eff_dt <= elig.elgblty_term_dt
                           AND elig.dw_cur_ind = 'Y'
                           AND elig.prod_cat_cd = 'M'
                           AND elig.elgblty_ind = 'Y'
                           AND elig.dw_lgcl_del_ind = 'N'
                  GROUP BY elig.mem_key)
           SELECT DISTINCT
                  mem.mem_hap_id,
                  -- Age/Gender
                  MAX (
                     CASE
                        WHEN mem.mem_gndr_cd = 'M' THEN 1
                        WHEN mem.mem_gndr_cd = 'F' THEN 0
                        ELSE NULL
                     END)
                     AS mem_gndr_cd,
                  MAX (
                     FLOOR (
                          MONTHS_BETWEEN (var_en_end,
                                          TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                        / 12))
                     AS mem_age,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 18
                                              AND 24
                             AND mem.mem_gndr_cd = 'M'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS m_18,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 25
                                              AND 34
                             AND mem.mem_gndr_cd = 'M'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS m_25,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 35
                                              AND 44
                             AND mem.mem_gndr_cd = 'M'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS m_35,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 45
                                              AND 54
                             AND mem.mem_gndr_cd = 'M'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS m_45,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 55
                                              AND 64
                             AND mem.mem_gndr_cd = 'M'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS m_55,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 65
                                              AND 69
                             AND mem.mem_gndr_cd = 'M'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS m_65,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 70
                                              AND 74
                             AND mem.mem_gndr_cd = 'M'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS m_70,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 75
                                              AND 79
                             AND mem.mem_gndr_cd = 'M'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS m_75,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 80
                                              AND 84
                             AND mem.mem_gndr_cd = 'M'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS m_80,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) >= 85
                             AND mem.mem_gndr_cd = 'M'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS m_85,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 18
                                              AND 24
                             AND mem.mem_gndr_cd = 'F'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS f_18,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 25
                                              AND 34
                             AND mem.mem_gndr_cd = 'F'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS f_25,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 35
                                              AND 44
                             AND mem.mem_gndr_cd = 'F'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS f_35,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 45
                                              AND 54
                             AND mem.mem_gndr_cd = 'F'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS f_45,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 55
                                              AND 64
                             AND mem.mem_gndr_cd = 'F'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS f_55,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 65
                                              AND 69
                             AND mem.mem_gndr_cd = 'F'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS f_65,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 70
                                              AND 74
                             AND mem.mem_gndr_cd = 'F'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS f_70,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 75
                                              AND 79
                             AND mem.mem_gndr_cd = 'F'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS f_75,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) BETWEEN 80
                                              AND 84
                             AND mem.mem_gndr_cd = 'F'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS f_80,
                  MAX (
                     CASE
                        WHEN     FLOOR (
                                      MONTHS_BETWEEN (
                                         var_en_end,
                                         TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                                    / 12) >= 85
                             AND mem.mem_gndr_cd = 'F'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS f_85,
                  -- Race
                  MAX (
                     CASE
                        WHEN    mmr.race_cd = '1'
                             OR UPPER (wmd.race) IN ('WHITE',
                                                     'WHITE OR CAUCASIAN')
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS race_white,
                  MAX (
                     CASE
                        WHEN    mmr.race_cd = '2'
                             OR UPPER (wmd.race) IN ('BLACK OR AFRICAN - AMERICAN',
                                                     'BLACK OR AFRICAN AMERICAN')
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS race_black,
                  MAX (
                     CASE
                        WHEN mmr.race_cd = '4' OR UPPER (wmd.race) = 'ASIAN'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS race_asian,
                  MAX (
                     CASE
                        WHEN mmr.race_cd = '5' OR UPPER (wmd.race) = 'HISPANIC'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS race_hispanic,
                  -- Living Status (has others covered)
                  MAX (sub.ind_mone) AS ind_mone,
                  -- Income
                  MAX (zpin.hh_inc_avg) AS hh_inc_avg,
                  -- Zip code
                  MAX (tst_pkg_hedis2018.format_zip (cntc.zip_cd)) AS mem_zipcd,
                  -- Number of member months
                  MAX (CASE WHEN en.mem_mth < 12 THEN 0 ELSE 1 END) AS mem_mth,
                  -- Group Size
                  MAX (
                     CASE
                        WHEN grp.drvd_grp_size_cat_cd = 'SMALL' THEN 1
                        ELSE 0
                     END)
                     AS sml_grp_ind,
                  -- VEBA
                  MAX (
                     CASE
                        WHEN grp.grp_id IN ('10000671', '10000672', '10000673')
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS veba_ind,
                  -- HFHS
                  MAX (CASE WHEN grp.grp_id = '10000014' THEN 1 ELSE 0 END)
                     AS hfhs_ind,
                  -- FEHB
                  MAX (CASE WHEN grp.grp_id = '10000017' THEN 1 ELSE 0 END)
                     AS fehb_ind,
                  MAX (
                     CASE
                        WHEN    prov.rptng_ipa_cd LIKE 'F%'
                             OR prov.rptng_ipa_cd = 'HF1'
                        THEN
                           1
                        ELSE
                           0
                     END)
                     AS hfmg_ind,
                  -- Risk Factor
                  MAX (mmr.part_a_risk_adj_fctr) AS part_a_risk_adj_fctr
             FROM edw.mem
                  INNER JOIN edw.grp
                     ON mem.grp_key = grp.grp_key AND grp.dw_cur_ind = 'Y'
                  INNER JOIN mem_en en ON mem.mem_key = en.mem_key
                  INNER JOIN
                  (  SELECT sub.sub_key,
                            sub.sub_id,
                            MAX (
                               CASE WHEN mem.mem_sfx_nbr <> 0 THEN 1 ELSE 0 END)
                               AS ind_mone
                       FROM edw.sub
                            INNER JOIN edw.mem
                               ON     sub.sub_key = mem.sub_key
                                  AND mem.dw_cur_ind = 'Y'
                      WHERE sub.dw_cur_ind = 'Y'
                   GROUP BY sub.sub_key, sub.sub_id) sub
                     ON mem.sub_key = sub.sub_key
                  LEFT OUTER JOIN edw.mem_prmry_cntct cntc
                     ON mem.mem_key = cntc.mem_key
                  LEFT OUTER JOIN
                  (SELECT mem_key,
                          prov_key,
                          RANK ()
                          OVER (PARTITION BY mem_key
                                ORDER BY mem_pcp_eff_dt DESC)
                             AS rnk
                     FROM edw.mem_pcp_rltn pcp
                    WHERE     mem_pcp_type_cd = 'MP'
                          AND mem_pcp_eff_dt <= num_en_end
                          AND mem_pcp_term_dt >= num_en_bgn
                          AND dw_cur_ind = 'Y'
                          AND dw_lgcl_del_ind = 'N') pcp
                     ON mem.mem_key = pcp.mem_key AND pcp.rnk = 1
                  LEFT OUTER JOIN edw.prov
                     ON pcp.prov_key = prov.prov_key AND prov.dw_cur_ind = 'Y'
                  LEFT OUTER JOIN ref_zip_income zpin
                     ON cntc.zip_cd = zpin.zip_cd
                  LEFT OUTER JOIN
                  (  SELECT mem_key,
                            MAX (race_cd) AS race_cd,
                            AVG (part_a_risk_adj_fctr) AS part_a_risk_adj_fctr
                       FROM (SELECT mmr.mem_key,
                                    mmr.race_cd,
                                    mmr.part_a_risk_adj_fctr,
                                    mmr.part_b_risk_adj_fctr,
                                    ROW_NUMBER ()
                                    OVER (
                                       PARTITION BY mmr.mem_key, mmr.pay_dt_key
                                       ORDER BY ROWNUM)
                                       AS rn
                               FROM edw.mth_cms_mmr_fact mmr
                              WHERE     mmr.pay_dt_key BETWEEN num_en_bgn
                                                           AND num_en_end
                                    AND mmr.mem_key <> '-1')
                      WHERE rn = 1
                   GROUP BY mem_key) mmr
                     ON mem.mem_key = mmr.mem_key
                  LEFT OUTER JOIN
                  (SELECT mem_hap_nbr,
                          race_cd.src_col_Val_desc AS race,
                          RANK ()
                          OVER (PARTITION BY mem_hap_nbr
                                ORDER BY rslt_dt DESC, hlth_risk_asmt_key)
                             AS rnk
                     FROM edw.hlth_risk_asmt hra
                          LEFT OUTER JOIN edw.src_cd_xref race_cd
                             ON hra.hap_rcl_desc_kcd =
                                   race_cd.src_cd_val_Xref_key) wmd
                     ON mem.mem_hap_id = wmd.mem_hap_nbr AND wmd.rnk = 1
            WHERE     mem.dw_cur_ind = 'Y'
                  AND   MONTHS_BETWEEN (var_en_end,
                                        TO_DATE (mem.mem_dob, 'YYYYMMDD'))
                      / 12 >= 18
         GROUP BY mem.mem_hap_id;


      COMMIT;
      write_to_log (obj_nm, v_tbl, batchid);
   END;

   --Medical Claims
   PROCEDURE hit_clm
   IS
   BEGIN
      v_tbl := 'stat_mdl_gap_cls_clm';
      p_trunc_table (v_tbl);

      INSERT INTO stat_mdl_gap_cls_clm (mem_hap_id,
                                        ttl_liability,
                                        ind_high_cost,
                                        amt_ip,
                                        amt_er)
           SELECT clm.clm_mem_hap_id AS mem_hap_id,
                  --               SUM (clm.drvd_tot_clm_alwd_amt) AS ttl_cost,
                  -- total liability
                  SUM (
                       clm.drvd_tot_clm_coins_amt
                     + clm.drvd_tot_clm_copay_amt
                     + clm.drvd_tot_clm_ddctbl_amt)
                     AS ttl_liability,
                  -- indicator of high cost claim
                  MAX (
                     CASE
                        WHEN clm.drvd_tot_clm_alwd_amt >= 50000 THEN 1
                        ELSE 0
                     END)
                     AS ind_high_cost,
                  -- amount of inpatient hospitalization
                  COUNT (
                     DISTINCT CASE
                                 WHEN     clm.clm_sub_type_cd = 'H' -- facility claims
                                      AND SUBSTR (clm.hosp_bill_type_cd, 2, 2) IN ('11',
                                                                                   '12') -- facility inpatient
                                 THEN
                                    clm.clm_mem_hap_id || clm.clm_srvc_from_dt
                              END)
                     AS amt_ip,
                  COUNT (
                     DISTINCT CASE
                                 WHEN     clm.clm_sub_type_cd = 'H'
                                      AND SUBSTR (clm.hosp_bill_type_cd, 1, 1) =
                                             'E'
                                 THEN
                                    clm.clm_mem_hap_id || clm.clm_srvc_from_dt
                              END)
                     AS amt_er
             FROM edw.med_clm clm
                  INNER JOIN stat_mdl_gap_cls_demo demo
                     ON clm.clm_mem_hap_id = demo.mem_hap_id
            WHERE     clm.clm_srvc_from_dt BETWEEN num_en_bgn AND num_en_end
                  AND clm.drvd_clm_paid_perd_dt >= clm.clm_srvc_from_dt
                  AND clm.clm_type_cd = 'M'
                  AND clm.drvd_clm_fnl_view_ind = 'Y'
                  AND clm.clm_mem_hap_id NOT LIKE '%*%'
                  AND clm.clm_line_lob_cd NOT IN ('7000', '8000', '9999')
         GROUP BY clm.clm_mem_hap_id;

      COMMIT;
      write_to_log (obj_nm, v_tbl, batchid);
   END;

   PROCEDURE hit_clm_cc
   IS
   BEGIN
      -- 65+ Chronic conditions
      v_tbl := 'stat_mdl_gap_cls_clm_cc';
      p_trunc_table (v_tbl);

      INSERT INTO stat_mdl_gap_cls_clm_cc (mem_hap_id,
                                           copd,
                                           depression,
                                           chronic_heartfailure,
                                           chronic_kidneydisease,
                                           diabetes,
                                           hypertension)
         -- Claims go back three years

         WITH vs
              AS (SELECT TRIM (REPLACE (code, '.')) AS code,
                         value_set_name,
                         RANK ()
                            OVER (PARTITION BY code ORDER BY value_set_name)
                            AS rnk
                    FROM value_set_code_2018
                   WHERE     value_set_name IN ('COPD Diagnosis',
                                                'Depression',
                                                'Chronic Heart Failure',
                                                'Chronic Kidney Disease',
                                                'Chronic Kidney Disease',
                                                'Diabetes',
                                                'Hypertension')
                         AND code_system LIKE '%ICD%CM%'),
              clm
              AS (SELECT clm_mem_hap_id, value_set_name
                    FROM dm.med_clm_line_extrct clm
                         INNER JOIN stat_mdl_gap_cls_demo demo
                            ON clm.clm_mem_hap_id = demo.mem_hap_id
                         INNER JOIN vs
                            ON clm.clm_diag_1_cd = vs.code AND rnk = 1
                   WHERE     clm.clm_prod_bsns_cat_cd = 'MA' -- only look at MA for now
                         AND clm.clm_sts_cd IN ('02', '14', '81')
                         AND clm.drvd_clm_fnl_view_ind = 'Y'
                         AND clm.clm_type_cd = 'M'
                         AND clm.clm_srvc_from_dt BETWEEN (num_en_bgn - 20000)
                                                      AND num_en_end
                         AND clm.drvd_clm_paid_perd_dt BETWEEN (  num_en_bgn
                                                                - 20000)
                                                           AND   num_en_end
                                                               + 3000),
              agt
              AS (  SELECT clm.clm_mem_hap_id AS mem_hap_id,
                           'copd' AS cc,
                           MAX (
                              CASE
                                 WHEN value_set_name = 'COPD Diagnosis' THEN 1
                                 ELSE 0
                              END)
                              AS ind
                      FROM clm
                  GROUP BY clm.clm_mem_hap_id
                  UNION ALL
                    SELECT clm.clm_mem_hap_id,
                           'depression',
                           MAX (
                              CASE
                                 WHEN value_set_name = 'Depression' THEN 1
                                 ELSE 0
                              END)
                              AS ind
                      FROM clm
                  GROUP BY clm.clm_mem_hap_id
                  UNION ALL
                    SELECT clm.clm_mem_hap_id,
                           'chronic_heartfailure',
                           MAX (
                              CASE
                                 WHEN value_set_name = 'Chronic Heart Failure'
                                 THEN
                                    1
                                 ELSE
                                    0
                              END)
                              AS ind
                      FROM clm
                  GROUP BY clm.clm_mem_hap_id
                  UNION ALL
                    SELECT clm.clm_mem_hap_id,
                           'chronic_kidneydisease',
                           MAX (
                              CASE
                                 WHEN value_set_name = 'Chronic Kidney Disease'
                                 THEN
                                    1
                                 ELSE
                                    0
                              END)
                              AS ind
                      FROM clm
                  GROUP BY clm.clm_mem_hap_id
                  UNION ALL
                    SELECT clm.clm_mem_hap_id,
                           'diabetes',
                           MAX (
                              CASE
                                 WHEN value_set_name = 'Diabetes' THEN 1
                                 ELSE 0
                              END)
                              AS ind
                      FROM clm
                  GROUP BY clm.clm_mem_hap_id
                  UNION ALL
                    SELECT clm.clm_mem_hap_id,
                           'hypertension',
                           MAX (
                              CASE
                                 WHEN value_set_name = 'Hypertension' THEN 1
                                 ELSE 0
                              END)
                              AS ind
                      FROM clm
                  GROUP BY clm.clm_mem_hap_id)
         SELECT *
           FROM agt PIVOT (MAX (ind)
                    FOR cc
                    IN  ('copd' copd,
                        'depression' depression,
                        'chronic_heartfailure' chronic_heartfailure,
                        'chronic_kidneydisease' chronic_kidneydisease,
                        'diabetes' diabetes,
                        'hypertension' hypertension));

      COMMIT;
      write_to_log (obj_nm, v_tbl, batchid);
   END;

   PROCEDURE hit_pcpipa
   AS
   BEGIN
      --PCP or IPA size
      v_tbl := 'stat_mdl_gap_cls_pcp';
      p_trunc_table (v_tbl);

      INSERT INTO stat_mdl_gap_cls_pcp (mem_hap_id, pcp_mem_cnt, ipa_mem_cnt)
           SELECT mem_hap_id,
                  MAX (pcp_mem_cnt) AS pcp_mem_cnt,
                  MAX (ipa_mem_cnt) AS ipa_mem_cnt
             FROM (SELECT mem.mem_hap_id,
                          COUNT (DISTINCT mem.mem_hap_id)
                             OVER (PARTITION BY pcp_id)
                             AS pcp_mem_cnt,
                          COUNT (DISTINCT mem.mem_hap_id)
                             OVER (PARTITION BY rptng_ipa_id)
                             AS ipa_mem_cnt
                     FROM dm.cnfrmd_memshp mem
                          INNER JOIN stat_mdl_gap_cls_demo demo
                             ON mem.mem_hap_id = demo.mem_hap_id
                    --      INNER JOIN (select mem_hap_id,
                    --           COUNT (DISTINCT mem_hap_id) OVER (PARTITION BY grp_id) AS grp_mem_cnt,
                    --           rank() over (partition by mem_hap_id order by mem_sfx_nbr) as rnk
                    --                from dm.cnfrmd_memshp cmem
                    --               where prod_cat_cd = 'M'
                    --           AND var_en_end BETWEEN drvd_mem_elgblty_eff_dt
                    --                            AND drvd_mem_elgblty_term_dt) cmem
                    --      on mem.mem_hap_id=cmem.mem_hap_id
                    --      and rnk=1
                    WHERE     prod_cat_cd = 'M'
                          AND drvd_mem_elgblty_eff_dt <= num_en_end
                          AND drvd_mem_elgblty_term_dt >= num_en_bgn
                          AND drvd_mem_elgblty_eff_dt <=
                                 drvd_mem_elgblty_term_dt)
         GROUP BY mem_hap_id;

      COMMIT;
      write_to_log (obj_nm, v_tbl, batchid);
   END;

   -- Member Plan Benefits
   PROCEDURE hit_bnft
   IS
   BEGIN
      v_tbl := 'stat_mdl_gap_cls_bnft';
      p_trunc_table (v_tbl);

      INSERT INTO stat_mdl_gap_cls_bnft (mem_hap_id,
                                         indv_tier_1_ddctbl_amt,
                                         pcp_tier_1_copay_amt,
                                         er_tier_1_copay_amt)
           SELECT mem.mem_hap_id,
                  --deductible
                  MAX (bnft.indv_tier_1_ddctbl_amt) AS indv_tier_1_ddctbl_amt,
                  --pcp copay
                  MAX (bnft.pcp_tier_1_copay_amt) AS pcp_tier_1_copay_amt,
                  --er copay
                  MAX (bnft.er_tier_1_copay_amt) AS er_tier_1_copay_amt
             FROM dm.cnfrmd_memshp mem
                  INNER JOIN dm.prod_bnft_smry_extrct bnft
                     ON     mem.prod_key = bnft.prod_key
                        AND bnft.dw_lgcl_del_ind = 'N'
                        AND num_en_end BETWEEN bnft.prod_bnft_smry_eff_dt
                                           AND bnft.prod_bnft_smry_term_dt
                  INNER JOIN stat_mdl_gap_cls_demo demo
                     ON mem.mem_hap_id = demo.mem_hap_id
            WHERE     mem.prod_cat_cd = 'M'
                  AND drvd_mem_elgblty_eff_dt <= num_en_end
                  AND drvd_mem_elgblty_term_dt >= num_en_bgn
                  AND drvd_mem_elgblty_eff_dt <= drvd_mem_elgblty_term_dt
         GROUP BY mem.mem_hap_id;

      COMMIT;
      write_to_log (obj_nm, v_tbl, batchid);
   END;

   /*PCP and IPA measure rates for the year before*/
   PROCEDURE hit_msrrslt
   AS
   BEGIN
      v_tbl := 'stat_mdl_gap_cls_msrrslt';

      p_trunc_table (v_tbl);

      INSERT INTO stat_mdl_gap_cls_msrrslt (mem_hap_id,
                                            pcp_comm_aba,
                                            pcp_comm_art,
                                            pcp_comm_bcs,
                                            pcp_comm_col,
                                            pcp_comm_cdc_a1c,
                                            pcp_comm_cdc_eye,
                                            pcp_comm_cdc_neph,
                                            pcp_comm_omw,
                                            pcp_comm_spc,
                                            pcp_care_aba,
                                            pcp_care_art,
                                            pcp_care_bcs,
                                            pcp_care_col,
                                            pcp_care_cdc_a1c,
                                            pcp_care_cdc_eye,
                                            pcp_care_cdc_neph,
                                            pcp_care_omw,
                                            pcp_care_spc,
                                            ipa_comm_aba,
                                            ipa_comm_art,
                                            ipa_comm_bcs,
                                            ipa_comm_col,
                                            ipa_comm_cdc_a1c,
                                            ipa_comm_cdc_eye,
                                            ipa_comm_cdc_neph,
                                            ipa_comm_omw,
                                            ipa_comm_spc,
                                            ipa_care_aba,
                                            ipa_care_art,
                                            ipa_care_bcs,
                                            ipa_care_col,
                                            ipa_care_cdc_a1c,
                                            ipa_care_cdc_eye,
                                            ipa_care_cdc_neph,
                                            ipa_care_omw,
                                            ipa_care_spc)
         WITH msr_rslt
              AS (SELECT ds.mem_hap_nbr,
                         ds.hedis_population,
                         ds.msr_cd,
                         ds.sub_msr_desc,
                         ds.admin_numerator,
                         ds.admin_denominator
                    FROM dim_sum_ref ds
                   WHERE     ds.revw_set_id = var_rs_id
                         AND ds.revw_set_yr_mnth = var_rs_yr_mnth
                         AND ds.hedis_population IN ('Commercial HMO POS',
                                                     'Medicare Adv HMO POS')
                         AND ds.msr_cd IN ('ABA',
                                           'ART',
                                           'BCS',
                                           'COL',
                                           'CDC',
                                           'OMW',
                                           'SPC')
                         AND (   ds.sub_msr_desc IS NULL
                              OR ds.sub_msr_desc IN ('HbA1c Poor Control',
                                                     'Eye Exam',
                                                     'Medical Attention For Nephropathy',
                                                     'Received Statin Therapy'))),
              pcp_msr_rslt
              AS (SELECT pcp_id,
                         /* convert all NULL values to 0, facilitate calculations later*/
                         NVL (comm_aba, 0) AS comm_aba,
                         NVL (comm_art, 0) AS comm_art,
                         NVL (comm_bcs, 0) AS comm_bcs,
                         NVL (comm_col, 0) AS comm_col,
                         NVL (comm_cdc_a1c, 0) AS comm_cdc_a1c,
                         NVL (comm_cdc_eye, 0) AS comm_cdc_eye,
                         NVL (comm_cdc_neph, 0) AS comm_cdc_neph,
                         NVL (comm_omw, 0) AS comm_omw,
                         NVL (comm_spc, 0) AS comm_spc,
                         NVL (care_aba, 0) AS care_aba,
                         NVL (care_art, 0) AS care_art,
                         NVL (care_bcs, 0) AS care_bcs,
                         NVL (care_col, 0) AS care_col,
                         NVL (care_cdc_a1c, 0) AS care_cdc_a1c,
                         NVL (care_cdc_eye, 0) AS care_cdc_eye,
                         NVL (care_cdc_neph, 0) AS care_cdc_neph,
                         NVL (care_omw, 0) AS care_omw,
                         NVL (care_spc, 0) AS care_spc
                    FROM (SELECT *
                            FROM (  SELECT mem.pcp_id,
                                           ds.hedis_population,
                                           ds.msr_cd,
                                           NVL (TRIM (ds.sub_msr_desc), 'NULL')
                                              AS sub_msr_desc,
                                           NVL (
                                              CASE
                                                 WHEN SUM (
                                                         ds.admin_denominator)
                                                         IS NOT NULL
                                                 THEN
                                                    CASE
                                                       WHEN ds.sub_msr_desc =
                                                               'HbA1c Poor Control'
                                                       THEN
                                                            1
                                                          -   SUM (
                                                                 ds.admin_numerator)
                                                            / SUM (
                                                                 ds.admin_denominator)
                                                       ELSE
                                                            SUM (
                                                               ds.admin_numerator)
                                                          / SUM (
                                                               ds.admin_denominator)
                                                    END
                                                 ELSE
                                                    0
                                              END,
                                              0)
                                              AS rate
                                      FROM msr_rslt ds
                                           INNER JOIN dm.cnfrmd_memshp mem
                                              ON     ds.mem_hap_nbr =
                                                        mem.mem_hap_id
                                                 AND mem.prod_cat_cd = 'M'
                                                 AND num_en_end BETWEEN mem.drvd_mem_elgblty_eff_dt
                                                                    AND mem.drvd_mem_elgblty_term_dt
                                  GROUP BY mem.pcp_id,
                                           ds.hedis_population,
                                           ds.msr_cd,
                                           ds.sub_msr_desc) PIVOT (SUM (rate) rate
                                                            FOR (
                                                               msr_cd,
                                                               sub_msr_desc)
                                                            IN  ( (
                                                               'ABA', 'NULL') aba,
                                                                (
                                                               'ART', 'NULL') art,
                                                                (
                                                               'BCS', 'NULL') bcs,
                                                                (
                                                               'COL', 'NULL') col,
                                                                (
                                                               'CDC',
                                                               'HbA1c Poor Control') cdc_a1c,
                                                                (
                                                               'CDC',
                                                               'Eye Exam') cdc_eye,
                                                                (
                                                               'CDC',
                                                               'Medical Attention For Nephropathy') cdc_neph,
                                                                (
                                                               'OMW', 'NULL') omw,
                                                                (
                                                               'SPC',
                                                               'Received Statin Therapy') spc))) PIVOT (SUM (
                                                                                                           aba_rate) aba,
                                                                                                       SUM (
                                                                                                          art_rate) art,
                                                                                                       SUM (
                                                                                                          bcs_rate) bcs,
                                                                                                       SUM (
                                                                                                          col_rate) col,
                                                                                                       SUM (
                                                                                                          cdc_a1c_rate) cdc_a1c,
                                                                                                       SUM (
                                                                                                          cdc_eye_rate) cdc_eye,
                                                                                                       SUM (
                                                                                                          cdc_neph_rate) cdc_neph,
                                                                                                       SUM (
                                                                                                          omw_rate) omw,
                                                                                                       SUM (
                                                                                                          spc_rate) spc
                                                                                                 FOR hedis_population
                                                                                                 IN  ('Commercial HMO POS' comm,
                                                                                                     'Medicare Adv HMO POS' care))),
              ipa_msr_rslt
              AS (SELECT rptng_ipa_id,
                         /* convert all NULL values to 0, facilitate calculations later*/
                         NVL (comm_aba, 0) AS comm_aba,
                         NVL (comm_art, 0) AS comm_art,
                         NVL (comm_bcs, 0) AS comm_bcs,
                         NVL (comm_col, 0) AS comm_col,
                         NVL (comm_cdc_a1c, 0) AS comm_cdc_a1c,
                         NVL (comm_cdc_eye, 0) AS comm_cdc_eye,
                         NVL (comm_cdc_neph, 0) AS comm_cdc_neph,
                         NVL (comm_omw, 0) AS comm_omw,
                         NVL (comm_spc, 0) AS comm_spc,
                         NVL (care_aba, 0) AS care_aba,
                         NVL (care_art, 0) AS care_art,
                         NVL (care_bcs, 0) AS care_bcs,
                         NVL (care_col, 0) AS care_col,
                         NVL (care_cdc_a1c, 0) AS care_cdc_a1c,
                         NVL (care_cdc_eye, 0) AS care_cdc_eye,
                         NVL (care_cdc_neph, 0) AS care_cdc_neph,
                         NVL (care_omw, 0) AS care_omw,
                         NVL (care_spc, 0) AS care_spc
                    FROM (SELECT *
                            FROM (  SELECT mem.rptng_ipa_id,
                                           ds.hedis_population,
                                           ds.msr_cd,
                                           NVL (TRIM (ds.sub_msr_desc), 'NULL')
                                              AS sub_msr_desc,
                                           NVL (
                                              CASE
                                                 WHEN SUM (
                                                         ds.admin_denominator)
                                                         IS NOT NULL
                                                 THEN
                                                    CASE
                                                       WHEN ds.sub_msr_desc =
                                                               'HbA1c Poor Control'
                                                       THEN
                                                            1
                                                          -   SUM (
                                                                 ds.admin_numerator)
                                                            / SUM (
                                                                 ds.admin_denominator)
                                                       ELSE
                                                            SUM (
                                                               ds.admin_numerator)
                                                          / SUM (
                                                               ds.admin_denominator)
                                                    END
                                                 ELSE
                                                    0
                                              END,
                                              0)
                                              AS rate
                                      FROM msr_rslt ds
                                           INNER JOIN dm.cnfrmd_memshp mem
                                              ON     ds.mem_hap_nbr =
                                                        mem.mem_hap_id
                                                 AND mem.prod_cat_cd = 'M'
                                                 AND num_en_end BETWEEN mem.drvd_mem_elgblty_eff_dt
                                                                    AND mem.drvd_mem_elgblty_term_dt
                                  GROUP BY mem.rptng_ipa_id,
                                           ds.hedis_population,
                                           ds.msr_cd,
                                           ds.sub_msr_desc) PIVOT (SUM (rate) rate
                                                            FOR (
                                                               msr_cd,
                                                               sub_msr_desc)
                                                            IN  ( (
                                                               'ABA', 'NULL') aba,
                                                                (
                                                               'ART', 'NULL') art,
                                                                (
                                                               'BCS', 'NULL') bcs,
                                                                (
                                                               'COL', 'NULL') col,
                                                                (
                                                               'CDC',
                                                               'HbA1c Poor Control') cdc_a1c,
                                                                (
                                                               'CDC',
                                                               'Eye Exam') cdc_eye,
                                                                (
                                                               'CDC',
                                                               'Medical Attention For Nephropathy') cdc_neph,
                                                                (
                                                               'OMW', 'NULL') omw,
                                                                (
                                                               'SPC',
                                                               'Received Statin Therapy') spc))) PIVOT (SUM (
                                                                                                           aba_rate) aba,
                                                                                                       SUM (
                                                                                                          art_rate) art,
                                                                                                       SUM (
                                                                                                          bcs_rate) bcs,
                                                                                                       SUM (
                                                                                                          col_rate) col,
                                                                                                       SUM (
                                                                                                          cdc_a1c_rate) cdc_a1c,
                                                                                                       SUM (
                                                                                                          cdc_eye_rate) cdc_eye,
                                                                                                       SUM (
                                                                                                          cdc_neph_rate) cdc_neph,
                                                                                                       SUM (
                                                                                                          omw_rate) omw,
                                                                                                       SUM (
                                                                                                          spc_rate) spc
                                                                                                 FOR hedis_population
                                                                                                 IN  ('Commercial HMO POS' comm,
                                                                                                     'Medicare Adv HMO POS' care))),
              pcp_ipa_msr_rslt
              AS (SELECT DISTINCT mem.mem_hap_id,
                                  pcp.comm_aba AS pcp_comm_aba,
                                  pcp.comm_art AS pcp_comm_art,
                                  pcp.comm_bcs AS pcp_comm_bcs,
                                  pcp.comm_col AS pcp_comm_col,
                                  pcp.comm_cdc_a1c AS pcp_comm_cdc_a1c,
                                  pcp.comm_cdc_eye AS pcp_comm_cdc_eye,
                                  pcp.comm_cdc_neph AS pcp_comm_cdc_neph,
                                  pcp.comm_omw AS pcp_comm_omw,
                                  pcp.comm_spc AS pcp_comm_spc,
                                  pcp.care_aba AS pcp_care_aba,
                                  pcp.care_art AS pcp_care_art,
                                  pcp.care_bcs AS pcp_care_bcs,
                                  pcp.care_col AS pcp_care_col,
                                  pcp.care_cdc_a1c AS pcp_care_cdc_a1c,
                                  pcp.care_cdc_eye AS pcp_care_cdc_eye,
                                  pcp.care_cdc_neph AS pcp_care_cdc_neph,
                                  pcp.care_omw AS pcp_care_omw,
                                  pcp.care_spc AS pcp_care_spc,
                                  ipa.comm_aba AS ipa_comm_aba,
                                  ipa.comm_art AS ipa_comm_art,
                                  ipa.comm_bcs AS ipa_comm_bcs,
                                  ipa.comm_col AS ipa_comm_col,
                                  ipa.comm_cdc_a1c AS ipa_comm_cdc_a1c,
                                  ipa.comm_cdc_eye AS ipa_comm_cdc_eye,
                                  ipa.comm_cdc_neph AS ipa_comm_cdc_neph,
                                  ipa.comm_omw AS ipa_comm_omw,
                                  ipa.comm_spc AS ipa_comm_spc,
                                  ipa.care_aba AS ipa_care_aba,
                                  ipa.care_art AS ipa_care_art,
                                  ipa.care_bcs AS ipa_care_bcs,
                                  ipa.care_col AS ipa_care_col,
                                  ipa.care_cdc_a1c AS ipa_care_cdc_a1c,
                                  ipa.care_cdc_eye AS ipa_care_cdc_eye,
                                  ipa.care_cdc_neph AS ipa_care_cdc_neph,
                                  ipa.care_omw AS ipa_care_omw,
                                  ipa.care_spc AS ipa_care_spc
                    FROM dm.cnfrmd_memshp mem
                         INNER JOIN pcp_msr_rslt pcp
                            ON mem.pcp_id = pcp.pcp_id
                         INNER JOIN ipa_msr_rslt ipa
                            ON mem.rptng_ipa_id = ipa.rptng_ipa_id
                   WHERE     mem.prod_cat_cd = 'M'
                         AND mem.drvd_mem_elgblty_eff_dt <= num_en_end
                         AND mem.drvd_mem_elgblty_term_dt >= num_en_bgn)
           SELECT ds.mem_hap_nbr,
                  MAX (pcp_comm_aba) AS pcp_comm_aba,
                  MAX (pcp_comm_art) AS pcp_comm_art,
                  MAX (pcp_comm_bcs) AS pcp_comm_bcs,
                  MAX (pcp_comm_col) AS pcp_comm_col,
                  MAX (pcp_comm_cdc_a1c) AS pcp_comm_cdc_a1c,
                  MAX (pcp_comm_cdc_eye) AS pcp_comm_cdc_eye,
                  MAX (pcp_comm_cdc_neph) AS pcp_comm_cdc_neph,
                  MAX (pcp_comm_omw) AS pcp_comm_omw,
                  MAX (pcp_comm_spc) AS pcp_comm_spc,
                  MAX (pcp_care_aba) AS pcp_care_aba,
                  MAX (pcp_care_art) AS pcp_care_art,
                  MAX (pcp_care_bcs) AS pcp_care_bcs,
                  MAX (pcp_care_col) AS pcp_care_col,
                  MAX (pcp_care_cdc_a1c) AS pcp_care_cdc_a1c,
                  MAX (pcp_care_cdc_eye) AS pcp_care_cdc_eye,
                  MAX (pcp_care_cdc_neph) AS pcp_care_cdc_neph,
                  MAX (pcp_care_omw) AS pcp_care_omw,
                  MAX (pcp_care_spc) AS pcp_care_spc,
                  MAX (ipa_comm_aba) AS ipa_comm_aba,
                  MAX (ipa_comm_art) AS ipa_comm_art,
                  MAX (ipa_comm_bcs) AS ipa_comm_bcs,
                  MAX (ipa_comm_col) AS ipa_comm_col,
                  MAX (ipa_comm_cdc_a1c) AS ipa_comm_cdc_a1c,
                  MAX (ipa_comm_cdc_eye) AS ipa_comm_cdc_eye,
                  MAX (ipa_comm_cdc_neph) AS ipa_comm_cdc_neph,
                  MAX (ipa_comm_omw) AS ipa_comm_omw,
                  MAX (ipa_comm_spc) AS ipa_comm_spc,
                  MAX (ipa_care_aba) AS ipa_care_aba,
                  MAX (ipa_care_aba) AS ipa_care_art,
                  MAX (ipa_care_bcs) AS ipa_care_bcs,
                  MAX (ipa_care_col) AS ipa_care_col,
                  MAX (ipa_care_cdc_a1c) AS ipa_care_cdc_a1c,
                  MAX (ipa_care_cdc_eye) AS ipa_care_cdc_eye,
                  MAX (ipa_care_cdc_neph) AS ipa_care_cdc_neph,
                  MAX (ipa_care_omw) AS ipa_care_omw,
                  MAX (ipa_care_spc) AS ipa_care_spc
             FROM msr_rslt ds
                  INNER JOIN pcp_ipa_msr_rslt rslt
                     ON ds.mem_hap_nbr = rslt.mem_hap_id
         GROUP BY ds.mem_hap_nbr;

      COMMIT;
      write_to_log (obj_nm, v_tbl, batchid);
   END;

   PROCEDURE hit_master_full
   AS
   BEGIN
      v_tbl := 'stat_mdl_gap_cls_master';
      p_trunc_table (v_tbl);

      INSERT INTO stat_mdl_gap_cls_master (mem_hap_id,
                                           hedis_population,
                                           msr_desc,
                                           admin_numerator,
                                           mem_gndr_cd,
                                           mem_age,
                                           m_18,
                                           m_25,
                                           m_35,
                                           m_45,
                                           m_55,
                                           m_65,
                                           m_70,
                                           m_75,
                                           m_80,
                                           m_85,
                                           f_18,
                                           f_25,
                                           f_35,
                                           f_45,
                                           f_55,
                                           f_65,
                                           f_70,
                                           f_75,
                                           f_80,
                                           f_85,
                                           race_white,
                                           race_black,
                                           race_asian,
                                           race_hispanic,
                                           ind_mone,
                                           hh_inc_avg,
                                           mem_zipcd,
                                           mem_mth,
                                           sml_grp_ind,
                                           veba_ind,
                                           hfhs_ind,
                                           fehb_ind,
                                           hfmg_ind,
                                           part_a_risk_adj_fctr,
                                           ttl_liability,
                                           ind_high_cost,
                                           amt_ip,
                                           amt_er,
                                           copd,
                                           depression,
                                           chronic_heartfailure,
                                           chronic_kidneydisease,
                                           diabetes,
                                           hypertension,
                                           pcp_mem_cnt,
                                           ipa_mem_cnt,
                                           indv_tier_1_ddctbl_amt,
                                           pcp_tier_1_copay_amt,
                                           er_tier_1_copay_amt,
                                           pcp_comm_aba,
                                           pcp_comm_art,
                                           pcp_comm_bcs,
                                           pcp_comm_col,
                                           pcp_comm_cdc_a1c,
                                           pcp_comm_cdc_eye,
                                           pcp_comm_cdc_neph,
                                           pcp_comm_omw,
                                           pcp_comm_spc,
                                           pcp_care_aba,
                                           pcp_care_art,
                                           pcp_care_bcs,
                                           pcp_care_col,
                                           pcp_care_cdc_a1c,
                                           pcp_care_cdc_eye,
                                           pcp_care_cdc_neph,
                                           pcp_care_omw,
                                           pcp_care_spc,
                                           ipa_comm_aba,
                                           ipa_comm_art,
                                           ipa_comm_bcs,
                                           ipa_comm_col,
                                           ipa_comm_cdc_a1c,
                                           ipa_comm_cdc_eye,
                                           ipa_comm_cdc_neph,
                                           ipa_comm_omw,
                                           ipa_comm_spc,
                                           ipa_care_aba,
                                           ipa_care_art,
                                           ipa_care_bcs,
                                           ipa_care_col,
                                           ipa_care_cdc_a1c,
                                           ipa_care_cdc_eye,
                                           ipa_care_cdc_neph,
                                           ipa_care_omw,
                                           ipa_care_spc)
         WITH msr_rslt
              AS (SELECT ds.mem_hap_nbr,
                         ds.hedis_population,
                         ds.msr_cd,
                         ds.sub_msr_desc,
                         ds.admin_numerator,
                         ds.admin_denominator
                    FROM dim_sum_ref ds
                   WHERE     ds.revw_set_id = var_fwd_rs_id
                         AND ds.revw_set_yr_mnth = var_fwd_rs_yr_mnth
                         AND ds.hedis_population IN ('Commercial HMO POS',
                                                     'Medicare Adv HMO POS')
                         AND ds.msr_cd IN ('ABA',
                                           'ART',
                                           'BCS',
                                           'COL',
                                           'CDC',
                                           'OMW',
                                           'SPC')
                         AND (   ds.sub_msr_desc IS NULL
                              OR ds.sub_msr_desc IN ('HbA1c Poor Control',
                                                     'Eye Exam',
                                                     'Medical Attention For Nephropathy',
                                                     'Received Statin Therapy'))),
              slct_msr
              AS (  SELECT mem_hap_nbr,
                           hedis_population,
                           msr_cd,
                           sub_msr_desc,
                           MAX (admin_numerator) AS admin_numerator
                      FROM msr_rslt
                  GROUP BY mem_hap_nbr,
                           hedis_population,
                           msr_cd,
                           sub_msr_desc)
         SELECT ds.mem_hap_nbr,
                ds.hedis_population,
                CASE
                   WHEN ds.sub_msr_desc IS NULL THEN ds.msr_cd
                   ELSE ds.msr_cd || ' - ' || ds.sub_msr_desc
                END
                   AS msr_desc,
                CASE
                   WHEN ds.sub_msr_desc = 'HbA1c Poor Control'
                   THEN
                      ABS (ds.admin_numerator - 1)
                   ELSE
                      ds.admin_numerator
                END
                   AS admin_numerator,
                demo.mem_gndr_cd,
                demo.mem_age,
                demo.m_18,
                demo.m_25,
                demo.m_35,
                demo.m_45,
                demo.m_55,
                demo.m_65,
                demo.m_70,
                demo.m_75,
                demo.m_80,
                demo.m_85,
                demo.f_18,
                demo.f_25,
                demo.f_35,
                demo.f_45,
                demo.f_55,
                demo.f_65,
                demo.f_70,
                demo.f_75,
                demo.f_80,
                demo.f_85,
                demo.race_white,
                demo.race_black,
                demo.race_asian,
                demo.race_hispanic,
                demo.ind_mone,
                demo.hh_inc_avg,
                demo.mem_zipcd,
                demo.mem_mth,
                demo.sml_grp_ind,
                demo.veba_ind,
                demo.hfhs_ind,
                demo.fehb_ind,
                demo.hfmg_ind,
                demo.part_a_risk_adj_fctr,
                clm.ttl_liability,
                clm.ind_high_cost,
                clm.amt_ip,
                clm.amt_er,
                cmcc.copd,
                cmcc.depression,
                cmcc.chronic_heartfailure,
                cmcc.chronic_kidneydisease,
                cmcc.diabetes,
                cmcc.hypertension,
                pcp.pcp_mem_cnt,
                pcp.ipa_mem_cnt,
                bnft.indv_tier_1_ddctbl_amt,
                bnft.pcp_tier_1_copay_amt,
                bnft.er_tier_1_copay_amt,
                pcp_comm_aba,
                pcp_comm_art,
                pcp_comm_bcs,
                pcp_comm_col,
                pcp_comm_cdc_a1c,
                pcp_comm_cdc_eye,
                pcp_comm_cdc_neph,
                pcp_comm_omw,
                pcp_comm_spc,
                pcp_care_aba,
                pcp_care_art,
                pcp_care_bcs,
                pcp_care_col,
                pcp_care_cdc_a1c,
                pcp_care_cdc_eye,
                pcp_care_cdc_neph,
                pcp_care_omw,
                pcp_care_spc,
                ipa_comm_aba,
                ipa_comm_art,
                ipa_comm_bcs,
                ipa_comm_col,
                ipa_comm_cdc_a1c,
                ipa_comm_cdc_eye,
                ipa_comm_cdc_neph,
                ipa_comm_omw,
                ipa_comm_spc,
                ipa_care_aba,
                ipa_care_art,
                ipa_care_bcs,
                ipa_care_col,
                ipa_care_cdc_a1c,
                ipa_care_cdc_eye,
                ipa_care_cdc_neph,
                ipa_care_omw,
                ipa_care_spc
           FROM slct_msr ds
                INNER JOIN stat_mdl_gap_cls_demo demo
                   ON ds.mem_hap_nbr = demo.mem_hap_id
                INNER JOIN stat_mdl_gap_cls_pcp pcp
                   ON ds.mem_hap_nbr = pcp.mem_hap_id
                INNER JOIN stat_mdl_gap_cls_bnft bnft
                   ON ds.mem_hap_nbr = bnft.mem_hap_id
                LEFT OUTER JOIN stat_mdl_gap_cls_msrrslt rslt
                   ON ds.mem_hap_nbr = rslt.mem_hap_id
                LEFT OUTER JOIN stat_mdl_gap_cls_clm clm
                   ON ds.mem_hap_nbr = clm.mem_hap_id
                LEFT OUTER JOIN stat_mdl_gap_cls_clm_cc cmcc
                   ON ds.mem_hap_nbr = cmcc.mem_hap_id;

      COMMIT;
      write_to_log (obj_nm, v_tbl, batchid);
   END;


   /*Only look at the gap that was open at the certain point of time of year*/
   PROCEDURE hit_master_selected
   AS
   BEGIN
      v_tbl := 'stat_mdl_gap_cls_master';
      p_trunc_table (v_tbl);

      INSERT INTO stat_mdl_gap_cls_master (mem_hap_id,
                                           hedis_population,
                                           msr_desc,
                                           admin_numerator,
                                           mem_gndr_cd,
                                           mem_age,
                                           m_18,
                                           m_25,
                                           m_35,
                                           m_45,
                                           m_55,
                                           m_65,
                                           m_70,
                                           m_75,
                                           m_80,
                                           m_85,
                                           f_18,
                                           f_25,
                                           f_35,
                                           f_45,
                                           f_55,
                                           f_65,
                                           f_70,
                                           f_75,
                                           f_80,
                                           f_85,
                                           race_white,
                                           race_black,
                                           race_asian,
                                           race_hispanic,
                                           ind_mone,
                                           hh_inc_avg,
                                           mem_zipcd,
                                           mem_mth,
                                           sml_grp_ind,
                                           veba_ind,
                                           hfhs_ind,
                                           fehb_ind,
                                           hfmg_ind,
                                           part_a_risk_adj_fctr,
                                           ttl_liability,
                                           ind_high_cost,
                                           amt_ip,
                                           amt_er,
                                           copd,
                                           depression,
                                           chronic_heartfailure,
                                           chronic_kidneydisease,
                                           diabetes,
                                           hypertension,
                                           pcp_mem_cnt,
                                           ipa_mem_cnt,
                                           indv_tier_1_ddctbl_amt,
                                           pcp_tier_1_copay_amt,
                                           er_tier_1_copay_amt,
                                           pcp_comm_aba,
                                           pcp_comm_art,
                                           pcp_comm_bcs,
                                           pcp_comm_col,
                                           pcp_comm_cdc_a1c,
                                           pcp_comm_cdc_eye,
                                           pcp_comm_cdc_neph,
                                           pcp_comm_omw,
                                           pcp_comm_spc,
                                           pcp_care_aba,
                                           pcp_care_art,
                                           pcp_care_bcs,
                                           pcp_care_col,
                                           pcp_care_cdc_a1c,
                                           pcp_care_cdc_eye,
                                           pcp_care_cdc_neph,
                                           pcp_care_omw,
                                           pcp_care_spc,
                                           ipa_comm_aba,
                                           ipa_comm_art,
                                           ipa_comm_bcs,
                                           ipa_comm_col,
                                           ipa_comm_cdc_a1c,
                                           ipa_comm_cdc_eye,
                                           ipa_comm_cdc_neph,
                                           ipa_comm_omw,
                                           ipa_comm_spc,
                                           ipa_care_aba,
                                           ipa_care_art,
                                           ipa_care_bcs,
                                           ipa_care_col,
                                           ipa_care_cdc_a1c,
                                           ipa_care_cdc_eye,
                                           ipa_care_cdc_neph,
                                           ipa_care_omw,
                                           ipa_care_spc)
         WITH msr_rslt
              AS (SELECT ds.mem_hap_nbr,
                         ds.hedis_population,
                         ds.msr_cd,
                         ds.sub_msr_desc,
                         ds.admin_numerator,
                         ds.admin_denominator
                    FROM dim_sum_ref ds
                   WHERE     ds.revw_set_id = var_fwd_rs_id
                         AND ds.revw_set_yr_mnth = var_fwd_rs_yr_mnth
                         AND ds.hedis_population IN ('Commercial HMO POS',
                                                     'Medicare Adv HMO POS')
                         AND ds.msr_cd IN ('ABA',
                                           'ART',
                                           'BCS',
                                           'COL',
                                           'CDC',
                                           'OMW',
                                           'SPC')
                         AND (   ds.sub_msr_desc IS NULL
                              OR ds.sub_msr_desc IN ('HbA1c Poor Control',
                                                     'Eye Exam',
                                                     'Medical Attention For Nephropathy',
                                                     'Received Statin Therapy'))),
              slct_msr
              AS (  SELECT rs.mem_hap_nbr,
                           rs.hedis_population,
                           rs.msr_cd,
                           rs.sub_msr_desc,
                           MAX (rs.admin_numerator) AS admin_numerator
                      FROM msr_rslt rs
                           INNER JOIN
                           (SELECT DISTINCT
                                   mem_hap_nbr,
                                   hedis_population,
                                   msr_cd,
                                   NVL (sub_msr_desc, 'NULL') AS sub_msr_desc
                              FROM dim_sum_ref
                             WHERE     revw_set_id =
                                             'YTD'
                                          || TO_CHAR (
                                                FLOOR (point_dt / 100) - 1)
                                   AND revw_set_yr_mnth =
                                          FLOOR (point_dt / 100) - 1
                                   AND hedis_population IN ('Commercial HMO POS',
                                                            'Medicare Adv HMO POS')
                                   AND msr_cd IN ('ABA',
                                                  'ART',
                                                  'BCS',
                                                  'COL',
                                                  'CDC',
                                                  'OMW',
                                                  'SPC')
                                   AND (   sub_msr_desc IS NULL
                                        OR sub_msr_desc IN ('HbA1c Poor Control',
                                                            'Eye Exam',
                                                            'Medical Attention For Nephropathy',
                                                            'Received Statin Therapy'))
                                   AND admin_numerator = 0) bgn
                              ON     rs.mem_hap_nbr = bgn.mem_hap_nbr
                                 AND rs.hedis_population = bgn.hedis_population
                                 AND rs.msr_cd = bgn.msr_cd
                                 AND NVL (rs.sub_msr_desc, 'NULL') =
                                        bgn.sub_msr_desc
                  GROUP BY rs.mem_hap_nbr,
                           rs.hedis_population,
                           rs.msr_cd,
                           rs.sub_msr_desc)
         SELECT ds.mem_hap_nbr,
                ds.hedis_population,
                CASE
                   WHEN ds.sub_msr_desc IS NULL THEN ds.msr_cd
                   ELSE ds.msr_cd || ' - ' || ds.sub_msr_desc
                END
                   AS msr_desc,
                CASE
                   WHEN ds.sub_msr_desc = 'HbA1c Poor Control'
                   THEN
                      ABS (ds.admin_numerator - 1)
                   ELSE
                      ds.admin_numerator
                END
                   AS admin_numerator,
                demo.mem_gndr_cd,
                demo.mem_age,
                demo.m_18,
                demo.m_25,
                demo.m_35,
                demo.m_45,
                demo.m_55,
                demo.m_65,
                demo.m_70,
                demo.m_75,
                demo.m_80,
                demo.m_85,
                demo.f_18,
                demo.f_25,
                demo.f_35,
                demo.f_45,
                demo.f_55,
                demo.f_65,
                demo.f_70,
                demo.f_75,
                demo.f_80,
                demo.f_85,
                demo.race_white,
                demo.race_black,
                demo.race_asian,
                demo.race_hispanic,
                demo.ind_mone,
                demo.hh_inc_avg,
                demo.mem_zipcd,
                demo.mem_mth,
                demo.sml_grp_ind,
                demo.veba_ind,
                demo.hfhs_ind,
                demo.fehb_ind,
                demo.hfmg_ind,
                demo.part_a_risk_adj_fctr,
                clm.ttl_liability,
                clm.ind_high_cost,
                clm.amt_ip,
                clm.amt_er,
                cmcc.copd,
                cmcc.depression,
                cmcc.chronic_heartfailure,
                cmcc.chronic_kidneydisease,
                cmcc.diabetes,
                cmcc.hypertension,
                pcp.pcp_mem_cnt,
                pcp.ipa_mem_cnt,
                bnft.indv_tier_1_ddctbl_amt,
                bnft.pcp_tier_1_copay_amt,
                bnft.er_tier_1_copay_amt,
                pcp_comm_aba,
                pcp_comm_art,
                pcp_comm_bcs,
                pcp_comm_col,
                pcp_comm_cdc_a1c,
                pcp_comm_cdc_eye,
                pcp_comm_cdc_neph,
                pcp_comm_omw,
                pcp_comm_spc,
                pcp_care_aba,
                pcp_care_art,
                pcp_care_bcs,
                pcp_care_col,
                pcp_care_cdc_a1c,
                pcp_care_cdc_eye,
                pcp_care_cdc_neph,
                pcp_care_omw,
                pcp_care_spc,
                ipa_comm_aba,
                ipa_comm_art,
                ipa_comm_bcs,
                ipa_comm_col,
                ipa_comm_cdc_a1c,
                ipa_comm_cdc_eye,
                ipa_comm_cdc_neph,
                ipa_comm_omw,
                ipa_comm_spc,
                ipa_care_aba,
                ipa_care_art,
                ipa_care_bcs,
                ipa_care_col,
                ipa_care_cdc_a1c,
                ipa_care_cdc_eye,
                ipa_care_cdc_neph,
                ipa_care_omw,
                ipa_care_spc
           FROM slct_msr ds
                INNER JOIN stat_mdl_gap_cls_demo demo
                   ON ds.mem_hap_nbr = demo.mem_hap_id
                INNER JOIN stat_mdl_gap_cls_pcp pcp
                   ON ds.mem_hap_nbr = pcp.mem_hap_id
                INNER JOIN stat_mdl_gap_cls_bnft bnft
                   ON ds.mem_hap_nbr = bnft.mem_hap_id
                LEFT OUTER JOIN stat_mdl_gap_cls_msrrslt rslt
                   ON ds.mem_hap_nbr = rslt.mem_hap_id
                LEFT OUTER JOIN stat_mdl_gap_cls_clm clm
                   ON ds.mem_hap_nbr = clm.mem_hap_id
                LEFT OUTER JOIN stat_mdl_gap_cls_clm_cc cmcc
                   ON ds.mem_hap_nbr = cmcc.mem_hap_id;

      COMMIT;
      write_to_log (obj_nm, v_tbl, batchid);
   END;

   PROCEDURE hit_all
   AS
   BEGIN
      hit_demo ();
      hit_clm ();
      hit_clm_cc ();
      hit_pcpipa ();
      hit_bnft ();
      hit_msrrslt ();
      --      hit_master_full ();
      hit_master_selected ();
   END;
END pkg_stat_mdl_gap_cls;