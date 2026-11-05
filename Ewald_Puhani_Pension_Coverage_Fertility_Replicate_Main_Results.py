#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replication Code for the main results of the paper:

Does pension coverage predict fertility? 
International evidence from OLS, fixed effects and machine learning random forests

by Anika Ewald and Patrick A. Puhani

Code was generated using OpenAI ChatGTP 5.6 Terra as distributed via Leibniz 
University Hannover artificial intelligence (LUHKI) 
and reviewed and edited by Anika Ewald 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR 
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r"SET_PATH")

# ---------------------------------------------------
# 0. Loading Data and Basic Setup
# ---------------------------------------------------

# Loading Data from Stata format
df = pd.read_stata("replicationdata.dta")

# Restricting Panel to relevant years (2000–2023)
df = df[(df["year"] > 1999) & (df["year"] < 2024)]

# central Variables
RANDOM_STATE = 4950
target = "tfr_c"
country_id = "id"
year_var = "year"

# Feature-Set 1: WBL overall
features1 = [
    "t_covpen_ilo", "i_blaw_wdi", "sexratio_c", "t_lifexp_c", "t_mort5_un",
    "t_meanys_c", "log_pop_wdi", "log_popdens_c", "pop_urb_wdi",
    "log_gdppc_2015_wdi", "pols_wdi", "netmig_wdi"
]

# Feature-Set 2: WBL Sub-Indicators
features2 = [
    "t_covpen_ilo", "i_blaw_assets_wdi", "i_blaw_entre_wdi", "i_blaw_marri_wdi",
    "i_blaw_mobil_wdi", "i_blaw_parent_wdi", "i_blaw_pay_wdi", "i_blaw_work_wdi",
    "sexratio_c", "t_lifexp_c", "t_mort5_un", "t_meanys_c", "log_pop_wdi",
    "log_popdens_c", "pop_urb_wdi", "log_gdppc_2015_wdi", "pols_wdi", "netmig_wdi"
]

# for RF/Regression only complete observations for both Feature-Sets
all_features = list(set(features1 + features2))
df_model = df.dropna(subset=[target] + all_features).copy()
years = np.sort(df_model[year_var].unique())

print(f"Total observations (complete cases): {df_model.shape[0]}")
print(f"Total countries: {df_model[country_id].nunique()}")
print(f"Years used: {years}")

# ---------------------------------------------------
# 1. Labeling Variables
# ---------------------------------------------------

var_labels = {
    "tfr_c": "Total fertility rate (births per woman)",
    "t_covpen_ilo": "Pension coverage (% of population above RA receiving a pension)",
    "i_blaw_wdi": "WBL overall score (0–100)",
    "i_blaw_assets_wdi": "WBL assets indicator score (0–100)",
    "i_blaw_entre_wdi": "WBL entrepreneurship indicator score (0–100)",
    "i_blaw_marri_wdi": "WBL marriage indicator score (0–100)",
    "i_blaw_mobil_wdi": "WBL mobility indicator score (0–100)",
    "i_blaw_parent_wdi": "WBL parenthood indicator score (0–100)",
    "i_blaw_pay_wdi": "WBL pay indicator score (0–100)",
    "i_blaw_work_wdi": "WBL workplace indicator score (0–100)",
    "sexratio_c": "Sex ratio at birth (males over females)",
    "t_lifexp_c": "Life expectancy at birth (years)",
    "t_mort5_un": "Under-five mortality rate (per 1,000 live births)",
    "t_meanys_c": "Mean years of schooling",
    "log_pop_wdi": "Log. of total population",
    "log_popdens_c": "Log. of population density",
    "pop_urb_wdi": "Urban population (% of total)",
    "log_gdppc_2015_wdi": "Log. of GDP per capita (constant 2015 USD)",
    "pols_wdi": "Political stability and absence of violence/terrorism (std. units)",
    "netmig_wdi": "Net migration (millions)"
}

# ---------------------------------------------------
# 2. Functions
# ---------------------------------------------------

def run_ols_fe(df, features):

    df_est = (
        df[[country_id, year_var, target] + features]
        .dropna()
        .copy()
    )

    print(
        f"\nModell mit {df_est[features].nunique()} Regressoren:, "
        f"N ={df_est.shape[0]}, "
        f"Länder ={df_est[country_id].nunique()}, "
        f"Jahre ={df_est[year_var].unique()}"
    )

    # Pooled OLS with Cluster-SE:
    X_ols = sm.add_constant(df_est[features].astype(float))
    y_ols = df_est[target].astype(float)

    ols_res = sm.OLS(y_ols, X_ols).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_est[country_id]},
        use_t=True
    )

    # Fixed Effects using PanelOLS:
    # Setting MultiIndex: (Country, Year)
    panel_df = (
        df_est
        .set_index([country_id, year_var])
        .sort_index()
    )

    fe_model = PanelOLS(
        dependent=panel_df[target].astype(float),
        exog=panel_df[features].astype(float),
        entity_effects=True,
        singletons=True
    )

    fe_res = fe_model.fit(
        cov_type="clustered",
        cluster_entity=True,
        debiased=True,
        auto_df=True
    )
    
    n_entities = panel_df.index.get_level_values(country_id).nunique()

    print(
        f"PanelOLS: N ={panel_df.shape[0]}, "
        f"Entity-FE ={n_entities}, "
        f"Within-R² ={fe_res.rsquared_within}"
    )

    return ols_res, fe_res

def extract_table(res, features, model_name):

    rows = []

    if hasattr(res, "std_errors"):
        se_values = res.std_errors
    else:
        se_values = res.bse

    for var in features:
        coef = res.params[var]
        se = se_values[var]
        p = res.pvalues[var]

        # Significance
        if p < 0.01:
            star = "***"
        elif p < 0.05:
            star = "**"
        elif p < 0.10:
            star = "*"
        else:
            star = ""

        label = var_labels[var]

        rows.append({
            "VARIABLES": label,
            model_name: f"{coef:.3f}{star} ({se:.3f})"
        })

    return pd.DataFrame(rows)


def rforest(df_model, features):

    # 1. Data for the respective Forest
    X = df_model[features].astype(float)
    y = df_model[target].astype(float)

    # 2. numvars(sqrt(p))
    n_features = len(features)
    numvars_default = max(1, int(np.floor(np.sqrt(n_features))))

    # 3. Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=400,                 # iterations(400)
        criterion="squared_error",        # Regression: Variance-/RSS-Reduction
        max_features=numvars_default,      
        max_depth=None,                   
        min_samples_leaf=1,               
        min_samples_split=2,              # sklearn-Default
        bootstrap=True,                   # Bootstrap-Sample each Tree
        oob_score=True,                   # saving OOB-Predictions
        random_state=RANDOM_STATE,        # seed(4950)
        n_jobs=-1                         # parallel computing
    )

    rf_model.fit(X, y)

    # 4. In-Sample Predictions
    p1 = rf_model.predict(X)

    prediction_df = pd.DataFrame({
        "p1": p1,
        target: y.to_numpy(),
    })

    print("\nFirst 20 Predictions:")
    print(prediction_df[["p1", target]].head(20))

    mae = np.mean(np.abs(y.to_numpy() - p1))
    rmse = np.sqrt(np.mean((y.to_numpy() - p1) ** 2))

    print(f"\nIn-Sample MAE: {mae}")
    print(f"In-Sample RMSE: {rmse}")

    # 5. OOB-RMSE
    oob_pred = rf_model.oob_prediction_
    valid_oob = ~np.isnan(oob_pred)

    oob_rmse = np.sqrt(
        np.mean((y.to_numpy()[valid_oob] - oob_pred[valid_oob]) ** 2)
    )

    print(f"OOB-RMSE: {oob_rmse}")
    print(f"Number of Features: {n_features}")
    print(f"numvars for each split: {numvars_default}")

    # 6. Impurity-based variable importance
    raw_importance = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": features,
        "raw_imp": raw_importance,
    })

    # Scaling importance values to their max 
    max_importance = importance_df["raw_imp"].max()

    if max_importance > 0:
        importance_df["norm_imp"] = (
            importance_df["raw_imp"] / max_importance
        )
    else:
        importance_df["norm_imp"] = 0.0

    print("\nVariable Importance (Maximum = 1):")
    print(
        importance_df
        .sort_values("norm_imp", ascending=False)
        .to_string(index=False)
    )


    importance_df.attrs["model"] = rf_model
    importance_df.attrs["predictions"] = prediction_df
    importance_df.attrs["oob_rmse"] = oob_rmse
    importance_df.attrs["rmse"] = rmse
    importance_df.attrs["mae"] = mae
    
    # 7. Permutation Importance
    permutation_result = permutation_importance(
        estimator=rf_model,
        X=X,
        y=y,
        scoring="neg_root_mean_squared_error",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importance_df["imp_mean_dRMSE"] = permutation_result.importances_mean
    importance_df["imp_sd_dRMSE"] = permutation_result.importances_std

    # Rescaling to max
    max_perm_importance = importance_df["imp_mean_dRMSE"].max()

    if max_perm_importance > 0:
        importance_df["perm_norm_imp"] = (
            importance_df["imp_mean_dRMSE"] / max_perm_importance
        )
    else:
        importance_df["perm_norm_imp"] = 0.0

    print("\nPermutation Importance: average worsening of the RMSE")
    print(
        importance_df[
            ["feature", "imp_mean_dRMSE", "imp_sd_dRMSE", "perm_norm_imp"]
        ]
        .sort_values("imp_mean_dRMSE", ascending=False)
        .to_string(index=False)
    )

    return importance_df
    

def plot_combined_importance(
    importance_df,
    var_labels,
    filename="CombinedImportance.pdf",
    color_impurity="#595959",
    color_permutation="#A6A6A6"):

    imp = importance_df.copy()

    imp["label"] = imp["feature"].map(var_labels)

    # Sorting by Permutation Importance.
    imp = imp.sort_values("perm_norm_imp", ascending=False).reset_index(drop=True)

    y_positions = np.arange(len(imp))

    bar_height = 0.34

    plt.figure(figsize=(7.2, 5.5))
    sns.set_style("ticks")
    plt.grid(axis="x", color="lightgray", linewidth=0.8)

    # Upper Bar: impurity-based importance
    plt.barh(
        y_positions + bar_height / 2,
        imp["norm_imp"],
        height=bar_height,
        color=color_impurity,
        edgecolor=color_impurity,
        linewidth=0.8,
        label="Impurity-based importance"
    )

    # Lower Bar: permutation importance
    plt.barh(
        y_positions - bar_height / 2,
        imp["perm_norm_imp"],
        height=bar_height,
        color=color_permutation,
        edgecolor=color_permutation,
        linewidth=0.8,
        label="Permutation importance"
    )

    ax = plt.gca()

    # Y-Axis: Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(imp["label"], fontsize=9)
    ax.tick_params(axis="y", length=0)

    # X-Axis
    min_value = min(
        0,
        imp["norm_imp"].min(),
        imp["perm_norm_imp"].min()
    )

    max_value = max(
        imp["norm_imp"].max(),
        imp["perm_norm_imp"].max()
    )

    ax.set_xlim(min_value - 0.00, max_value * 1.12)

    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"], fontsize=11)

    plt.xlabel("Variable importance (normalized)", fontsize=12, fontweight="roman")
    plt.ylabel("")

    # Legend
    plt.legend(
        loc="upper right",
        frameon=False,
        fontsize=10
    )

    plt.gca().set_facecolor("white")
    plt.gcf().patch.set_facecolor("white")

    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.show()
    
# ---------------------------------------------------
# 3. OLS & FE for both Feature-Sets
# ---------------------------------------------------

ols1, fe1 = run_ols_fe(df, features1)
ols2, fe2 = run_ols_fe(df, features2)


rmse_ols1 = np.sqrt(np.mean(ols1.resid ** 2))
rmse_fe1 = np.sqrt(np.mean(fe1.resids ** 2))

rmse_ols2 = np.sqrt(np.mean(ols2.resid ** 2))
rmse_fe2 = np.sqrt(np.mean(fe2.resids ** 2))

print("\n=== Full-sample OLS (Set 1) ===")
print(ols1.summary())
print("\n=== Full-sample FE (Set 1) ===")
print(fe1.summary)
print("\n=== Full-sample OLS (Set 2) ===")
print(ols2.summary())
print("\n=== Full-sample FE (Set 2) ===")
print(fe2.summary)


print("\n=== In-sample RMSE: OLS and FE models ===")
print(f"OLS Set 1 RMSE:{rmse_ols1}")
print(f"FE  Set 1 RMSE:{rmse_fe1}")
print(f"OLS Set 2 RMSE:{rmse_ols2}")
print(f"FE  Set 2 RMSE:{rmse_fe2}")

# ---------------------------------------------------
# 4. Table 1: OLS & FE
# ---------------------------------------------------

# Tables for every model
table_ols1 = extract_table(ols1, features1, "(1) OLS Set 1")
table_ols2 = extract_table(ols2, features2, "(2) OLS Set 2")
table_fe1  = extract_table(fe1,  features1, "(3) FE Set 1")
table_fe2  = extract_table(fe2,  features2, "(4) FE Set 2")

# Common "VARIABLES"-Column
results_table = table_ols1.merge(table_ols2, on="VARIABLES", how="outer")
results_table = results_table.merge(table_fe1, on="VARIABLES", how="outer")
results_table = results_table.merge(table_fe2, on="VARIABLES", how="outer")

results_table = results_table.fillna("")

# Additional model information
n_ols1 = int(ols1.nobs)
n_ols2 = int(ols2.nobs)
n_fe1 = int(fe1.nobs)
n_fe2 = int(fe2.nobs)

n_clust_ols1 = int(df[[country_id, target] + features1].dropna()[country_id].nunique())
n_clust_ols2 = int(df[[country_id, target] + features2].dropna()[country_id].nunique())

n_entities_fe1 = int(fe1.entity_info["total"])
n_entities_fe2 = int(fe2.entity_info["total"])

# Observations
results_table.loc[len(results_table)] = {
    "VARIABLES": "Observations",
    "(1) OLS Set 1": n_ols1,
    "(2) OLS Set 2": n_ols2,
    "(3) FE Set 1": n_fe1,
    "(4) FE Set 2": n_fe2
}
# R²
results_table.loc[len(results_table)] = {
    "VARIABLES": "R-squared",
    "(1) OLS Set 1": f"{ols1.rsquared:.3f}",
    "(2) OLS Set 2": f"{ols2.rsquared:.3f}",
    "(3) FE Set 1": f"{fe1.rsquared_within:.3f}",
    "(4) FE Set 2": f"{fe2.rsquared_within:.3f}"
}

# Number of Clusters
results_table.loc[len(results_table)] = {
    "VARIABLES": "N_clust",
    "(1) OLS Set 1": n_clust_ols1,
    "(2) OLS Set 2": n_clust_ols2,
    "(3) FE Set 1": n_entities_fe1,
    "(4) FE Set 2": n_entities_fe2
}

# Excel-Export
results_table.to_excel("Table_1_OLS_FE.xlsx", index=False)
print("\nExcel-Table saved: regression_results_4cols_2000.xlsx")
print(results_table)

# ---------------------------------------------------
# 5. Random Forest: Replication of Stata's rforest
# ---------------------------------------------------

# RF for features1 (WBL overall)
print("\n### Random Forest: Feature-Set 1 (WBL overall) ###")
importance_df1 = rforest(
    df_model=df_model,
    features=features1
)

# RF for features2 (WBL Sub-Indicators)
print("\n### Random Forest: Feature-Set 2 (WBL sub-indicators) ###")
importance_df2 = rforest(
    df_model=df_model,
    features=features2
)

# ---------------------------------------------------
# 6. Figure 1 & 2: Variable-Importance-Graphics
# ---------------------------------------------------

plot_combined_importance(
    importance_df1,
    var_labels,
    filename="Figure_1_Importance_WBL_overall.pdf"
)

plot_combined_importance(
    importance_df2,
    var_labels,
    filename="Figure_2_Importance_WBL_subcomponents.pdf"
)