import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import auc, roc_curve
from sklearn.calibration import calibration_curve


def optim_plot_prevalence(
    df, group_col, target_col="has_bc", title=None, palette="viridis"
):
    """Génère un graphique de prévalence automatisé pour n'importe quelle colonne.

    Parameters:
    -----------
    df : pandas.DataFrame
        Ton tableau de données.
    group_col : str
        La colonne de groupe à analyser (ex: 'dominant_origin', 'age_group',
        etc.).
    target_col : str, default='has_bc'
        La variable binaire (0/1) dont on cherche la prévalence.
    title : str, optional
        Titre personnalisé du graphique.
    palette : str, default='viridis'
        La palette de couleurs Seaborn à utiliser.
    """
    # 1. Calcul du taux de prévalence en % pour le groupe choisi
    df_prevalence = (
        (df.groupby(group_col)[target_col].mean() * 100)
        .reset_index()
        .sort_values(by=target_col, ascending=False)
    )

    # 2. Création du graphique
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(
        data=df_prevalence,
        x=group_col,
        y=target_col,
        hue=group_col,
        legend=False,
        palette=palette,
        edgecolor="black",
    )

    # Ajouter les pourcentages exacts au-dessus de chaque barre
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # Sécurité si une catégorie est vide
            ax.annotate(
                f"{height:.2f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
            )

    # Gestion dynamique des titres et labels
    clean_group_name = group_col.replace("_", " ").title()
    if title is None:
        title = f"Taux de prévalence de '{target_col}' selon '{clean_group_name}'"

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.xlabel(clean_group_name, fontsize=12)
    plt.ylabel(f"% de {target_col} touchés", fontsize=12)

    # Donne de l'espace pour les étiquettes en haut des barres
    plt.ylim(0, df_prevalence[target_col].max() * 1.15)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def bootstrap_auc(y_true_ser, y_scores_ser, n_bootstraps=1000, rng_seed=42):
    """Calcule l'IC à 95% de l'AUC par bootstrap de manière sécurisée."""
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)

    # Conversion stricte en tableaux Numpy 1D pour éviter les erreurs d'indexation
    y_true_arr = (
        y_true_ser.values.ravel()
        if hasattr(y_true_ser, "values")
        else np.asarray(y_true_ser).ravel()
    )
    y_scores_arr = (
        y_scores_ser.values.ravel()
        if hasattr(y_scores_ser, "values")
        else np.asarray(y_scores_ser).ravel()
    )

    n_samples = len(y_scores_arr)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true_arr[indices])) < 2:
            continue
        score = auc(
            *roc_curve(y_true_arr[indices], y_scores_arr[indices])[:2]
        )
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    return sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[
        int(0.975 * len(sorted_scores))
    ]


def recalibrate_prs_only(
    df_data,
    train_ids,
    test_ids,
    prs_col="prs_298",
    target_col="has_bc",
    id_col="person_id",
):
    df_out = df_data.copy()

    ids_train_flat = (
        train_ids.values.ravel()
        if hasattr(train_ids, "values")
        else np.asarray(train_ids).ravel()
    )
    ids_test_flat = (
        test_ids.values.ravel()
        if hasattr(test_ids, "values")
        else np.asarray(test_ids).ravel()
    )

    is_train = df_out[id_col].isin(ids_train_flat)
    is_test = df_out[id_col].isin(ids_test_flat)

    # 1. Normalisation Z-score sur Témoins du Train
    train_controls_mask = is_train & (df_out[target_col] == 0)
    mu_prs = df_out.loc[train_controls_mask, prs_col].mean()
    sigma_prs = df_out.loc[train_controls_mask, prs_col].std()

    z_col = f"{prs_col}_z"
    df_out[z_col] = (df_out[prs_col] - mu_prs) / sigma_prs

    # 2. Apprentissage de alpha et beta sur le Train (PRS SEUL)
    df_train = df_out[is_train].dropna(subset=[target_col, z_col])
    X_train = sm.add_constant(df_train[z_col])
    y_train = df_train[target_col]

    model = sm.Logit(y_train, X_train).fit(disp=False)
    alpha = model.params.loc["const"]
    beta_prs = model.params.loc[z_col]

    # 3. Prédiction des probabilités sur le Test
    df_test = df_out[is_test].dropna(subset=[target_col, z_col]).copy()
    logit_test = alpha + beta_prs * df_test[z_col]

    recalib_col = f"{prs_col}_prob_recalib_only"
    df_test[recalib_col] = 1.0 / (1.0 + np.exp(-logit_test))

    df_out.loc[df_test.index, recalib_col] = df_test[recalib_col]

    print(f"=== RECALIBRATION SEULE : {prs_col.upper()} ===")
    print(f"• Intercept (α) : {alpha:.4f}")
    print(f"• Pente PRS (β) : {beta_prs:.4f}\n")

    return df_out, {
        "prs_col": prs_col,
        "recalib_col": recalib_col,
        "alpha": alpha,
        "beta": beta_prs,
    }


def compare_prs_calibration(
    df_data,
    train_ids,
    test_ids,
    prs_col="prs_298",
    recalib_col="prs_298_prob_recalib_only",
    target_col="has_bc",
    id_col="person_id",
    n_bins=10,
    model_name="PRS-298",
    figsize=(6, 6),
):
    """Compare la calibration du PRS Brut et du PRS Recalibré (univarié)

    sur la même échelle réelle d'incidence.
    """
    ids_train_flat = (
        train_ids.values.ravel()
        if hasattr(train_ids, "values")
        else np.asarray(train_ids).ravel()
    )
    ids_test_flat = (
        test_ids.values.ravel()
        if hasattr(test_ids, "values")
        else np.asarray(test_ids).ravel()
    )

    is_train = df_data[id_col].isin(ids_train_flat)
    is_test = df_data[id_col].isin(ids_test_flat)

    # Standardisation sur Témoins du Train
    train_controls_mask = is_train & (df_data[target_col] == 0)
    mu_prs = df_data.loc[train_controls_mask, prs_col].mean()
    sigma_prs = df_data.loc[train_controls_mask, prs_col].std()

    df_test = df_data[is_test].dropna(subset=[target_col, prs_col, recalib_col]).copy()
    z_test = (df_test[prs_col] - mu_prs) / sigma_prs
    y_test = df_test[target_col]

    # Probabilités
    base_incidence = y_test.mean()
    prob_brut_scaled = base_incidence * np.exp(z_test)
    prob_recalib = df_test[recalib_col]

    # Points des courbes de calibration
    prob_true_brut, prob_pred_brut = calibration_curve(
        y_test, prob_brut_scaled, n_bins=n_bins, strategy="quantile"
    )
    prob_true_rec, prob_pred_rec = calibration_curve(
        y_test, prob_recalib, n_bins=n_bins, strategy="quantile"
    )

    max_val = max(
        prob_true_rec.max(),
        prob_pred_rec.max(),
        prob_pred_brut.max(),
        prob_true_brut.max(),
    ) * 1.15

    # Tracé
    plt.figure(figsize=figsize)

    plt.plot(
        [0, max_val],
        [0, max_val],
        linestyle="--",
        color="gray",
        linewidth=2,
        label="Calibration parfaite (y = x)",
    )

    plt.plot(
        prob_pred_brut,
        prob_true_brut,
        marker="s",
        linewidth=2,
        color="crimson",
        label=f"{model_name} Brut (Score réaligné)",
    )

    plt.plot(
        prob_pred_rec,
        prob_true_rec,
        marker="o",
        linewidth=2,
        color="dodgerblue",
        label=f"{model_name} Recalibré (α + β·PRS)",
    )

    plt.xlim([0, max_val])
    plt.ylim([0, max_val])

    plt.xlabel("Risque prédit moyen", fontsize=11, fontweight="bold")
    plt.ylabel("Proportion observée de cas", fontsize=11, fontweight="bold")
    plt.title(
        f"Comparaison de Calibration - {model_name}\n(Jeu de Test N = {len(df_test)})",
        fontsize=12,
        fontweight="bold",
    )

    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


def evaluate_and_plot_model(
    df_data,
    test_ids,
    score_col,
    target_col="has_bc",
    alpha_val=0.0,
    id_col="person_id",
    model_name="PRS-298",
    quantiles=None,
    labels=None,
    ref_quantile_idx=2,
    figsize=(16, 6),
):
    """Évalue et synthétise la discrimination, la calibration globale (O/E)

    et la stratification par quantile en une figure unique.
    """
    if quantiles is None:
        quantiles = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
    if labels is None:
        labels = [
            "Q1 (<20%)",
            "Q2 (20-40%)",
            "Q3 (40-60%)",
            "Q4 (60-80%)",
            "Q5 (>80%)",
        ]

    ids_to_match = (
        test_ids.values.ravel()
        if hasattr(test_ids, "values")
        else np.asarray(test_ids).ravel()
    )
    df_test = df_data[df_data[id_col].isin(ids_to_match)].copy()
    df_eval = (
        df_test.dropna(subset=[target_col, score_col])
        .reset_index(drop=True)
        .copy()
    )

    y_true = df_eval[target_col]
    p_recalib = df_eval[score_col]

    # 1. Métriques ROC et AUC
    fpr, tpr, _ = roc_curve(y_true, p_recalib)
    roc_auc = auc(fpr, tpr)
    ci_lower, ci_upper = bootstrap_auc(y_true, p_recalib)

    # 2. Métriques O/E (Calibration globale)
    observed_cases = int(y_true.sum())
    expected_recalib = float(p_recalib.sum())
    oe_recalib = (
        observed_cases / expected_recalib if expected_recalib > 0 else np.nan
    )

    # 3. Découpage en Quantiles
    name_column = f"{model_name}_risk_group"
    df_eval[name_column] = pd.qcut(
        df_eval[score_col], q=quantiles, labels=labels, duplicates="drop"
    )

    incidence = df_eval.groupby(name_column, observed=False)[target_col].agg(
        ["mean", "count"]
    )
    incidence["pct"] = incidence["mean"] * 100
    incidence = incidence.reindex(labels)

    pred_risk = df_eval.groupby(name_column, observed=False)[score_col].mean()
    pred_risk_pct = (pred_risk * 100).reindex(labels)

    dummies = pd.get_dummies(df_eval[name_column], drop_first=False)
    ref_col = labels[ref_quantile_idx]
    if ref_col not in dummies.columns:
        ref_col = dummies.columns[len(dummies.columns) // 2]

    X = dummies.drop(columns=[ref_col]).astype(int)
    X = sm.add_constant(X)

    logit_mod = sm.Logit(y_true, X).fit(disp=False)
    conf = logit_mod.conf_int()

    or_df = pd.DataFrame(
        {
            "OR": np.exp(logit_mod.params),
            "lower": np.exp(conf[0]),
            "upper": np.exp(conf[1]),
        }
    )
    or_df.loc["const"] = [1.0, 1.0, 1.0]
    or_df.index = [c if c != "const" else ref_col for c in or_df.index]
    or_df = or_df.reindex(labels)

    # 4. Tracé de la Figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Subplot 1 : ROC & Cartouche O/E ---
    axes[0].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"{model_name} (AUC = {roc_auc:.3f} [{ci_lower:.3f}-{ci_upper:.3f}])",
    )
    axes[0].plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=1.5,
        linestyle="--",
        label="Hasard (AUC = 0.50)",
    )
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("1 - Spécificité (Taux Faux Positifs)", fontsize=10)
    axes[0].set_ylabel("Sensibilité (Taux Vrais Positifs)", fontsize=10)
    axes[0].set_title(
        f"Discrimination (ROC) - {model_name} (Test N={len(df_eval)})",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    text_str = (
        r"$\mathbf{Calibration\ globale\ (Test\ 75\%):}$"
        "\n"
        f"• Cas observés (O) : {observed_cases}"
        "\n"
        f"• Cas attendus (E) : {expected_recalib:.1f}"
        "\n"
        rf"• $\mathbf{{Ratio\ O/E\ :}}$ {oe_recalib:.3f}"
        "\n\n"
        rf"$\mathbf{{Paramètre\ appris\ :}}$"
        "\n"
        rf"• Intercept ($\alpha$) : {alpha_val:.4f}"
    )

    axes[0].text(
        0.04,
        0.96,
        text_str,
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

    # --- Subplot 2 : Barres d'Incidence + Risque prédit + OR ---
    x_pos = np.arange(len(labels))

    axes[1].bar(
        x_pos - 0.15,
        incidence["pct"],
        width=0.3,
        alpha=0.6,
        color="skyblue",
        edgecolor="navy",
        label="Incidence réelle (%)",
    )

    axes[1].bar(
        x_pos + 0.15,
        pred_risk_pct,
        width=0.3,
        alpha=0.6,
        color="orange",
        edgecolor="darkorange",
        label="Risque prédit réétalonné (%)",
    )

    axes[1].set_xlabel("Quantiles de risque génétique", fontsize=10)
    axes[1].set_ylabel(
        "Pourcentage (%)", fontsize=10, color="navy", fontweight="bold"
    )
    axes[1].tick_params(axis="y", labelcolor="navy")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, rotation=30)
    axes[1].grid(True, linestyle=":", alpha=0.4)

    ax2 = axes[1].twinx()
    valid_or = or_df.dropna(subset=["OR"])
    valid_x = [labels.index(idx) for idx in valid_or.index if idx in labels]

    line_or = ax2.errorbar(
        valid_x,
        valid_or["OR"],
        yerr=[
            valid_or["OR"] - valid_or["lower"],
            valid_or["upper"] - valid_or["OR"],
        ],
        fmt="o-",
        color="darkred",
        ecolor="crimson",
        elinewidth=1.8,
        capsize=4,
        markersize=6,
        label="Odds Ratio (IC 95%)",
    )

    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_ylabel(
        f"Odds Ratio (OR) vs Quantile Médian ({ref_col})",
        fontsize=10,
        color="darkred",
        fontweight="bold",
    )
    ax2.tick_params(axis="y", labelcolor="darkred")

    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    axes[1].legend(
        lines_1 + [line_or],
        labels_1 + ["Odds Ratio (IC 95%)"],
        loc="upper left",
        frameon=True,
        fontsize=8.5,
    )

    axes[1].set_title(
        f"Stratification du risque : Incidence vs Prédit ({model_name})",
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()

    metrics = {
        "AUC": roc_auc,
        "AUC_CI": (ci_lower, ci_upper),
        "OE_recalib": oe_recalib,
        "alpha": alpha_val,
        "N_test": len(df_eval),
    }

    return metrics, df_eval


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve


def evaluate_and_compare_all_models(
    df_data,
    test_ids,
    models_dict=None,
    target_col="has_bc",
    id_col="person_id",
    n_bins=10,
    figsize=(16, 7),
):
    """Évalue et compare simultanément BCSC, Mammorisk, PRS-311 et leurs combinaisons

    sur la discrimination (ROC) et la calibration globale (O/E & calibration plot).
    """
    if models_dict is None:
        models_dict = {
            "BCSC (Recalibré)": "score_bcscRisk_recalib",
            "Mammorisk (Recalibré)": "score_mammorisk_recalib",
            "PRS-311 (Seul)": "prs_311_prob_recalib_only",
            "BCSC + PRS-311": "prob_bcsc_prs311",
            "Mammorisk + PRS-311": "prob_mammorisk_prs311",
        }

    # 1. Extraction du jeu de TEST
    ids_test_flat = (
        test_ids.values.ravel()
        if hasattr(test_ids, "values")
        else np.asarray(test_ids).ravel()
    )
    df_test = df_data[df_data[id_col].isin(ids_test_flat)].copy()

    # Couleurs distinctes pour les 5 modèles
    colors = {
        "BCSC (Recalibré)": "steelblue",
        "Mammorisk (Recalibré)": "darkorange",
        "PRS-311 (Seul)": "mediumpurple",
        "BCSC + PRS-311": "navy",
        "Mammorisk + PRS-311": "firebrick",
    }

    results = []
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Axe de calibration parfaite y = x
    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Diagonale (y = x)",
    )

    max_calib_val = 0.0

    # 2. Boucle d'évaluation sur chaque modèle
    for name, col in models_dict.items():
        if col not in df_test.columns:
            print(f"⚠️ Colonne '{col}' manquante dans df_data, ignorée.")
            continue

        df_eval = df_test.dropna(subset=[target_col, col]).copy()
        y_true = df_eval[target_col]
        p_pred = df_eval[col]

        # ROC / AUC
        fpr, tpr, _ = roc_curve(y_true, p_pred)
        roc_auc = auc(fpr, tpr)
        ci_lower, ci_upper = bootstrap_auc(y_true, p_pred)

        # Ratio O/E
        obs = int(y_true.sum())
        exp = float(p_pred.sum())
        oe_ratio = obs / exp if exp > 0 else np.nan

        results.append(
            {
                "Modèle": name,
                "AUC": round(roc_auc, 3),
                "IC 95% AUC": f"[{ci_lower:.3f} - {ci_upper:.3f}]",
                "Cas Observés (O)": obs,
                "Cas Attendus (E)": round(exp, 1),
                "Ratio O/E": round(oe_ratio, 3),
            }
        )

        color = colors.get(name, "black")

        # Subplot 1 : Courbes ROC
        axes[0].plot(
            fpr,
            tpr,
            lw=2,
            color=color,
            label=f"{name} (AUC = {roc_auc:.3f})",
        )

        # Subplot 2 : Calibration
        prob_true, prob_pred = calibration_curve(
            y_true, p_pred, n_bins=n_bins, strategy="quantile"
        )
        axes[1].plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            color=color,
            label=f"{name} (O/E = {oe_ratio:.3f})",
        )

        max_calib_val = max(
            max_calib_val, prob_true.max(), prob_pred.max()
        )

    # Style Subplot 1 : ROC
    axes[0].plot(
        [0, 1],
        [0, 1],
        color="gray",
        lw=1.5,
        linestyle="--",
        label="Hasard (AUC = 0.50)",
    )
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("1 - Spécificité", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Sensibilité", fontsize=10, fontweight="bold")
    axes[0].set_title(
        f"Comparaison de Discrimination (ROC) - N={len(df_test)}",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Style Subplot 2 : Calibration
    axes[1].set_xlim([0, max_calib_val * 1.15])
    axes[1].set_ylim([0, max_calib_val * 1.15])
    axes[1].set_xlabel("Risque prédit moyen", fontsize=10, fontweight="bold")
    axes[1].set_ylabel(
        "Proportion observée de cas", fontsize=10, fontweight="bold"
    )
    axes[1].set_title(
        "Comparaison de Calibration (Déciles)", fontsize=12, fontweight="bold"
    )
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()

    # 3. Tableau synthétique des métriques
    df_metrics = pd.DataFrame(results)
    return df_metrics


import numpy as np
import pandas as pd
import statsmodels.api as sm


def recalibrate_combined_models(
    df_data,
    train_ids,
    test_ids,
    combined_cols=None,
    target_col="has_bc",
    id_col="person_id",
):
    """Ajuste l'intercept (alpha_c) des modèles combinés sur le jeu de Train (25%)

    et applique la correction sur le jeu de Test (75%) pour ramener le ratio O/E à 1.00.
    """
    if combined_cols is None:
        combined_cols = ["prob_bcsc_prs311", "prob_mammorisk_prs311"]

    df_out = df_data.copy()

    ids_train_flat = (
        train_ids.values.ravel()
        if hasattr(train_ids, "values")
        else np.asarray(train_ids).ravel()
    )
    ids_test_flat = (
        test_ids.values.ravel()
        if hasattr(test_ids, "values")
        else np.asarray(test_ids).ravel()
    )

    is_train = df_out[id_col].isin(ids_train_flat)
    is_test = df_out[id_col].isin(ids_test_flat)

    eps = 1e-7

    print("=== RECALIBRATION DE L'INTERCEPT DES MODÈLES COMBINÉS ===")

    for col in combined_cols:
        if col not in df_out.columns:
            print(f"⚠️ Colonne '{col}' introuvable dans df_data.")
            continue

        # 1. Calcul du logit sur le TRAIN
        df_train = df_out[is_train].dropna(subset=[target_col, col]).copy()
        p_train = np.clip(df_train[col], eps, 1 - eps)
        logit_train = np.log(p_train / (1 - p_train))

        # 2. Estimation de alpha_c (offset = logit_train, const seule apprise)
        X_train = np.ones((len(df_train), 1))
        y_train = df_train[target_col]

        model = sm.Logit(y_train, X_train, offset=logit_train).fit(disp=False)
        alpha_c = model.params[0]

        # 3. Application du décalage sur le TEST
        df_test = df_out[is_test].dropna(subset=[target_col, col]).copy()
        p_test = np.clip(df_test[col], eps, 1 - eps)
        logit_test = np.log(p_test / (1 - p_test))

        new_col = f"{col}_recalib"
        df_out.loc[df_test.index, new_col] = 1.0 / (
            1.0 + np.exp(-(logit_test + alpha_c))
        )

        print(f"• Modèle '{col}' : Intercept d'ajustement (α_c) = {alpha_c:.4f}")
        print(f"  ➜ Nouvelle colonne générée : '{new_col}'")

    print()
    return df_out