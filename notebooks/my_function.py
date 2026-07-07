import matplotlib.pyplot as plt
import seaborn as sns


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