"""
NLMK — Dashboard Analytique Achats
Framework : Flask + Plotly + scikit-learn
"""

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "nlmk_secret_2026"

USERS = {
    "admin":  {"password": "nlmk2026",  "role": "Admin",  "name": "Administrateur"},
    "client": {"password": "client123", "role": "Client", "name": "NLMK Client"},
}

def load_data():
    df = pd.read_excel("Data_LAF_TCC.xlsx", sheet_name="Feuil1", parse_dates=["Date comptable"])
    df = df[df["Val./Devise objet"] > 0]
    df["Année"]  = df["Date comptable"].dt.year
    df["Mois"]   = df["Date comptable"].dt.month
    df["Mois_P"] = df["Date comptable"].dt.to_period("M").astype(str)
    df["Site"]   = df["Désignation de l'objet"].apply(
        lambda x: "LAF" if "LAF" in str(x) else ("TCC" if "TCC" in str(x) else "Commun"))
    return df

DF = load_data()

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def make_chart_layout(title, dark=False):
    bg = "#1E2A3A" if dark else "white"
    paper = "#1E2A3A" if dark else "white"
    font_color = "#E0E8F0" if dark else "#1a1814"
    grid = "#2A3A4A" if dark else "#F0F0F0"
    return dict(plot_bgcolor=bg, paper_bgcolor=paper,
                font=dict(family="DM Sans", color=font_color),
                title_font_size=14, title_text=title,
                margin=dict(t=50, b=40, l=40, r=20))

@app.route("/", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        u = USERS.get(request.form.get("username",""))
        if u and u["password"] == request.form.get("password",""):
            session.update({"user": request.form["username"], "name": u["name"], "role": u["role"]})
            return redirect(url_for("dashboard"))
        error = "Identifiant ou mot de passe incorrect."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    df    = DF.copy()
    year  = request.args.get("year", "Tout")
    site  = request.args.get("site", "Tout")
    search= request.args.get("search","").strip()
    dark  = request.args.get("dark","0")

    if year  != "Tout": df = df[df["Année"] == int(year)]
    if site  != "Tout": df = df[df["Site"]  == site]
    if search:
        mask = (df["Name1"].str.contains(search,case=False,na=False) |
                df["Désignation de l'objet"].str.contains(search,case=False,na=False))
        df = df[mask]

    is_dark = dark == "1"
    bg      = "#1E2A3A" if is_dark else "white"
    grid_c  = "#2A3A4A" if is_dark else "#F0F0F0"

    total       = df["Val./Devise objet"].sum()
    nb_trans    = len(df)
    nb_fourniss = df["Name1"].nunique()
    moy_trans   = df["Val./Devise objet"].mean()

    # Fig1 : mensuelle
    monthly = df.groupby("Mois_P")["Val./Devise objet"].sum().reset_index().sort_values("Mois_P")
    fig1 = px.line(monthly, x="Mois_P", y="Val./Devise objet",
                   labels={"Mois_P":"Mois","Val./Devise objet":"€"},
                   color_discrete_sequence=["#4A9EE0"])
    fig1.update_traces(mode="lines+markers", line_width=2, marker_size=5)
    fig1.update_layout(**make_chart_layout("Évolution mensuelle des dépenses", is_dark))
    fig1.update_xaxes(tickangle=-45, showgrid=False)
    fig1.update_yaxes(showgrid=True, gridcolor=grid_c)

    # Fig2 : top fournisseurs
    top_f = df.groupby("Name1")["Val./Devise objet"].sum().sort_values(ascending=True).tail(10).reset_index()
    fig2 = px.bar(top_f, x="Val./Devise objet", y="Name1", orientation="h",
                  labels={"Val./Devise objet":"€","Name1":""},
                  color_discrete_sequence=["#1B3A6B"])
    fig2.update_layout(**make_chart_layout("Top 10 Fournisseurs", is_dark))
    fig2.update_layout(margin=dict(t=50,l=220,b=20,r=20))
    fig2.update_xaxes(showgrid=True, gridcolor=grid_c)
    fig2.update_yaxes(showgrid=False)

    # Fig3 : centres de coûts
    by_cc = df.groupby("Désignation de l'objet")["Val./Devise objet"].sum().sort_values(ascending=False).head(10).reset_index()
    fig3 = px.bar(by_cc, x="Désignation de l'objet", y="Val./Devise objet",
                  labels={"Val./Devise objet":"€","Désignation de l'objet":""},
                  color_discrete_sequence=["#C8563A"])
    fig3.update_layout(**make_chart_layout("Top 10 Centres de Coûts", is_dark))
    fig3.update_xaxes(tickangle=-30)
    fig3.update_yaxes(showgrid=True, gridcolor=grid_c)

    # Fig4 : LAF vs TCC
    by_site = df.groupby(["Année","Site"])["Val./Devise objet"].sum().reset_index()
    fig4 = px.bar(by_site, x="Année", y="Val./Devise objet", color="Site", barmode="group",
                  labels={"Val./Devise objet":"€","Année":""},
                  color_discrete_map={"LAF":"#1B3A6B","TCC":"#C8563A","Commun":"#8A9BBE"})
    fig4.update_layout(**make_chart_layout("LAF vs TCC vs Commun — par année", is_dark))
    fig4.update_yaxes(showgrid=True, gridcolor=grid_c)

    # Fig5 : donut
    pie_data = df.groupby("Site")["Val./Devise objet"].sum().reset_index()
    fig5 = px.pie(pie_data, values="Val./Devise objet", names="Site",
                  color_discrete_map={"LAF":"#1B3A6B","TCC":"#C8563A","Commun":"#8A9BBE"}, hole=0.45)
    fig5.update_layout(**make_chart_layout("Répartition LAF / TCC / Commun", is_dark))

    # Top transactions
    top_trans = (df.nlargest(10,"Val./Devise objet")
                   [["Date comptable","Désignation de l'objet","Name1",
                     "Texte de la commande d'achat","Val./Devise objet"]].copy())
    top_trans["Date comptable"]    = top_trans["Date comptable"].dt.strftime("%d/%m/%Y")
    top_trans["Val./Devise objet"] = top_trans["Val./Devise objet"].apply(lambda x: f"{x:,.0f} €")
    top_trans = top_trans.rename(columns={
        "Date comptable":"Date","Désignation de l'objet":"Centre",
        "Name1":"Fournisseur","Texte de la commande d'achat":"Description","Val./Devise objet":"Montant"})

    # Conclusion analytique
    by_year = DF.groupby("Année")["Val./Devise objet"].sum()
    top_cc  = DF.groupby("Désignation de l'objet")["Val./Devise objet"].sum().idxmax()
    top_sup = DF.groupby("Name1")["Val./Devise objet"].sum().idxmax()
    best_y  = int(by_year.idxmax())
    low_y   = int(by_year.idxmin())
    pct_tcc = DF[DF["Site"]=="TCC"]["Val./Devise objet"].sum() / DF["Val./Devise objet"].sum() * 100
    conclusion = {
        "total":    f"{DF['Val./Devise objet'].sum():,.0f} €",
        "best_y":   best_y,
        "best_y_v": f"{by_year[best_y]:,.0f} €",
        "low_y":    low_y,
        "low_y_v":  f"{by_year[low_y]:,.0f} €",
        "top_cc":   top_cc,
        "top_sup":  top_sup,
        "pct_tcc":  f"{pct_tcc:.1f}%",
        "nb_sup":   DF["Name1"].nunique(),
    }

    years = sorted(DF["Année"].dropna().unique().tolist())
    return render_template("dashboard.html",
        fig1=fig1.to_html(full_html=False,include_plotlyjs=False),
        fig2=fig2.to_html(full_html=False,include_plotlyjs=False),
        fig3=fig3.to_html(full_html=False,include_plotlyjs=False),
        fig4=fig4.to_html(full_html=False,include_plotlyjs=False),
        fig5=fig5.to_html(full_html=False,include_plotlyjs=False),
        total=f"{total:,.0f} €", nb_trans=f"{nb_trans:,}",
        nb_fourniss=f"{nb_fourniss:,}", moy_trans=f"{moy_trans:,.0f} €",
        top_trans=top_trans.to_dict("records"),
        conclusion=conclusion, years=years,
        selected_year=year, selected_site=site, search=search,
        dark=dark, user_name=session.get("name"), user_role=session.get("role"))

@app.route("/prediction")
@login_required
def prediction():
    dark   = request.args.get("dark","0")
    is_dark= dark == "1"
    bg     = "#1E2A3A" if is_dark else "white"
    grid_c = "#2A3A4A" if is_dark else "#F0F0F0"

    df = DF.copy()
    monthly = df.groupby("Mois_P")["Val./Devise objet"].sum().reset_index().sort_values("Mois_P")
    monthly["t"]    = range(len(monthly))
    monthly["mois"] = monthly["Mois_P"].apply(lambda x: int(x.split("-")[1]))

    def feats(t, m):
        return np.column_stack([t, np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)])

    X = feats(monthly["t"].values, monthly["mois"].values)
    y = monthly["Val./Devise objet"].values
    model = LinearRegression().fit(X, y)

    # Prévision 2026
    fut_2026 = pd.period_range("2026-01", periods=12, freq="M")
    t_2026   = np.arange(48, 60)
    m_2026   = np.array([m.month for m in fut_2026])
    p_2026   = np.maximum(model.predict(feats(t_2026, m_2026)), 0)

    # Prévision 2027
    fut_2027 = pd.period_range("2027-01", periods=12, freq="M")
    t_2027   = np.arange(60, 72)
    m_2027   = np.array([m.month for m in fut_2027])
    p_2027   = np.maximum(model.predict(feats(t_2027, m_2027)), 0)

    noms = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

    # Fig A : Historique + prévisions
    figA = go.Figure()
    figA.add_trace(go.Scatter(x=monthly["Mois_P"], y=monthly["Val./Devise objet"],
        mode="lines+markers", name="Réel (2022–2025)",
        line=dict(color="#4A9EE0", width=2), marker=dict(size=4)))
    figA.add_trace(go.Scatter(
        x=[str(m) for m in fut_2026], y=p_2026,
        mode="lines+markers", name="Prévu 2026",
        line=dict(color="#E8A020", width=2, dash="dash"), marker=dict(size=5, symbol="square")))
    figA.add_trace(go.Scatter(
        x=[str(m) for m in fut_2027], y=p_2027,
        mode="lines+markers", name="Prévu 2027",
        line=dict(color="#C8563A", width=2, dash="dot"), marker=dict(size=5, symbol="diamond")))
    figA.add_vrect(x0="2026-01", x1="2026-12", fillcolor="rgba(232,160,32,0.07)", line_width=0, annotation_text="2026", annotation_position="top left")
    figA.add_vrect(x0="2027-01", x1="2027-12", fillcolor="rgba(200,86,58,0.07)", line_width=0, annotation_text="2027", annotation_position="top left")
    figA.update_layout(**make_chart_layout("Historique & Prévisions 2026–2027", is_dark))
    figA.update_xaxes(tickangle=-45, showgrid=False)
    figA.update_yaxes(showgrid=True, gridcolor=grid_c)

    # Fig B : Comparaison barres 2025 vs 2026 vs 2027
    vals_2025 = df[df["Année"]==2025].groupby("Mois")["Val./Devise objet"].sum().reindex(range(1,13), fill_value=0).values
    figB = go.Figure()
    figB.add_trace(go.Bar(x=noms, y=vals_2025, name="2025 réel", marker_color="#1B3A6B", opacity=0.85))
    figB.add_trace(go.Bar(x=noms, y=p_2026,    name="2026 prévu", marker_color="#E8A020", opacity=0.85))
    figB.add_trace(go.Bar(x=noms, y=p_2027,    name="2027 prévu", marker_color="#C8563A", opacity=0.85))
    figB.update_layout(**make_chart_layout("Comparaison mensuelle 2025 / 2026 / 2027", is_dark), barmode="group")
    figB.update_yaxes(showgrid=True, gridcolor=grid_c)

    # Fig C : LAF vs TCC prévisions 2026
    def prevoir_site(site, t_f, m_f):
        d = df[df["Site"]==site]
        m = d.groupby("Mois_P")["Val./Devise objet"].sum().reset_index().sort_values("Mois_P")
        m["t"]  = range(len(m))
        m["mn"] = m["Mois_P"].apply(lambda x: int(x.split("-")[1]))
        mod = LinearRegression().fit(feats(m["t"].values, m["mn"].values), m["Val./Devise objet"].values)
        return np.maximum(mod.predict(feats(t_f, m_f)), 0)

    laf_26 = prevoir_site("LAF", t_2026, m_2026)
    tcc_26 = prevoir_site("TCC", t_2026, m_2026)
    laf_27 = prevoir_site("LAF", t_2027, m_2027)
    tcc_27 = prevoir_site("TCC", t_2027, m_2027)

    figC = go.Figure()
    x2 = np.arange(12)
    figC.add_trace(go.Bar(x=noms, y=laf_26/1e6, name="LAF 2026", marker_color="#1B3A6B", opacity=0.85))
    figC.add_trace(go.Bar(x=noms, y=tcc_26/1e6, name="TCC 2026", marker_color="#E8A020", opacity=0.85))
    figC.add_trace(go.Bar(x=noms, y=laf_27/1e6, name="LAF 2027", marker_color="#2E6BB0", opacity=0.6))
    figC.add_trace(go.Bar(x=noms, y=tcc_27/1e6, name="TCC 2027", marker_color="#C8563A", opacity=0.6))
    figC.update_layout(**make_chart_layout("Prévisions LAF vs TCC — 2026 & 2027 (M€)", is_dark), barmode="group")
    figC.update_yaxes(showgrid=True, gridcolor=grid_c)

    # Fig D : Totaux annuels prévus
    by_year_hist = df.groupby("Année")["Val./Devise objet"].sum()
    figD = go.Figure()
    figD.add_trace(go.Bar(x=list(by_year_hist.index), y=list(by_year_hist.values),
        name="Réel", marker_color="#1B3A6B", opacity=0.85))
    figD.add_trace(go.Bar(x=[2026, 2027], y=[p_2026.sum(), p_2027.sum()],
        name="Prévu", marker_color="#E8A020", opacity=0.85))
    figD.update_layout(**make_chart_layout("Dépenses annuelles totales — Réel & Prévu", is_dark), barmode="group")
    figD.update_yaxes(showgrid=True, gridcolor=grid_c)

    # KPIs prévision
    kpis_pred = {
        "total_2026": f"{p_2026.sum():,.0f} €",
        "total_2027": f"{p_2027.sum():,.0f} €",
        "max_2026":   noms[int(np.argmax(p_2026))],
        "max_2027":   noms[int(np.argmax(p_2027))],
        "laf_2026":   f"{laf_26.sum():,.0f} €",
        "tcc_2026":   f"{tcc_26.sum():,.0f} €",
    }

    table_2026 = [{"mois": noms[i], "prevu": f"{p_2026[i]:,.0f} €",
                   "laf": f"{laf_26[i]:,.0f} €", "tcc": f"{tcc_26[i]:,.0f} €"} for i in range(12)]
    table_2027 = [{"mois": noms[i], "prevu": f"{p_2027[i]:,.0f} €",
                   "laf": f"{laf_27[i]:,.0f} €", "tcc": f"{tcc_27[i]:,.0f} €"} for i in range(12)]

    return render_template("prediction.html",
        figA=figA.to_html(full_html=False, include_plotlyjs=False),
        figB=figB.to_html(full_html=False, include_plotlyjs=False),
        figC=figC.to_html(full_html=False, include_plotlyjs=False),
        figD=figD.to_html(full_html=False, include_plotlyjs=False),
        kpis_pred=kpis_pred, table_2026=table_2026, table_2027=table_2027,
        dark=dark, user_name=session.get("name"), user_role=session.get("role"))

if __name__ == "__main__":
    app.run(debug=True, port=5050)
