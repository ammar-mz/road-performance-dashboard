import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Road Performance Dashboard",
    page_icon=" road",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== LOAD DATA & MODEL ========================
@st.cache_data
def load_data():
    df = pd.read_excel('DATA JALAN PER WEEK.xlsx', usecols='B:N', engine='openpyxl')
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df.groupby('Road Segment')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())
    df['WEEK'] = df['WEEK'].astype(str)
    df['Speed_Ratio'] = df['Act. Speed Per Segment'] / df['Plan Speed Per Segment']
    df['Speed_Deviation'] = df['Act. Speed Per Segment'] - df['Plan Speed Per Segment']
    df['Grade_Category'] = pd.cut(df['Grade'], bins=[0, 0.03, 0.06, 0.10, 1.0],
                                   labels=['Flat', 'Medium', 'Steep', 'Very Steep'])
    return df

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, 'best_road_speed_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'road_speed_scaler.pkl'))
    features = joblib.load(os.path.join(BASE_DIR, 'road_speed_features.pkl'))
    return model, scaler, features

df = load_data()
model, scaler, features = load_model()

# ======================== SIDEBAR ========================
st.sidebar.title("Road Performance Dashboard")
page = st.sidebar.radio("Navigation", [
    "Overview",
    "EDA Analysis",
    "Model Comparison",
    "Feature Importance",
    "Interactive Prediction",
    "Data Explorer"
])

# ======================== OVERVIEW ========================
if page == "Overview":
    st.title("Road Performance Analysis & Prediction")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Segments", f"{df['Road Segment'].nunique()}")
    with col2:
        st.metric("Avg Actual Speed", f"{df['Act. Speed Per Segment'].mean():.1f} km/h")
    with col3:
        st.metric("Avg Plan Speed", f"{df['Plan Speed Per Segment'].mean():.1f} km/h")
    with col4:
        avg_dev = df['Speed_Deviation'].mean()
        st.metric("Avg Deviation", f"{avg_dev:+.1f} km/h",
                   delta=f"{avg_dev/df['Plan Speed Per Segment'].mean()*100:+.1f}%")

    st.markdown("---")
    st.subheader("Speed Trend by Week")
    weekly = df.groupby('WEEK').agg({
        'Act. Speed Per Segment': 'mean',
        'Plan Speed Per Segment': 'mean',
        'Count Cycle': 'sum'
    }).reset_index()
    weekly.columns = ['Week', 'Actual Speed', 'Plan Speed', 'Total Cycles']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=weekly['Week'], y=weekly['Actual Speed'],
                             mode='lines+markers', name='Actual Speed',
                             line=dict(color='#1f77b4', width=3)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=weekly['Week'], y=weekly['Plan Speed'],
                             mode='lines+markers', name='Plan Speed',
                             line=dict(color='#d62728', width=2, dash='dash')),
                  secondary_y=False)
    fig.add_trace(go.Bar(x=weekly['Week'], y=weekly['Total Cycles'],
                         name='Total Cycles', marker_color='rgba(100,100,200,0.3)',
                         width=0.4),
                  secondary_y=True)
    fig.update_layout(height=450, title="Weekly Speed Trend vs Cycle Count",
                      hovermode='x unified')
    fig.update_yaxes(title_text="Speed (km/h)", secondary_y=False)
    fig.update_yaxes(title_text="Cycle Count", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Road Segment Speed Distribution")
        fig = px.histogram(df, x='Act. Speed Per Segment', nbins=40,
                           color_discrete_sequence=['#1f77b4'],
                           title='Distribution of Actual Speed')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Grade Category Distribution")
        grade_counts = df['Grade_Category'].value_counts().reset_index()
        grade_counts.columns = ['Grade Category', 'Count']
        fig = px.pie(grade_counts, values='Count', names='Grade Category',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ======================== EDA ANALYSIS ========================
elif page == "EDA Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    st.subheader("Correlation Heatmap")
    num_cols = ['Count Cycle', 'Avg. Truck Per Hour', 'Average of DISTANCE_METER',
                'Act. Speed Per Segment', 'Plan Speed Per Segment', 'Grade',
                'Crossfall', 'Lebar Jalan', 'HRSI', 'Sudut Jalan']
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, aspect='auto')
    fig.update_layout(height=600, title='Feature Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Actual vs Plan Speed")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x='Plan Speed Per Segment', y='Act. Speed Per Segment',
                         color='Grade_Category', opacity=0.5,
                         title='Plan vs Actual Speed by Grade Category',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines',
                                 line=dict(color='red', dash='dash', width=2),
                                 name='Perfect'))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='Speed_Deviation', nbins=50,
                           color_discrete_sequence=['#2ca02c'],
                           title='Speed Deviation Distribution')
        fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top & Bottom Road Segments")
    road_stats = df.groupby('Road Segment').agg({
        'Act. Speed Per Segment': 'mean',
        'Plan Speed Per Segment': 'mean',
        'Speed_Deviation': 'mean',
        'Count Cycle': 'sum'
    }).reset_index()
    road_stats.columns = ['Road Segment', 'Avg Actual', 'Avg Plan', 'Avg Deviation', 'Total Cycles']
    road_stats['Rank'] = road_stats['Avg Actual'].rank(ascending=False).astype(int)

    col1, col2 = st.columns(2)
    with col1:
        top10 = road_stats.nlargest(10, 'Avg Actual')
        fig = px.bar(top10, x='Avg Actual', y='Road Segment', orientation='h',
                     color_discrete_sequence=['#2ca02c'], title='Top 10 Fastest Segments')
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        bot10 = road_stats.nsmallest(10, 'Avg Actual')
        fig = px.bar(bot10, x='Avg Actual', y='Road Segment', orientation='h',
                     color_discrete_sequence=['#d62728'], title='Top 10 Slowest Segments')
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature vs Speed")
    feat_opts = ['Grade', 'Crossfall', 'Lebar Jalan', 'HRSI', 'Sudut Jalan',
                 'Average of DISTANCE_METER', 'Avg. Truck Per Hour']
    selected_feat = st.selectbox("Select Feature", feat_opts)
    fig = px.scatter(df, x=selected_feat, y='Act. Speed Per Segment',
                     color='Grade_Category', opacity=0.4,
                     title=f'{selected_feat} vs Actual Speed',
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ======================== MODEL COMPARISON ========================
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("---")

    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import xgboost as xgb

    X = df[features].dropna()
    y = df.loc[X.index, 'Act. Speed Per Segment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_new = StandardScaler()
    X_train_scaled = scaler_new.fit_transform(X_train)
    X_test_scaled = scaler_new.transform(X_test)

    param_grids = {
        'Ridge': {'alpha': [0.01, 0.1, 1.0, 10.0]},
        'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5]},
        'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}
    }
    base_models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0)
    }

    results = {}
    best_models = {}
    progress = st.progress(0, text="Training models...")

    for idx, (name, model_obj) in enumerate(base_models.items()):
        progress.progress((idx + 1) / len(base_models), text=f"Training {name}...")
        try:
            if name in param_grids:
                gs = GridSearchCV(model_obj, param_grids[name], cv=5, scoring='r2', n_jobs=1)
                if name in ['Ridge', 'Lasso']:
                    gs.fit(X_train_scaled, y_train)
                else:
                    gs.fit(X_train, y_train)
                best_models[name] = gs.best_estimator_
                best_params = gs.best_params_
            else:
                model_obj.fit(X_train_scaled, y_train)
                best_models[name] = model_obj
                best_params = 'default'

            if name in ['Linear Regression', 'Ridge', 'Lasso']:
                y_pred = best_models[name].predict(X_test_scaled)
                cv_scores = cross_val_score(best_models[name], X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                y_pred = best_models[name].predict(X_test)
                cv_scores = cross_val_score(best_models[name], X_train, y_train, cv=5, scoring='r2')

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[name] = {
                'R2': r2, 'MAE': mae, 'RMSE': rmse,
                'CV_R2_Mean': cv_scores.mean(), 'CV_R2_Std': cv_scores.std(),
                'Best Params': best_params, 'y_pred': y_pred
            }
        except Exception as e:
            st.error(f"Error training {name}: {e}")

    progress.empty()

    res_df = pd.DataFrame(results).T[['R2', 'MAE', 'RMSE', 'CV_R2_Mean', 'CV_R2_Std', 'Best Params']]
    res_df = res_df.sort_values('R2', ascending=False)

    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

    with col1:
        fig = px.bar(x=res_df.index, y=res_df['R2'], color=res_df.index,
                     color_discrete_sequence=colors, title='R² Score')
        fig.update_layout(height=350, showlegend=False, yaxis_title='R²', yaxis_range=[0, 1])
        for i, v in enumerate(res_df['R2']):
            fig.add_annotation(x=res_df.index[i], y=v, text=f'{v:.4f}',
                               showarrow=False, yshift=10, font=dict(size=11, color='black'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(x=res_df.index, y=res_df['MAE'], color=res_df.index,
                     color_discrete_sequence=colors, title='MAE (km/h)')
        fig.update_layout(height=350, showlegend=False, yaxis_title='MAE')
        for i, v in enumerate(res_df['MAE']):
            fig.add_annotation(x=res_df.index[i], y=v, text=f'{v:.2f}',
                               showarrow=False, yshift=10, font=dict(size=11, color='black'))
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.bar(x=res_df.index, y=res_df['RMSE'], color=res_df.index,
                     color_discrete_sequence=colors, title='RMSE (km/h)')
        fig.update_layout(height=350, showlegend=False, yaxis_title='RMSE')
        for i, v in enumerate(res_df['RMSE']):
            fig.add_annotation(x=res_df.index[i], y=v, text=f'{v:.2f}',
                               showarrow=False, yshift=10, font=dict(size=11, color='black'))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cross-Validation R² (5-Fold)")
    cv_data = pd.DataFrame({
        'Model': res_df.index,
        'CV R² Mean': res_df['CV_R2_Mean'].values,
        'CV R² Std': res_df['CV_R2_Std'].values
    }).sort_values('CV R² Mean', ascending=True)
    fig = px.bar(cv_data, x='CV R² Mean', y='Model', orientation='h',
                 color_discrete_sequence=colors, title='CV R² with Error Bars')
    fig.add_trace(go.Scatter(x=cv_data['CV R² Mean'] + cv_data['CV R² Std'],
                             y=cv_data['Model'], mode='markers',
                             marker=dict(color='rgba(0,0,0,0.3)', size=6),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=cv_data['CV R² Mean'] - cv_data['CV R² Std'],
                             y=cv_data['Model'], mode='markers',
                             marker=dict(color='rgba(0,0,0,0.3)', size=6),
                             showlegend=False))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Best Model - Detailed Analysis")
    best_name = res_df.index[0]
    best = best_models[best_name]
    y_pred_best = results[best_name]['y_pred']

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.scatter(x=y_test, y=y_pred_best, opacity=0.5,
                         title=f'Actual vs Predicted ({best_name})',
                         labels={'x': 'Actual', 'y': 'Predicted'})
        lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
                                 line=dict(color='red', dash='dash', width=2),
                                 name='Perfect'))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        residuals = y_test.values - y_pred_best
        fig = px.histogram(x=residuals, nbins=40, title='Residual Distribution',
                           color_discrete_sequence=['#1f77b4'])
        fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        st.metric("R² Score", f"{results[best_name]['R2']:.4f}")
        st.metric("MAE", f"{results[best_name]['MAE']:.2f} km/h")
        st.metric("RMSE", f"{results[best_name]['RMSE']:.2f} km/h")
        st.metric("CV R²", f"{results[best_name]['CV_R2_Mean']:.4f} +/- {results[best_name]['CV_R2_Std']:.4f}")
        st.write(f"**Best Params:** {results[best_name]['Best Params']}")

    st.subheader("Comparison Table")
    st.dataframe(res_df.drop(columns='Best Params').style.format({
        'R2': '{:.4f}', 'MAE': '{:.2f}', 'RMSE': '{:.2f}',
        'CV_R2_Mean': '{:.4f}', 'CV_R2_Std': '{:.4f}'
    }), use_container_width=True)

# ======================== FEATURE IMPORTANCE ========================
elif page == "Feature Importance":
    st.title("Feature Importance Analysis")
    st.markdown("---")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    import xgboost as xgb

    X = df[features].dropna()
    y = df.loc[X.index, 'Act. Speed Per Segment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_new = StandardScaler()
    X_train_scaled = scaler_new.fit_transform(X_train)

    lr = LinearRegression().fit(X_train_scaled, y_train)
    ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
    lasso = Lasso(alpha=0.01).fit(X_train_scaled, y_train)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42).fit(X_train, y_train)
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   random_state=42, verbosity=0).fit(X_train, y_train)

    feat_imp = pd.DataFrame(index=features)
    for name, m in [('Linear Regression', lr), ('Ridge', ridge), ('Lasso', lasso)]:
        coefs = np.abs(m.coef_)
        feat_imp[name] = coefs / coefs.sum() * 100
    feat_imp['Random Forest'] = rf.feature_importances_ / rf.feature_importances_.sum() * 100
    feat_imp['XGBoost'] = xgb_model.feature_importances_ / xgb_model.feature_importances_.sum() * 100
    feat_imp['Average'] = feat_imp.mean(axis=1)
    feat_imp = feat_imp.sort_values('Average', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.imshow(feat_imp.drop(columns='Average'), text_auto='.1f',
                        color_continuous_scale='YlOrRd', aspect='auto',
                        title='Feature Importance per Model (%)')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        feat_sorted = feat_imp.sort_values('Average', ascending=True)
        fig = px.bar(x=feat_sorted['Average'], y=feat_sorted.index, orientation='h',
                     color=feat_sorted['Average'], color_continuous_scale='RdYlGn_r',
                     title='Average Feature Importance (%)')
        fig.update_layout(height=450, xaxis_title='Importance (%)', showlegend=False)
        for i, v in enumerate(feat_sorted['Average']):
            fig.add_annotation(x=v + 0.5, y=feat_sorted.index[i], text=f'{v:.1f}%',
                               showarrow=False, font=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mutual Information Scores")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': features, 'MI Score': mi_scores}).sort_values('MI Score', ascending=True)
    fig = px.bar(x=mi_df['MI Score'], y=mi_df['Feature'], orientation='h',
                 color=mi_df['MI Score'], color_continuous_scale='viridis',
                 title='Mutual Information Regression Scores')
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"Top features based on average importance: **{', '.join(feat_imp.head(3).index.tolist())}**")

# ======================== INTERACTIVE PREDICTION ========================
elif page == "Interactive Prediction":
    st.title("Interactive Speed Prediction")
    st.markdown("Masukkan kondisi jalan untuk memprediksi kecepatan aktual.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        grade = st.slider("Grade", 0.0, 0.30, 0.05, 0.01,
                          help="Kemiringan jalan (0 = datar, 0.30 = sangat curam)")
        crossfall = st.slider("Crossfall", -0.05, 0.10, 0.02, 0.01,
                              help="Kemiringan melintang jalan")
        lebar_jalan = st.slider("Lebar Jalan (m)", 5.0, 50.0, 25.0, 1.0)
    with col2:
        min_lebar = st.slider("Min Lebar Jalan (m)", 5.0, 50.0, 20.0, 1.0)
        hrsi = st.slider("HRSI", 0.0, 15.0, 7.0, 0.5,
                         help="Haul Road Serviceability Index")
    with col3:
        sudut_jalan = st.slider("Sudut Jalan (derajat)", 0, 180, 30, 5)
        distance = st.slider("Distance (m)", 50.0, 5000.0, 500.0, 50.0)

    if st.button("Predict Speed", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            'Grade': grade, 'Crossfall': crossfall, 'Lebar Jalan': lebar_jalan,
            'Min Lebar Jalan': min_lebar, 'HRSI': hrsi,
            'Sudut Jalan': sudut_jalan, 'Average of DISTANCE_METER': distance
        }])
        input_df = input_df[features]

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Speed", f"{prediction:.1f} km/h")
        with col2:
            avg_speed = df['Act. Speed Per Segment'].mean()
            delta_pct = (prediction - avg_speed) / avg_speed * 100
            st.metric("vs Average", f"{avg_speed:.1f} km/h",
                       delta=f"{delta_pct:+.1f}%")
        with col3:
            plan_speed = df['Plan Speed Per Segment'].mean()
            st.metric("vs Plan Avg", f"{plan_speed:.1f} km/h",
                       delta=f"{(prediction - plan_speed):+.1f} km/h")

        st.subheader("Comparison with All Models")
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        X = df[features].dropna()
        y = df.loc[X.index, 'Act. Speed Per Segment']
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler().fit(X_tr)

        model_dict = {
            'Linear Regression': LinearRegression().fit(sc.fit_transform(X_tr), y_tr),
            'Ridge': Ridge(alpha=1.0).fit(sc.fit_transform(X_tr), y_tr),
            'Lasso': Lasso(alpha=0.01).fit(sc.fit_transform(X_tr), y_tr),
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42).fit(X_tr, y_tr),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                         random_state=42, verbosity=0).fit(X_tr, y_tr),
        }

        preds = {}
        for n, m in model_dict.items():
            if n in ['Linear Regression', 'Ridge', 'Lasso']:
                preds[n] = m.predict(sc.transform(input_df))[0]
            else:
                preds[n] = m.predict(input_df)[0]

        pred_df = pd.DataFrame({'Model': preds.keys(), 'Prediction': preds.values()})
        fig = px.bar(pred_df, x='Model', y='Prediction', color='Model',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title='Prediction Across All Models')
        fig.update_layout(height=350, showlegend=False)
        fig.add_hline(y=prediction, line_dash="dash", line_color="red",
                      annotation_text=f"Best Model: {prediction:.1f}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Grade Category Reference")
    ref_df = pd.DataFrame({
        'Grade Category': ['Flat', 'Medium', 'Steep', 'Very Steep'],
        'Range': ['0 - 3%', '3 - 6%', '6 - 10%', '> 10%'],
        'Avg Speed (km/h)': [
            df[df['Grade_Category'] == 'Flat']['Act. Speed Per Segment'].mean(),
            df[df['Grade_Category'] == 'Medium']['Act. Speed Per Segment'].mean(),
            df[df['Grade_Category'] == 'Steep']['Act. Speed Per Segment'].mean(),
            df[df['Grade_Category'] == 'Very Steep']['Act. Speed Per Segment'].mean(),
        ]
    })
    st.dataframe(ref_df.style.format({'Avg Speed (km/h)': '{:.1f}'}), use_container_width=True)

# ======================== DATA EXPLORER ========================
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("---")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        weeks = sorted(df['WEEK'].unique())
        sel_week = st.multiselect("Week", weeks, default=weeks[:3])
        segments = sorted(df['Road Segment'].unique())
        sel_segment = st.multiselect("Road Segment", segments[:20], default=segments[:5])

    filtered = df.copy()
    if sel_week:
        filtered = filtered[filtered['WEEK'].isin(sel_week)]
    if sel_segment:
        filtered = filtered[filtered['Road Segment'].isin(sel_segment)]

    with col2:
        st.subheader(f"Filtered Data ({len(filtered)} rows)")
        st.dataframe(filtered, use_container_width=True)

    st.markdown("---")
    st.subheader("Summary Statistics")
    stat_cols = ['Act. Speed Per Segment', 'Plan Speed Per Segment', 'Speed_Deviation',
                 'Grade', 'Crossfall', 'Lebar Jalan', 'HRSI', 'Sudut Jalan']
    st.dataframe(filtered[stat_cols].describe().style.format('{:.2f}'), use_container_width=True)
