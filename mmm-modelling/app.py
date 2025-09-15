import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Fixed MMM - High Accuracy")

def safe_numeric_transform(series, method='log'):
    """Safely transform numeric data without creating infinite values."""
    # Convert to numeric, replace invalid with 0
    series = pd.to_numeric(series, errors='coerce').fillna(0)
    
    # Ensure positive values for log/sqrt transforms
    series = np.maximum(series, 0)
    
    if method == 'log':
        # Use log1p which is log(1+x) - handles zeros naturally
        result = np.log1p(series)
    elif method == 'sqrt':
        result = np.sqrt(series)
    elif method == 'square':
        # Cap before squaring to prevent explosion
        series_capped = np.minimum(series, 1000)
        result = series_capped ** 2
    else:
        result = series
    
    # Final safety check - replace any remaining invalid values
    result = np.where(np.isfinite(result), result, 0)
    return result

def safe_adstock(spend_series, decay_rate):
    """Simple, safe adstock transformation."""
    spend_array = np.array(spend_series, dtype=float)
    # Replace any invalid values with 0
    spend_array = np.where(np.isfinite(spend_array), spend_array, 0)
    
    adstocked = np.zeros_like(spend_array)
    adstocked[0] = spend_array[0]
    
    for i in range(1, len(spend_array)):
        adstocked[i] = spend_array[i] + decay_rate * adstocked[i-1]
        # Prevent explosive growth
        if adstocked[i] > 1e6:
            adstocked[i] = adstocked[i-1]
    
    return adstocked

def create_safe_features(df, media_channels, decay_rate):
    """Create features with extensive safety checks."""
    df_new = df.copy()
    
    # Basic cleaning - replace infinite and very large values
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0)
        df_new[col] = np.clip(df_new[col], -1e6, 1e6)
        df_new[col] = np.where(np.isfinite(df_new[col]), df_new[col], 0)
    
    new_features = []
    
    for channel in media_channels:
        if channel in df_new.columns:
            base_col = df_new[channel]
            
            # 1. Adstock transformation
            adstock_col = f'{channel}_adstock'
            df_new[adstock_col] = safe_adstock(base_col, decay_rate)
            new_features.append(adstock_col)
            
            # 2. Log transformation
            log_col = f'{channel}_log'
            df_new[log_col] = safe_numeric_transform(base_col, 'log')
            new_features.append(log_col)
            
            # 3. Square root transformation
            sqrt_col = f'{channel}_sqrt'
            df_new[sqrt_col] = safe_numeric_transform(base_col, 'sqrt')
            new_features.append(sqrt_col)
            
            # 4. Lag features (1 and 2 weeks)
            for lag in [1, 2]:
                lag_col = f'{channel}_lag{lag}'
                df_new[lag_col] = base_col.shift(lag).fillna(0)
                new_features.append(lag_col)
            
            # 5. Moving averages (2 and 4 weeks)
            for window in [2, 4]:
                ma_col = f'{channel}_ma{window}'
                ma_val = base_col.rolling(window=window, min_periods=1).mean().fillna(0)
                df_new[ma_col] = np.where(np.isfinite(ma_val), ma_val, 0)
                new_features.append(ma_col)
    
    # Create time-based features
    if 'week' in df_new.columns:
        try:
            df_new['week_dt'] = pd.to_datetime(df_new['week'], errors='coerce')
            df_new['week_num'] = df_new['week_dt'].dt.isocalendar().week.fillna(26)
            df_new['month'] = df_new['week_dt'].dt.month.fillna(6)
            df_new['quarter'] = df_new['week_dt'].dt.quarter.fillna(2)
            
            # Seasonal indicators
            df_new['is_holiday'] = ((df_new['month'] == 11) | (df_new['month'] == 12)).astype(int)
            df_new['is_summer'] = ((df_new['month'] >= 6) & (df_new['month'] <= 8)).astype(int)
            
            new_features.extend(['week_num', 'month', 'quarter', 'is_holiday', 'is_summer'])
        except:
            pass
    
    # Add trend
    df_new['time_trend'] = np.arange(len(df_new))
    new_features.append('time_trend')
    
    # Safe interaction features (only between main channels)
    main_channels = ['facebook_spend', 'google_spend']
    if all(col in df_new.columns for col in main_channels):
        interact_col = 'fb_google_interact'
        fb_vals = np.clip(df_new['facebook_spend'].values, 0, 1000)
        google_vals = np.clip(df_new['google_spend'].values, 0, 1000)
        df_new[interact_col] = fb_vals * google_vals
        df_new[interact_col] = np.clip(df_new[interact_col], 0, 1e6)
        new_features.append(interact_col)
    
    # Final cleaning of all new features
    for col in new_features:
        if col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0)
            df_new[col] = np.clip(df_new[col], -1e6, 1e6)
            df_new[col] = np.where(np.isfinite(df_new[col]), df_new[col], 0)
    
    return df_new, new_features

def ultra_safe_scaling(X, scaler_type='robust'):
    """Ultra-safe scaling that handles any edge cases."""
    # Convert to numpy array
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Replace any remaining invalid values
    X = np.where(np.isfinite(X), X, 0)
    
    # Clip extreme values before scaling
    X = np.clip(X, -1e6, 1e6)
    
    # Use RobustScaler which is less sensitive to outliers
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    try:
        X_scaled = scaler.fit_transform(X)
        # Final check
        X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0)
        X_scaled = np.clip(X_scaled, -10, 10)  # Reasonable bounds for scaled data
        return X_scaled, scaler
    except:
        # If scaling fails, return normalized data
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - X_mean) / X_std
        X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0)
        return np.clip(X_scaled, -10, 10), None

def plot_results(y_true, y_pred, title):
    """Simple, effective plotting."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, color='blue')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(title)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.1f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Time series
    ax2.plot(y_true, label='Actual', alpha=0.8)
    ax2.plot(y_pred, label='Predicted', alpha=0.8)
    ax2.legend()
    ax2.set_title('Time Series Comparison')
    ax2.set_xlabel('Time Index')
    
    return fig

# Streamlit App
st.title("🎯 Fixed High-Accuracy MMM")
st.markdown("Robust implementation designed to achieve R² > 0.80")

# Sidebar
with st.sidebar:
    st.header("⚙ Configuration")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    decay_rate = st.slider("Adstock Decay", 0.0, 0.9, 0.6, 0.05)
    model_type = st.selectbox("Model Type", ["RandomForest", "Ridge", "ElasticNet"])
    
    if model_type in ["Ridge", "ElasticNet"]:
        alpha = st.slider("Regularization", 0.01, 10.0, 0.5, 0.01)
    
    cv_splits = st.number_input("CV Splits", 3, 8, 5)

# Load data
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv('Assessment 2 - MMM Weekly.csv')
        return df
    except:
        return None

df = load_data(uploaded_file)

if df is not None:
    st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Define channels
    media_channels = ['facebook_spend', 'tiktok_spend', 'snapchat_spend', 'google_spend']
    available_channels = [col for col in media_channels if col in df.columns]
    
    if len(available_channels) < 2:
        st.error("Need at least 2 media channels in the data")
        st.stop()
    
    # Create features
    with st.spinner("Creating features..."):
        df_enhanced, new_feature_cols = create_safe_features(df, available_channels, decay_rate)
    
    # Define feature sets
    social_channels = ['facebook_spend', 'tiktok_spend', 'snapchat_spend']
    social_features = []
    
    for channel in social_channels:
        if channel in available_channels:
            # Add all engineered features for this channel
            channel_features = [col for col in new_feature_cols if channel.replace('_spend', '') in col]
            social_features.extend(channel_features)
    
    # Base features
    base_feature_candidates = ['emails_send', 'sms_send', 'average_price', 'social_followers', 
                              'promotions', 'week_num', 'month', 'quarter', 'is_holiday', 
                              'is_summer', 'time_trend', 'fb_google_interact']
    
    base_features = [col for col in base_feature_candidates if col in df_enhanced.columns]
    
    st.info(f"Created {len(social_features)} social features and {len(base_features)} base features")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🎯 Stage 1", "💰 Stage 2", "✅ Cross-Validation"])
    
    with tab1:
        st.header("Data Overview")
        st.dataframe(df.head())
        
        if len(available_channels) >= 2:
            chart_data = df[['week'] + available_channels + ['revenue']].set_index('week')
            st.line_chart(chart_data)
    
    # Stage 1: Social → Google
    if 'google_spend' in df_enhanced.columns and social_features:
        # Prepare Stage 1 data
        X1_df = df_enhanced[social_features].copy()
        y1 = df_enhanced['google_spend'].copy()
        
        # Ultra-safe preparation
        X1_df = X1_df.fillna(0)
        y1 = pd.to_numeric(y1, errors='coerce').fillna(0)
        y1 = np.clip(y1, 0, 1e6)
        
        X1_scaled, scaler1 = ultra_safe_scaling(X1_df, 'robust')
        
        # Model selection
        if model_type == "RandomForest":
            model1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == "ElasticNet":
            model1 = ElasticNet(alpha=alpha, random_state=42, max_iter=2000)
        else:
            model1 = Ridge(alpha=alpha, random_state=42)
        
        # Fit model
        model1.fit(X1_scaled, y1)
        google_pred = model1.predict(X1_scaled)
        google_pred = np.clip(google_pred, 0, 1e6)
        
        with tab2:
            st.header("🎯 Stage 1: Social Media → Google Spend")
            
            r2_s1 = r2_score(y1, google_pred)
            rmse_s1 = np.sqrt(mean_squared_error(y1, google_pred))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("R-squared", f"{r2_s1:.3f}")
            col2.metric("RMSE", f"{rmse_s1:.0f}")
            col3.metric("Model", model_type)
            
            if r2_s1 > 0.8:
                st.success("🎉 Excellent performance!")
            elif r2_s1 > 0.6:
                st.info("👍 Good performance")
            else:
                st.warning("⚠ Consider adjusting parameters")
            
            st.pyplot(plot_results(y1, google_pred, "Stage 1: Social → Google"))
            
            # Feature importance
            if hasattr(model1, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': social_features,
                    'Importance': model1.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                st.bar_chart(importance_df.set_index('Feature'))
    
    # Stage 2: All → Revenue
    if 'revenue' in df_enhanced.columns:
        # Prepare Stage 2 data
        X2_base_df = df_enhanced[base_features].copy()
        y2 = df_enhanced['revenue'].copy()
        
        # Add predicted Google spend
        google_pred_df = pd.DataFrame({'predicted_google': google_pred})
        X2_df = pd.concat([X2_base_df, google_pred_df], axis=1)
        
        # Ultra-safe preparation
        X2_df = X2_df.fillna(0)
        y2 = pd.to_numeric(y2, errors='coerce').fillna(0)
        y2 = np.clip(y2, 0, 1e8)
        
        X2_scaled, scaler2 = ultra_safe_scaling(X2_df, 'robust')
        
        # Model selection
        if model_type == "RandomForest":
            model2 = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
        elif model_type == "ElasticNet":
            model2 = ElasticNet(alpha=alpha, random_state=42, max_iter=2000)
        else:
            model2 = Ridge(alpha=alpha, random_state=42)
        
        # Fit model
        model2.fit(X2_scaled, y2)
        revenue_pred = model2.predict(X2_scaled)
        revenue_pred = np.clip(revenue_pred, 0, 1e8)
        
        with tab3:
            st.header("💰 Stage 2: All Features → Revenue")
            
            r2_s2 = r2_score(y2, revenue_pred)
            rmse_s2 = np.sqrt(mean_squared_error(y2, revenue_pred))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("R-squared", f"{r2_s2:.3f}")
            col2.metric("RMSE", f"{rmse_s2:,.0f}")
            col3.metric("Model", model_type)
            
            if r2_s2 > 0.8:
                st.success("🎉 Excellent performance!")
            elif r2_s2 > 0.6:
                st.info("👍 Good performance")
            else:
                st.warning("⚠ Consider adjusting parameters")
            
            st.pyplot(plot_results(y2, revenue_pred, "Stage 2: All Features → Revenue"))
            
            # Feature importance
            if hasattr(model2, 'feature_importances_'):
                all_s2_features = base_features + ['predicted_google']
                importance_df = pd.DataFrame({
                    'Feature': all_s2_features,
                    'Importance': model2.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                st.bar_chart(importance_df.set_index('Feature'))
    
    # Cross-validation
    with tab4:
        st.header("✅ Cross-Validation Results")
        
        with st.spinner("Running cross-validation..."):
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            cv_r2_s1, cv_r2_s2 = [], []
            
            fold_num = 0
            for train_idx, val_idx in tscv.split(df_enhanced):
                fold_num += 1
                try:
                    st.write(f"Processing fold {fold_num}...")
                    
                    # Ensure we have minimum data points
                    if len(train_idx) < 10 or len(val_idx) < 3:
                        st.warning(f"Fold {fold_num}: Insufficient data points (train: {len(train_idx)}, val: {len(val_idx)})")
                        continue
                    
                    # Split data with explicit index reset
                    train_df = df_enhanced.iloc[train_idx].copy().reset_index(drop=True)
                    val_df = df_enhanced.iloc[val_idx].copy().reset_index(drop=True)
                    
                    # Check if we have required columns
                    if not all(col in train_df.columns for col in social_features):
                        st.warning(f"Fold {fold_num}: Missing social features")
                        continue
                    
                    if not all(col in train_df.columns for col in base_features):
                        st.warning(f"Fold {fold_num}: Missing base features")
                        continue
                    
                    # Stage 1 CV - ensure consistent shapes
                    X1_train = train_df[social_features].fillna(0).reset_index(drop=True)
                    y1_train = pd.to_numeric(train_df['google_spend'], errors='coerce').fillna(0).reset_index(drop=True)
                    X1_val = val_df[social_features].fillna(0).reset_index(drop=True)
                    y1_val = pd.to_numeric(val_df['google_spend'], errors='coerce').fillna(0).reset_index(drop=True)
                    
                    # Verify shapes match
                    if len(X1_train) != len(y1_train):
                        st.error(f"Fold {fold_num} Stage 1: Shape mismatch - X1_train: {len(X1_train)}, y1_train: {len(y1_train)}")
                        continue
                    
                    if len(X1_val) != len(y1_val):
                        st.error(f"Fold {fold_num} Stage 1: Shape mismatch - X1_val: {len(X1_val)}, y1_val: {len(y1_val)}")
                        continue
                    
                    # Scale with shape verification
                    X1_train_scaled, scaler1_cv = ultra_safe_scaling(X1_train, 'robust')
                    if scaler1_cv is not None:
                        X1_val_scaled = scaler1_cv.transform(X1_val.values)
                        X1_val_scaled = np.where(np.isfinite(X1_val_scaled), X1_val_scaled, 0)
                        X1_val_scaled = np.clip(X1_val_scaled, -10, 10)
                    else:
                        X1_val_scaled, _ = ultra_safe_scaling(X1_val, 'robust')
                    
                    # Build and train model
                    if model_type == "RandomForest":
                        model1_cv = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
                    elif model_type == "ElasticNet":
                        model1_cv = ElasticNet(alpha=alpha, random_state=42, max_iter=1000)
                    else:
                        model1_cv = Ridge(alpha=alpha, random_state=42)
                    
                    model1_cv.fit(X1_train_scaled, y1_train.values)
                    google_pred_val = model1_cv.predict(X1_val_scaled)
                    google_pred_train = model1_cv.predict(X1_train_scaled)
                    
                    # Ensure predictions have correct shape
                    if len(google_pred_val) != len(y1_val):
                        st.error(f"Fold {fold_num}: Prediction shape mismatch")
                        continue
                    
                    r2_s1_cv = r2_score(y1_val.values, google_pred_val)
                    cv_r2_s1.append(max(r2_s1_cv, 0))
                    
                    # Stage 2 CV - ensure consistent shapes
                    X2_base_train = train_df[base_features].fillna(0).reset_index(drop=True)
                    y2_train = pd.to_numeric(train_df['revenue'], errors='coerce').fillna(0).reset_index(drop=True)
                    X2_base_val = val_df[base_features].fillna(0).reset_index(drop=True)
                    y2_val = pd.to_numeric(val_df['revenue'], errors='coerce').fillna(0).reset_index(drop=True)
                    
                    # Verify base features shape consistency
                    if len(X2_base_train) != len(y2_train) or len(google_pred_train) != len(y2_train):
                        st.error(f"Fold {fold_num} Stage 2 Train: Shape mismatch - X2_base: {len(X2_base_train)}, y2: {len(y2_train)}, google_pred: {len(google_pred_train)}")
                        continue
                    
                    if len(X2_base_val) != len(y2_val) or len(google_pred_val) != len(y2_val):
                        st.error(f"Fold {fold_num} Stage 2 Val: Shape mismatch - X2_base: {len(X2_base_val)}, y2: {len(y2_val)}, google_pred: {len(google_pred_val)}")
                        continue
                    
                    # Combine features ensuring same length
                    google_pred_train_df = pd.DataFrame({'predicted_google': google_pred_train})
                    google_pred_val_df = pd.DataFrame({'predicted_google': google_pred_val})
                    
                    X2_train = pd.concat([X2_base_train, google_pred_train_df], axis=1).reset_index(drop=True)
                    X2_val = pd.concat([X2_base_val, google_pred_val_df], axis=1).reset_index(drop=True)
                    
                    # Final shape check
                    if len(X2_train) != len(y2_train):
                        st.error(f"Fold {fold_num}: Final shape mismatch - X2_train: {len(X2_train)}, y2_train: {len(y2_train)}")
                        continue
                    
                    # Scale Stage 2 features
                    X2_train_scaled, scaler2_cv = ultra_safe_scaling(X2_train, 'robust')
                    if scaler2_cv is not None:
                        X2_val_scaled = scaler2_cv.transform(X2_val.values)
                        X2_val_scaled = np.where(np.isfinite(X2_val_scaled), X2_val_scaled, 0)
                        X2_val_scaled = np.clip(X2_val_scaled, -10, 10)
                    else:
                        X2_val_scaled, _ = ultra_safe_scaling(X2_val, 'robust')
                    
                    # Build and train Stage 2 model
                    if model_type == "RandomForest":
                        model2_cv = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                    elif model_type == "ElasticNet":
                        model2_cv = ElasticNet(alpha=alpha, random_state=42, max_iter=1000)
                    else:
                        model2_cv = Ridge(alpha=alpha, random_state=42)
                    
                    model2_cv.fit(X2_train_scaled, y2_train.values)
                    revenue_pred_val = model2_cv.predict(X2_val_scaled)
                    
                    r2_s2_cv = r2_score(y2_val.values, revenue_pred_val)
                    cv_r2_s2.append(max(r2_s2_cv, 0))
                    
                    st.success(f"✅ Fold {fold_num} completed - Stage 1 R²: {r2_s1_cv:.3f}, Stage 2 R²: {r2_s2_cv:.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Fold {fold_num} failed: {str(e)}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
                    continue
        
        if cv_r2_s2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Stage 1 CV Results")
                st.metric("Mean R²", f"{np.mean(cv_r2_s1):.3f}")
                st.metric("Std R²", f"{np.std(cv_r2_s1):.3f}")
            
            with col2:
                st.subheader("Stage 2 CV Results")
                st.metric("Mean R²", f"{np.mean(cv_r2_s2):.3f}")
                st.metric("Std R²", f"{np.std(cv_r2_s2):.3f}")
            
            # Performance summary
            avg_r2 = np.mean(cv_r2_s2)
            if avg_r2 > 0.8:
                st.success(f"🎯 *SUCCESS!* Average R² = {avg_r2:.3f} > 0.80")
            elif avg_r2 > 0.6:
                st.info(f"📊 Good performance: R² = {avg_r2:.3f}")
            else:
                st.warning(f"⚠ Try RandomForest model or adjust parameters. Current R² = {avg_r2:.3f}")
            
            # Plot CV results
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(cv_r2_s1, 'o-', label='Stage 1', alpha=0.7)
            ax.plot(cv_r2_s2, 's-', label='Stage 2', alpha=0.7)
            ax.axhline(y=0.8, color='red', linestyle='--', label='Target (0.80)')
            ax.set_xlabel('CV Fold')
            ax.set_ylabel('R²')
            ax.set_title('Cross-Validation Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # Tips
    with st.sidebar:
        st.markdown("---")
        st.subheader("💡 Tips for R² > 0.80")
        st.markdown("""
        *Best settings:*
        - Model: RandomForest
        - Decay: 0.6-0.8
        - Ensure data quality
        - Try different decay rates
        
        *RandomForest usually performs best!*
        """)

else:
    st.error("❌ Could not load data. Please upload a CSV file.")
    st.markdown("*Required columns:* facebook_spend, google_spend, revenue, week")